#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace nb = nanobind;

struct MemorySample {
  int64_t bytes;
  int64_t timestamp_ns;
};

struct DeviceInfo {
  nb::object device;
  bool should_sample;
  int64_t allocator_baseline;
};

class MemoryTracker {
public:
  MemoryTracker(std::vector<nb::object> devices, double poll_interval_sec)
      : poll_interval_ns_(static_cast<int64_t>(poll_interval_sec * 1e9)),
      running_(false), peak_logical_bytes_(0), sampled_baseline_(0) {
    nb::gil_scoped_acquire gil;
    for (auto &device : devices) {
      bool is_cpu = nb::cast<std::string>(device.attr("platform")) == "cpu";
      bool has_clear = nb::hasattr(device, "clear_memory_stats");
      bool should_sample = is_cpu || !has_clear;
      if (should_sample) {
        std::string p = nb::cast<std::string>(device.attr("platform"));
        if (std::find(sampled_platforms_.begin(), sampled_platforms_.end(), p)
            == sampled_platforms_.end()) {
          sampled_platforms_.push_back(std::move(p));
        }
      }
      devices_info_.push_back({std::move(device), should_sample, 0});
    }

    // Cache the bound `jax.live_arrays` callable once, instead of importing jax
    // on every poll tick. Also precompute, per sampled platform, whether we
    // track *every* device on that backend: if so, every array `live_arrays`
    // returns is on a tracked device and the per-array membership check (which
    // builds a Python set via `arr.devices()`) can be skipped.
    nb::module_ jax = nb::module_::import_("jax");
    live_arrays_fn_ = jax.attr("live_arrays");
    for (const auto &p : sampled_platforms_) {
      int tracked = 0;
      for (const auto &info : devices_info_) {
        if (info.should_sample &&
            nb::cast<std::string>(info.device.attr("platform")) == p) {
          ++tracked;
        }
      }
      int total = -1;
      try {
        total = static_cast<int>(nb::len(jax.attr("devices")(p)));
      } catch (...) {
      }
      platform_track_all_.push_back(total > 0 && tracked == total);
    }
  }

  ~MemoryTracker() { stop(); }

  void start() {
    if (running_.load()) {
      throw std::runtime_error("Tracker is already running");
    }

    peak_logical_bytes_.store(0);
    samples_.clear();
    started_ = true;
    stopped_ = false;
    bool has_polled_device = false;

    {
      nb::gil_scoped_acquire gil;
      for (auto &info : devices_info_) {
        if (!info.should_sample) {
          try {
            info.device.attr("clear_memory_stats")();
          } catch (...) {}
          nb::object stats = info.device.attr("memory_stats")();
          info.allocator_baseline = nb::cast<int64_t>(stats["bytes_in_use"]);
        } else {
          has_polled_device = true;
        }
      }
      // Capture the live-array baseline last and start the clock immediately
      // after, so the measured interval opens at a quiescent point and no
      // allocation can fall into the baseline once timing has begun.
      sampled_baseline_.store(read_all_tracked_live_arrays());
      start_time_ = std::chrono::steady_clock::now();
    }

    // Only spawn the polling thread when at least one device is sample-tracked.
    // Allocator-backed devices (GPU/TPU) derive their peak from the monotonic
    // `peak_bytes_in_use` stat read in peak(), so polling them would add ~1kHz
    // of GIL contention to the measured region for zero accuracy gain.
    if (has_polled_device) {
      samples_.reserve(1024);
      running_.store(true);
      worker_ = std::thread(&MemoryTracker::poll_loop, this);
    }
  }

  void stop() {
    // Idempotent: an explicit stop() followed by ~MemoryTracker (or any double
    // stop) must not overwrite stop_time_ or take a second snapshot, and a
    // never-started tracker must not touch Python state during destruction.
    if (!started_ || stopped_) return;
    stopped_ = true;
    stop_time_ = std::chrono::steady_clock::now();

    if (running_.load()) {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        running_.store(false);
      }
      cv_.notify_all();

      {
        nb::gil_scoped_release release;
        if (worker_.joinable())
          worker_.join();
      }
    }

    // Final synchronous snapshot after the caller's exit barrier. Works for
    // both polled and allocator-only trackers and keeps .samples non-empty
    // even when no polling thread ran.
    nb::gil_scoped_acquire gil;
    record_sample();
  }

  int64_t peak() const {
    // Single GIL acquisition for the whole read: the optional fresh sample and
    // the per-device memory_stats() queries below both need it.
    nb::gil_scoped_acquire gil;
    if (running_.load()) {
      const_cast<MemoryTracker *>(this)->record_sample();
    }

    // Allocator-tracked devices: sum of per-device peak_bytes_in_use above baseline.
    int64_t physical_peak = 0;
    for (const auto &info : devices_info_) {
      if (!info.should_sample) {
        try {
          nb::object stats = info.device.attr("memory_stats")();
          int64_t current_peak = nb::cast<int64_t>(stats["peak_bytes_in_use"]);
          int64_t above_baseline = current_peak - info.allocator_baseline;
          if (above_baseline > 0) {
            physical_peak += above_baseline;
          }
        } catch (...) {
        }
      }
    }

    // Polled devices (CPU): max of logical_delta across samples.
    int64_t logical_peak = peak_logical_bytes_.load(std::memory_order_relaxed);

    return physical_peak + logical_peak;
  }

  int64_t baseline() const {
    return sampled_baseline_.load();
  }

  std::vector<MemorySample> samples() const {
    std::lock_guard<std::mutex> lock(samples_mutex_);
    return samples_;
  }

  double duration_sec() const {
    return std::chrono::duration<double>(stop_time_ - start_time_).count();
  }

private:
  void poll_loop() {
    while (running_.load()) {
      {
        nb::gil_scoped_acquire gil;
        record_sample();
      }
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait_for(lock, std::chrono::nanoseconds(poll_interval_ns_),
                   [this] { return !running_.load(); });
    }
  }

  // Sum nbytes of live arrays located on any tracked `should_sample` device.
  // `jax.live_arrays(platform)` only enumerates arrays on a single backend, so
  // we query each unique tracked platform via the callable cached at
  // construction. When we track every device on a platform (`track_all`), the
  // per-array `arr.devices()` membership check is skipped entirely.
  int64_t read_all_tracked_live_arrays() {
    if (sampled_platforms_.empty()) return 0;
    int64_t total = 0;
    for (size_t i = 0; i < sampled_platforms_.size(); ++i) {
      nb::object arrays;
      try {
        arrays = live_arrays_fn_(sampled_platforms_[i]);
      } catch (...) {
        continue;
      }
      const bool track_all = platform_track_all_[i];
      for (auto arr_handle : arrays) {
        nb::object arr = nb::borrow<nb::object>(arr_handle);
        try {
          if (track_all) {
            total += nb::cast<int64_t>(arr.attr("nbytes"));
            continue;
          }
          nb::object devices = arr.attr("devices")();
          for (const auto& info : devices_info_) {
            if (!info.should_sample) continue;
            if (nb::cast<bool>(devices.attr("__contains__")(info.device))) {
              total += nb::cast<int64_t>(arr.attr("nbytes"));
              break;  // count each array once even if on several tracked devices
            }
          }
        } catch (...) {
        }
      }
    }
    return total;
  }

  void record_sample() {
    try {
      // CPU-style tracking via live_arrays (only `should_sample` devices).
      int64_t logical_delta = read_all_tracked_live_arrays() - sampled_baseline_.load();
      if (logical_delta < 0) logical_delta = 0;

      // Allocator-tracked devices (GPU) — recorded for the sample snapshot only;
      // the peak comes from `peak_bytes_in_use` in `peak()`.
      int64_t physical_delta = 0;
      for (const auto& info : devices_info_) {
          if (!info.should_sample) {
              try {
                  nb::object stats = info.device.attr("memory_stats")();
                  int64_t d = nb::cast<int64_t>(stats["bytes_in_use"]) - info.allocator_baseline;
                  if (d > 0) physical_delta += d;
              } catch (...) {}
          }
      }

      // Track max logical-only; GPU peak is the allocator's monotonic stat.
      int64_t prev = peak_logical_bytes_.load(std::memory_order_relaxed);
      while (logical_delta > prev &&
             !peak_logical_bytes_.compare_exchange_weak(prev, logical_delta,
                                                       std::memory_order_relaxed)) {
      }

      auto now = std::chrono::steady_clock::now();
      int64_t ts = std::chrono::duration_cast<std::chrono::nanoseconds>(
                       now - start_time_)
                       .count();

      std::lock_guard<std::mutex> lock(samples_mutex_);
      samples_.push_back({logical_delta + physical_delta, ts});
    } catch (...) {
    }
  }

  std::vector<DeviceInfo> devices_info_;
  std::vector<std::string> sampled_platforms_;
  nb::object live_arrays_fn_;
  std::vector<bool> platform_track_all_;
  int64_t poll_interval_ns_;
  std::atomic<bool> running_;
  bool started_ = false;
  bool stopped_ = false;
  std::atomic<int64_t> peak_logical_bytes_;
  std::atomic<int64_t> sampled_baseline_;
  std::chrono::steady_clock::time_point start_time_;

  std::thread worker_;
  std::mutex mutex_;
  std::condition_variable cv_;

  mutable std::mutex samples_mutex_;
  std::vector<MemorySample> samples_;
  std::chrono::steady_clock::time_point stop_time_;
};

class TimeTracker {
public:
  TimeTracker() = default;

  void start() { start_time_ = std::chrono::steady_clock::now(); }

  void stop() { stop_time_ = std::chrono::steady_clock::now(); }

  double duration_sec() const {
    return std::chrono::duration<double>(stop_time_ - start_time_).count();
  }

private:
  std::chrono::steady_clock::time_point start_time_;
  std::chrono::steady_clock::time_point stop_time_;
};

NB_MODULE(xla_mem_bridge, m) {
  m.doc() = "Hybrid XLA memory tracking for JAX (Polling CPU, Event GPU)";

  nb::class_<MemorySample>(m, "MemorySample")
      .def_ro("bytes", &MemorySample::bytes)
      .def_ro("timestamp_ns", &MemorySample::timestamp_ns)
      .def("__repr__", [](const MemorySample &s) {
        return "<MemorySample bytes=" + std::to_string(s.bytes) +
               " ts=" + std::to_string(s.timestamp_ns) + "ns>";
      });

  nb::class_<MemoryTracker>(m, "MemoryTracker")
      .def(nb::init<std::vector<nb::object>, double>(), nb::arg("devices"),
           nb::arg("poll_interval_sec") = 0.001)
      .def("start", &MemoryTracker::start)
      .def("stop", &MemoryTracker::stop)
      .def_prop_ro("peak", &MemoryTracker::peak)
      .def_prop_ro("baseline", &MemoryTracker::baseline)
      .def_prop_ro("samples", &MemoryTracker::samples)
      .def_prop_ro("duration", &MemoryTracker::duration_sec);

  nb::class_<TimeTracker>(m, "TimeTracker")
      .def(nb::init<>())
      .def("start", &TimeTracker::start)
      .def("stop", &TimeTracker::stop)
      .def_prop_ro("duration", &TimeTracker::duration_sec);
}
