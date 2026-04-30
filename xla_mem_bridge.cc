#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

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
      running_(false), peak_bytes_(0), sampled_baseline_(0) {
    nb::gil_scoped_acquire gil;
    for (auto &device : devices) {
      bool is_cpu = nb::cast<std::string>(device.attr("platform")) == "cpu";
      bool has_clear = nb::hasattr(device, "clear_memory_stats");
      bool should_sample = is_cpu || !has_clear;
      devices_info_.push_back({std::move(device), should_sample, 0});
    }
  }

  ~MemoryTracker() { stop(); }

  void start() {
    if (running_.load()) {
      throw std::runtime_error("Tracker is already running");
    }

    peak_bytes_.store(0);
    samples_.clear();
    start_time_ = std::chrono::steady_clock::now();

    bool needs_polling = false;
    {
      nb::gil_scoped_acquire gil;
      for (auto &info : devices_info_) {
        if (!info.should_sample) {
          try {
            info.device.attr("clear_memory_stats")();
          } catch (...) {}
          nb::object stats = info.device.attr("memory_stats")();
          info.allocator_baseline = nb::cast<int64_t>(stats["bytes_in_use"]);
        }
      }
      sampled_baseline_.store(read_all_tracked_live_arrays());
    }

    if (!devices_info_.empty()) {
      samples_.reserve(1024);
      running_.store(true);
      worker_ = std::thread(&MemoryTracker::poll_loop, this);
    }
  }

  void stop() {
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

      {
        nb::gil_scoped_acquire gil;
        record_sample();
      }
    }
  }

  int64_t peak() const {
    if (running_.load()) {
      nb::gil_scoped_acquire gil;
      const_cast<MemoryTracker *>(this)->record_sample();
    }

    int64_t total_peak = 0;
    nb::gil_scoped_acquire gil;
    for (const auto &info : devices_info_) {
      if (!info.should_sample) {
        try {
          nb::object stats = info.device.attr("memory_stats")();
          int64_t current_peak = nb::cast<int64_t>(stats["peak_bytes_in_use"]);
          int64_t above_baseline = current_peak - info.allocator_baseline;
          if (above_baseline > 0) {
            total_peak += above_baseline;
          }
        } catch (...) {
        }
      }
    }

    int64_t sampled_peak = peak_bytes_.load(std::memory_order_relaxed);

    return std::max(total_peak, sampled_peak);
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

  int64_t read_live_arrays_for_device(const nb::object& device) {
    nb::module_ jax = nb::module_::import_("jax");
    nb::object arrays = jax.attr("live_arrays")();
    int64_t total = 0;
    for (auto arr_handle : arrays) {
      nb::object arr = nb::borrow<nb::object>(arr_handle);
      try {
        nb::object devices = arr.attr("devices")();
        if (nb::cast<bool>(devices.attr("__contains__")(device))) {
          total += nb::cast<int64_t>(arr.attr("nbytes"));
        }
      } catch (...) {
      }
    }
    return total;
  }

  int64_t read_all_tracked_live_arrays() {
    nb::module_ jax = nb::module_::import_("jax");
    nb::object arrays = jax.attr("live_arrays")();
    int64_t total = 0;
    for (auto arr_handle : arrays) {
      nb::object arr = nb::borrow<nb::object>(arr_handle);
      try {
        nb::object devices = arr.attr("devices")();
        for (const auto& info : devices_info_) {
          if (nb::cast<bool>(devices.attr("__contains__")(info.device))) {
            total += nb::cast<int64_t>(arr.attr("nbytes"));
          }
        }
      } catch (...) {
      }
    }
    return total;
  }

  void record_sample() {
    try {
      int64_t logical_peak = read_all_tracked_live_arrays() - sampled_baseline_.load();

      int64_t physical_peak = 0;
      for (const auto& info : devices_info_) {
          if (!info.should_sample) {
              try {
                  nb::object stats = info.device.attr("memory_stats")();
                  physical_peak += (nb::cast<int64_t>(stats["bytes_in_use"]) - info.allocator_baseline);
              } catch (...) {}
          }
      }

      int64_t above_baseline = std::max(logical_peak, physical_peak);
      if (above_baseline < 0)
        above_baseline = 0;

      auto now = std::chrono::steady_clock::now();
      int64_t ts = std::chrono::duration_cast<std::chrono::nanoseconds>(
                       now - start_time_)
                       .count();

      int64_t prev = peak_bytes_.load(std::memory_order_relaxed);
      while (above_baseline > prev &&
             !peak_bytes_.compare_exchange_weak(prev, above_baseline,
                                                 std::memory_order_relaxed)) {
      }

      std::lock_guard<std::mutex> lock(samples_mutex_);
      samples_.push_back({above_baseline, ts});
    } catch (...) {
    }
  }

  std::vector<DeviceInfo> devices_info_;
  int64_t poll_interval_ns_;
  std::atomic<bool> running_;
  std::atomic<int64_t> peak_bytes_;
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
