"""Unified resource monitor wrapping the C++ `xla_mem_bridge` extension.

Tracks wall-clock time and/or peak device memory. Memory polling runs in
native code via `MemoryTracker`:

  - GPU/TPU: exact peak via `clear_memory_stats()` + `peak_bytes_in_use`.
  - CPU: native worker thread polling `jax.live_arrays()` (catches even
    single-element arrays).

`peak` is reported as bytes above the per-device baseline at `__enter__`,
summed across all monitored devices.
"""
import warnings
from typing import Any, List, Optional, Sequence

import jax

from xla_mem_bridge import MemoryTracker, TimeTracker


class ResourceMonitor:
    def __init__(
        self,
        device: Optional[jax.Device] = None,
        devices: Optional[Sequence[jax.Device]] = None,
        time: bool = True,
        peak: bool = True,
        poll_interval: float = 0.001,
    ):
        if devices is not None:
            if isinstance(devices, (list, tuple, set)):
                self.devices: List[jax.Device] = list(devices)
            else:
                self.devices = [devices]
        elif device is not None:
            self.devices = [device]
        else:
            self.devices = list(jax.local_devices())

        self._do_time = time
        self._do_peak = peak

        self._tracker: Optional[MemoryTracker] = None
        if self._do_peak:
            for dev in self.devices:
                if not hasattr(dev, "clear_memory_stats"):
                    warnings.warn(
                        f"Device '{dev}' lacks 'clear_memory_stats'. "
                        "Peak memory will report a sampled high-water mark instead "
                        "of the exact allocator peak.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            self._tracker = MemoryTracker(self.devices, poll_interval)

        self._time_tracker: Optional[TimeTracker] = None
        if self._do_time:
            self._time_tracker = TimeTracker()

    def __enter__(self) -> "ResourceMonitor":
        jax.effects_barrier()
        if self._tracker is not None:
            self._tracker.start()
        if self._time_tracker is not None:
            self._time_tracker.start()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self._time_tracker is not None:
            self._time_tracker.stop()
        if self._tracker is not None:
            self._tracker.stop()
        jax.effects_barrier()

    @property
    def peak(self) -> int:
        if not self._do_peak:
            raise RuntimeError("Peak memory tracking was disabled (peak=False)")
        return self._tracker.peak

    @property
    def duration(self) -> float:
        if not self._do_time:
            raise RuntimeError("Time tracking was disabled (time=False)")
        return self._time_tracker.duration

    @property
    def stats(self) -> dict:
        s = {}
        if self._do_time:
            s["time"] = self.duration
        if self._do_peak:
            s["memory"] = self.peak
        return s

    @property
    def baseline(self) -> int:
        if not self._do_peak:
            raise RuntimeError("Peak memory tracking was disabled (peak=False)")
        return self._tracker.baseline

    @property
    def peak_mb(self) -> float:
        return self.peak / (1024 * 1024)

    @property
    def duration_ms(self) -> float:
        return self.duration * 1000.0

    @property
    def samples(self) -> list:
        if not self._do_peak:
            raise RuntimeError("Peak memory tracking was disabled (peak=False)")
        return self._tracker.samples
