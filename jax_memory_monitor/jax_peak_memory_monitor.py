"""Deprecated sampling-based peak memory monitor.

Polls `jax.profiler.device_memory_profile()` (pprof) from a background thread.
Works on any platform but is high-overhead and only as accurate as the sample
interval. Prefer `ResourceMonitor`.
"""
import gzip
import threading
import warnings
from typing import Any, Optional

import jax
import jax.numpy as jnp
from jax.profiler import device_memory_profile

from . import profile_pb2


def mem_diff(prof1: profile_pb2.Profile, prof2: Optional[profile_pb2.Profile] = None) -> int:
    total1 = sum(x.value[1] for x in prof1.sample)
    if prof2 is None:
        return total1
    total2 = sum(x.value[1] for x in prof2.sample)
    return total1 - total2


def decode_pprof(data: bytes) -> profile_pb2.Profile:
    if not data:
        return profile_pb2.Profile()
    try:
        decompressed_data = gzip.decompress(data)
    except (TypeError, EOFError, gzip.BadGzipFile):
        decompressed_data = data
    profile = profile_pb2.Profile()
    profile.ParseFromString(decompressed_data)
    if len(profile.sample) == 0 and not profile.string_table:
        raise ValueError("Decoded profile is empty despite non-empty input data")
    return profile


def device_memory() -> int:
    try:
        prof = decode_pprof(device_memory_profile())
        return mem_diff(prof)
    except Exception as e:
        print(f"Error: {e}\nfailed to read memory, returning 0")
        return 0


def _sync() -> None:
    try:
        jax.device_get(jnp.array(0.0)).block_until_ready()
    except Exception:
        pass


class PeakMemoryMonitor:
    def __init__(self, interval: float = 0.01, base_memory: int = 0):
        warnings.warn(
            "PeakMemoryMonitor is deprecated; prefer ResourceMonitor.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.interval = interval
        self._base_memory = base_memory
        self._peak: int = 0
        self._stop_flag = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None

    def _sample_loop(self) -> None:
        while not self._stop_flag.is_set():
            try:
                self._peak = max(self._peak, device_memory() + self._base_memory)
            except Exception as e:
                print(f"Memory monitoring error: {e}")
                self._stop_flag.set()
                return
            if self.interval:
                self._stop_flag.wait(self.interval)

    def __enter__(self) -> "PeakMemoryMonitor":
        self._peak = device_memory() + self._base_memory
        self._stop_flag.clear()
        self._monitor_thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._monitor_thread.start()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self._stop_flag.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join()
            self._monitor_thread = None
        _sync()

    @property
    def peak(self) -> int:
        return self._peak
