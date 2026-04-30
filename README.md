# jax_memory_monitor

A lightweight JAX resource monitor (peak memory + wall-clock time) with a
native C++ polling backend.

## Overview

`ResourceMonitor` is the recommended class. Internally it wraps the
`xla_mem_bridge.MemoryTracker` C++ extension (built with nanobind), which
polls without holding the GIL longer than necessary:

- **GPU/TPU** (devices with `clear_memory_stats`): exact peak via
  `clear_memory_stats()` + `peak_bytes_in_use`.
- **CPU**: native worker thread polling `jax.live_arrays()` — fast enough to
  catch single-element arrays.

`peak` is reported as bytes above the per-device baseline at `__enter__`,
summed across all monitored devices. Either tracker can be disabled with
`time=False` or `peak=False`.

The legacy `PeakMemoryMonitor`, which polls
`jax.profiler.device_memory_profile()` (pprof) from a Python thread, is kept
around for compatibility and emits a `DeprecationWarning`.

## Installation

```bash
pip install .
```

This builds the `xla_mem_bridge` C++ extension via `setup.py`. Requires
`nanobind` and a C++17 compiler.

## Usage

```python
import jax.numpy as jnp
from jax_memory_monitor import ResourceMonitor

with ResourceMonitor() as monitor:
    x = jnp.ones((10000, 10000))
    y = jnp.dot(x, x).block_until_ready()

print(f"Peak: {monitor.peak_mb:.2f} MB, time: {monitor.duration_ms:.2f} ms")
print(monitor.stats)  # {"time": ..., "memory": ...}
```

Target a specific device or set of devices (e.g. inside an `io_callback`
running on a different GPU than the caller):

```python
with ResourceMonitor(device=jax.devices()[1]) as monitor:
    ...

with ResourceMonitor(devices=jax.local_devices(), peak=True, time=False) as monitor:
    ...
```

`monitor.samples` exposes the per-poll history; `monitor.baseline` is the
total bytes already alive at `__enter__`.

## License

MIT
