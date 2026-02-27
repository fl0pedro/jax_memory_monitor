# jax_memory_monitor

A lightweight JAX peak memory monitor using `pprof`.

## Overview

`jax_memory_monitor` provides a simple context manager to track the peak device memory usage of JAX operations. It's particularly useful for benchmarking and optimizing memory-intensive models.

## Installation

```bash
pip install jax_memory_monitor
```

## Usage

```python
import jax
import jax.numpy as jnp
from jax_memory_monitor import PeakMemoryMonitor

# Monitor peak memory during an operation
with PeakMemoryMonitor() as monitor:
    x = jrand.normal(jrand.PRNGKey(0), (10000, 10000))
    y = jnp.dot(x, x).block_until_ready()

print(f"Peak memory usage: {monitor.peak / 1024**2:.2f} MB")
```

## Features

- Real-time monitoring of device memory.
- Uses JAX's built-in profiler via `pprof`.
- Lightweight context manager interface.

## License

MIT
