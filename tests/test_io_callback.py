"""Verify `ResourceMonitor` inside `io_callback`.

Inside an io_callback the host code can compile/run JAX work on any device
(typically a different one from the device that issued the callback). These
tests check that:

  - a single monitor wrapping the entire callback captures the peak
  - multiple sequential monitors inside the callback each capture their scope
  - both work when targeting a device different from the caller's device
"""
import jax
import jax.numpy as jnp
import pytest

try:
    from jax import io_callback
except ImportError:
    from jax.experimental import io_callback

from jax_memory_monitor import ResourceMonitor


_GPUS = [d for d in jax.devices() if d.platform == "gpu"]
_HAS_TWO_GPUS = len(_GPUS) >= 2


def test_io_callback_single_monitor():
    target = jax.devices()[0]
    SIZE = (2000, 2000)

    def cb():
        with ResourceMonitor(device=target) as m:
            x = jax.device_put(jnp.ones(SIZE, dtype=jnp.float32), target).block_until_ready()
            y = jax.device_put(jnp.zeros(SIZE, dtype=jnp.float32), target).block_until_ready()
        return jnp.asarray(m.peak, dtype=jnp.int32)

    @jax.jit
    def f():
        return io_callback(cb, jax.ShapeDtypeStruct((), jnp.int32))

    peak = int(f())
    assert peak >= 2 * SIZE[0] * SIZE[1] * 4


def test_io_callback_multiple_monitors():
    target = jax.devices()[0]
    A = (2000, 2000)
    B = (3000, 3000)

    def cb():
        with ResourceMonitor(device=target) as m1:
            a = jax.device_put(jnp.ones(A, dtype=jnp.float32), target).block_until_ready()
        peak1 = m1.peak
        del a
        with ResourceMonitor(device=target) as m2:
            b = jax.device_put(jnp.ones(B, dtype=jnp.float32), target).block_until_ready()
        peak2 = m2.peak
        return jnp.asarray([peak1, peak2], dtype=jnp.int32)

    @jax.jit
    def f():
        return io_callback(cb, jax.ShapeDtypeStruct((2,), jnp.int32))

    peaks = f()
    assert int(peaks[0]) >= A[0] * A[1] * 4
    assert int(peaks[1]) >= B[0] * B[1] * 4


@pytest.mark.skipif(not _HAS_TWO_GPUS, reason="needs >=2 GPUs")
def test_io_callback_targets_different_gpu():
    """jit input lives on GPU 0; callback measures peak on GPU 1."""
    src, dst = _GPUS[0], _GPUS[1]
    SIZE = (3000, 3000)

    def cb(_):
        with ResourceMonitor(device=dst) as m:
            x = jax.device_put(jnp.ones(SIZE, dtype=jnp.float32), dst).block_until_ready()
            y = jax.device_put(jnp.zeros(SIZE, dtype=jnp.float32), dst).block_until_ready()
        return jnp.asarray(m.peak, dtype=jnp.int32)

    @jax.jit
    def f(seed):
        return io_callback(cb, jax.ShapeDtypeStruct((), jnp.int32), seed)

    peak = int(f(jax.device_put(jnp.zeros(()), src)))
    assert peak >= 2 * SIZE[0] * SIZE[1] * 4


@pytest.mark.skipif(not _HAS_TWO_GPUS, reason="needs >=2 GPUs")
def test_io_callback_multiple_monitors_different_gpu():
    src, dst = _GPUS[0], _GPUS[1]
    A = (2000, 2000)
    B = (3000, 3000)

    def cb(_):
        with ResourceMonitor(device=dst) as m1:
            a = jax.device_put(jnp.ones(A, dtype=jnp.float32), dst).block_until_ready()
        peak1 = m1.peak
        del a
        with ResourceMonitor(device=dst) as m2:
            b = jax.device_put(jnp.ones(B, dtype=jnp.float32), dst).block_until_ready()
        peak2 = m2.peak
        return jnp.asarray([peak1, peak2], dtype=jnp.int32)

    @jax.jit
    def f(seed):
        return io_callback(cb, jax.ShapeDtypeStruct((2,), jnp.int32), seed)

    peaks = f(jax.device_put(jnp.zeros(()), src))
    assert int(peaks[0]) >= A[0] * A[1] * 4
    assert int(peaks[1]) >= B[0] * B[1] * 4
