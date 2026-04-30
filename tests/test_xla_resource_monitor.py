import time

import jax
import jax.numpy as jnp
import pytest

from jax_memory_monitor import ResourceMonitor


def test_basic():
    with ResourceMonitor() as monitor:
        x = jnp.ones((5000, 5000)).block_until_ready()
    assert monitor.peak >= 5000 * 5000 * 4
    assert monitor.duration > 0


def test_no_work():
    with ResourceMonitor() as monitor:
        pass
    assert monitor.peak >= 0
    assert monitor.duration >= 0


def test_nested():
    with ResourceMonitor() as outer:
        x = jnp.ones((3000, 3000)).block_until_ready()
        with ResourceMonitor() as inner:
            y = jnp.ones((5000, 5000)).block_until_ready()
        assert inner.peak >= 5000 * 5000 * 4
    assert outer.peak >= 5000 * 5000 * 4 + 3000 * 3000 * 4


def test_nested_outer_dominates():
    """Outer captures the cumulative peak even when inner clears."""
    with ResourceMonitor() as outer:
        big = jnp.ones((6000, 6000)).block_until_ready()
        big_bytes = big.nbytes
        with ResourceMonitor() as inner:
            small = jnp.ones((1000, 1000)).block_until_ready()
        del big
    assert outer.peak >= big_bytes


def test_does_not_suppress_exceptions():
    with pytest.raises(RuntimeError, match="boom"):
        with ResourceMonitor():
            raise RuntimeError("boom")


def test_explicit_device():
    dev = jax.devices()[0]
    with ResourceMonitor(device=dev) as monitor:
        x = jnp.ones((1000, 1000)).block_until_ready()
    assert monitor.peak >= 1000 * 1000 * 4


def test_peak_disabled():
    with ResourceMonitor(peak=False) as monitor:
        time.sleep(0.001)
    assert monitor.duration > 0
    with pytest.raises(RuntimeError, match="peak=False"):
        _ = monitor.peak


def test_time_disabled():
    with ResourceMonitor(time=False) as monitor:
        x = jnp.ones((100, 100)).block_until_ready()
    assert monitor.peak >= 0
    with pytest.raises(RuntimeError, match="time=False"):
        _ = monitor.duration


def test_stats_contains_both():
    with ResourceMonitor() as monitor:
        x = jnp.ones((1000, 1000)).block_until_ready()
    s = monitor.stats
    assert set(s.keys()) == {"time", "memory"}
    assert s["memory"] == monitor.peak
    assert s["time"] == monitor.duration


def test_convenience_units():
    with ResourceMonitor() as monitor:
        x = jnp.ones((1000, 1000)).block_until_ready()
        time.sleep(0.002)
    assert monitor.peak_mb == monitor.peak / (1024 * 1024)
    assert monitor.duration_ms == monitor.duration * 1000.0


@pytest.mark.skipif(
    not any(d.platform == "cpu" for d in jax.devices()),
    reason="needs a CPU device",
)
def test_cpu_catches_super_small_arrays():
    """Native polling sees tiny arrays the Python thread + pprof would miss."""
    cpu = next(d for d in jax.devices() if d.platform == "cpu")
    n_elements = 200
    with ResourceMonitor(device=cpu, poll_interval=0.0001) as m:
        arrs = [jnp.array(float(i), dtype=jnp.float32) for i in range(n_elements)]
        jax.block_until_ready(arrs)
        time.sleep(0.005)
    assert m.peak >= n_elements * 4


@pytest.mark.skipif(
    not any(d.platform == "cpu" for d in jax.devices()),
    reason="needs a CPU device",
)
def test_cpu_catches_single_scalar():
    cpu = next(d for d in jax.devices() if d.platform == "cpu")
    with ResourceMonitor(device=cpu, poll_interval=0.0001) as m:
        x = jnp.array(1.0, dtype=jnp.float32).block_until_ready()
        time.sleep(0.005)
    assert m.peak >= 4


def test_samples_recorded():
    with ResourceMonitor(poll_interval=0.0005) as m:
        x = jnp.ones((2000, 2000)).block_until_ready()
        time.sleep(0.01)
    assert len(m.samples) > 0
    assert all(s.bytes >= 0 for s in m.samples)
    assert all(s.timestamp_ns >= 0 for s in m.samples)
