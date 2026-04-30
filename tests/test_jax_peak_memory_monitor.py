import time
import warnings

import jax.numpy as jnp
import pytest
from jax._src.lib import _profiler
from jax.profiler import device_memory_profile

from jax_memory_monitor import PeakMemoryMonitor
from jax_memory_monitor.jax_peak_memory_monitor import (
    decode_pprof,
    device_memory,
    mem_diff,
)


@pytest.fixture
def make_monitor():
    def _make(**kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return PeakMemoryMonitor(**kwargs)
    return _make


def test_emits_deprecation_warning():
    with pytest.warns(DeprecationWarning, match="ResourceMonitor"):
        PeakMemoryMonitor()


def test_basic(make_monitor):
    with make_monitor(interval=0.01) as monitor:
        x = jnp.ones((10000, 10000)).block_until_ready()
        time.sleep(0.1)
        peak_during = monitor.peak
    peak_after = monitor.peak
    assert peak_during > 0
    assert peak_after >= peak_during


def test_no_work(make_monitor):
    with make_monitor(interval=0.01) as monitor:
        pass
    assert monitor.peak >= 0


def test_nested(make_monitor):
    with make_monitor() as outer:
        x = jnp.ones((5000, 5000)).block_until_ready()
        peak1 = outer.peak
        with make_monitor() as inner:
            y = jnp.ones((7000, 7000)).block_until_ready()
            assert inner.peak > 0
        assert outer.peak >= peak1
        assert outer.peak >= inner.peak


def test_does_not_suppress_exceptions(make_monitor):
    with pytest.raises(RuntimeError, match="boom"):
        with make_monitor():
            raise RuntimeError("boom")


def test_device_memory_call():
    mem = device_memory()
    assert isinstance(mem, int)
    assert mem >= 0


def test_mem_diff_basic():
    prof1 = decode_pprof(device_memory_profile())
    x = jnp.zeros((1000, 1000)).block_until_ready()
    prof2 = decode_pprof(device_memory_profile())
    diff = mem_diff(prof2, prof1)
    assert diff >= 0


def test_mem_diff_single_arg_returns_total():
    prof = decode_pprof(device_memory_profile())
    total = mem_diff(prof)
    assert total >= 0
    assert total == sum(s.value[1] for s in prof.sample)


def test_pprof_decoding_failure_on_xspace():
    options = _profiler.ProfileOptions()
    options.python_tracer_level = 1
    session = _profiler.ProfilerSession(options)
    x = jnp.dot(jnp.ones((100, 100)), jnp.ones((100, 100))).block_until_ready()
    xspace_bytes = session.stop()
    assert len(xspace_bytes) > 0
    with pytest.raises((ValueError, Exception)):
        decode_pprof(xspace_bytes)
