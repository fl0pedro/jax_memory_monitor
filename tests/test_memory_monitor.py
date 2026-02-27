import time
import jax.numpy as jnp
import pytest
from jax.profiler import device_memory_profile
from jax._src.lib import _profiler
from jax_memory_monitor import PeakMemoryMonitor
from jax_memory_monitor.jax_peak_memory_monitor import decode_pprof, mem_diff, device_memory

def test_peak_memory_monitor_basic():
    with PeakMemoryMonitor(interval=0.01) as monitor:
        x = jnp.ones((10000, 10000)).block_until_ready()
        time.sleep(0.1)
        peak_during = monitor.peak
        
    peak_after = monitor.peak
    assert peak_during > 0
    assert peak_after >= peak_during
    print(f"Captured peak: {peak_after / 1e6:.2f} MB")

def test_peak_memory_monitor_no_work():
    with PeakMemoryMonitor(interval=0.01) as monitor:
        pass
    assert monitor.peak >= 0

def test_nested_monitors():
    with PeakMemoryMonitor() as outer:
        x = jnp.ones((5000, 5000)).block_until_ready()
        peak1 = outer.peak
        
        with PeakMemoryMonitor() as inner:
            y = jnp.ones((7000, 7000)).block_until_ready()
            assert inner.peak > 0
            
        assert outer.peak >= peak1
        assert outer.peak >= inner.peak

def test_device_memory_call():
    mem = device_memory()
    assert isinstance(mem, int)
    assert mem >= 0

def test_mem_diff_basic():
    p1_bytes = device_memory_profile()
    prof1 = decode_pprof(p1_bytes)
    
    x = jnp.zeros((1000, 1000)).block_until_ready()
    
    p2_bytes = device_memory_profile()
    prof2 = decode_pprof(p2_bytes)
    
    diff = mem_diff(prof2, prof1)
    assert diff >= 0

def test_pprof_decoding_failure_on_xspace():
    options = _profiler.ProfileOptions()
    options.python_tracer_level = 1
    session = _profiler.ProfilerSession(options)
    
    x = jnp.dot(jnp.ones((100, 100)), jnp.ones((100, 100))).block_until_ready()
    
    xspace_bytes = session.stop()
    assert len(xspace_bytes) > 0
    
    with pytest.raises((ValueError, Exception)):
         decode_pprof(xspace_bytes)
