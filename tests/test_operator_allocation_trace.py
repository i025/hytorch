"""Exercise allocation callbacks on CPU; no CUDA context or device query."""
import importlib.util
from pathlib import Path
import sys
from cupy.cuda.memory_hook import get_memory_hooks

directory=Path(__file__).resolve().parents[1]/'docs/experiments/gpu0-autonomous-7h/scripts'
sys.path.insert(0,str(directory))
spec=importlib.util.spec_from_file_location('operator_memory',directory/'operator_memory.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_temporaries_freed_before_return_still_contribute_to_peak():
    trace=module.AllocationTrace()
    trace.malloc_postprocess(0,100,512,1000,1)
    trace.malloc_postprocess(0,1000,1024,2000,2)
    trace.free_postprocess(0,1024,2000,2)
    assert trace.peak==1536
    assert trace.live==512
    assert trace.requested_total==1100
    assert trace.calls==2


def test_frees_outside_scope_and_pool_pointer_reuse():
    trace=module.AllocationTrace()
    trace.free_postprocess(0,1024,1000,1)
    trace.malloc_postprocess(0,100,512,1000,1)
    trace.free_postprocess(0,512,1000,1)
    trace.malloc_postprocess(0,1000,1024,1000,1)
    assert trace.live==trace.peak==1024
    assert trace.calls==2


def test_hook_context_is_removed_after_measurement():
    before=dict(get_memory_hooks())
    trace=module.AllocationTrace()
    with trace:
        assert get_memory_hooks()[trace.name] is trace
    assert dict(get_memory_hooks())==before
