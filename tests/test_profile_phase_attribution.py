"""Asynchronous GPU execution must be attributed using its CPU launch API."""
import importlib.util
from pathlib import Path

P=Path(__file__).resolve().parents[1]/'docs/experiments/gpu0-autonomous-7h/scripts/analyze_phases.py'
spec=importlib.util.spec_from_file_location('phase_attribution',P)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_asynchronous_kernel_not_assigned_by_gpu_time_overlap():
    ranges=[dict(start=10,end=20,globalTid=1,name='forward'),dict(start=21,end=40,globalTid=1,name='backward')]
    apis=[dict(start=12,end=16,globalTid=1,correlationId=9),
          dict(start=13,end=15,globalTid=1,correlationId=9),
          dict(start=25,end=27,globalTid=1,correlationId=10)]
    kernels=[dict(start=30,end=35,correlationId=9),dict(start=42,end=45,correlationId=10)]
    assert module.attribute(ranges,apis,kernels)==0
    assert [k['phase'] for k in kernels]==['forward','backward']


def test_unknown_thread_or_api_is_reported():
    ranges=[dict(start=10,end=20,globalTid=1,name='forward')]
    apis=[dict(start=12,end=15,globalTid=2,correlationId=9)]
    kernels=[dict(start=30,end=35,correlationId=9),dict(start=40,end=45,correlationId=11)]
    assert module.attribute(ranges,apis,kernels)==1
    assert [k['phase'] for k in kernels]==['OUTSIDE_MEASURED_PHASE','UNMATCHED_API']
