"""Real CUDA events and current-stream telemetry; do not silently accept GPU skips."""
import json
import os
import time
import numpy as np
import pytest
# This shared environment has CUDA 12.4 CuPy and a CUDA 12.8 PyTorch wheel.
# Load PyTorch's compatible CUDA runtime before the first CuPy CUDA operation.
import torch
import hytorch
import hytorch.functions as F
from hytorch.profiling import TrainingMonitor
from hytorch.backends.cuda import runtime as rt
from test_col2im_cuda import cp


@pytest.mark.parametrize('nondefault',[False,True])
def test_stream_events_mapping_and_repeat(cp,tmp_path,nondefault):
    stream=cp.cuda.Stream(non_blocking=True) if nondefault else cp.cuda.get_current_stream()
    m=TrainingMonitor(enabled=True,gpu_index=cp.cuda.runtime.getDevice(),sample_interval=.01,output_dir=tmp_path)
    for run in range(2):
        with stream, m:
            with m.step(0,epoch=1,batch_size=4,data_bytes=4096):
                with m.phase('forward'):
                    x=cp.ones((32,32),cp.float32);y=x+x
            deadline=time.monotonic()+2
            while len(m.gpu_samples)<2 and time.monotonic()<deadline:time.sleep(.01)
            assert m.steps[0]['step_wall_s'] is None
        assert m.steps[0]['forward_gpu_event_s']>0
        assert not m.thread.is_alive() and not m._pending
        np.testing.assert_array_equal(cp.asnumpy(y),np.full((32,32),2,np.float32))
        assert m.gpu_samples
        row=m.gpu_samples[-1]
        if row.get('gpu_uuid') is None:
            assert m.errors or 'UNAVAILABLE' in row['status_json']
        else:
            assert row['pci_bus_id'].lower().split(':')[-2:]==cp.cuda.Device().pci_bus_id.lower().split(':')[-2:]
            assert row['logical_cuda_index']==cp.cuda.runtime.getDevice()
            assert row['gpu_used_bytes']>0 and row['gpu_total_bytes']>=row['gpu_used_bytes']
            if os.environ.get('CUDA_VISIBLE_DEVICES')=='1':assert row['physical_gpu_index']==1
        assert m.cpu_samples[0]['system_cpu_pct'] is None
    assert len(list(tmp_path.glob('run_*')))==2


def test_allocator_peak_and_exception(cp,tmp_path):
    m=TrainingMonitor(enabled=True,resources=False,output_dir=tmp_path,allocator_peaks=True)
    with pytest.raises(RuntimeError,match='intentional'):
        with m:
            with m.step(7):
                with m.phase('backward'):
                    x=cp.empty((1024,1024),cp.float32)
                    raise RuntimeError('intentional')
    peak=m.summary()['allocator_tracked_peak']
    assert peak['active_bytes']>=x.nbytes and peak['phase']=='backward' and peak['step_id']==7
    assert m._hook is None and not m.running
    assert json.loads((m.run_dir/'STEP_TIMINGS.json').read_text())[0]['status']=='ERROR'


@pytest.mark.parametrize('mode',['throughput','latency'])
def test_enabled_disabled_numeric_alignment(cp,tmp_path,mode):
    rng=np.random.default_rng(91);initial=rng.normal(0,.1,(3,2,3,3)).astype('f')
    x=cp.asarray(rng.normal(0,.1,(2,2,7,9)).astype('f'))
    results=[]
    for enabled in [False,True]:
        w=hytorch.Variable(cp.asarray(initial))
        with rt.backend(conv='im2col',col2im='native',dweight='packed',metadata_cache=True),hytorch.core.using_config('compact_function',True):
            m=TrainingMonitor(enabled=enabled,resources=False,mode=mode,output_dir=tmp_path)
            with m:
                for step in range(3):
                    with m.step(step):
                        with m.phase('forward'):y=F.conv2d(x,w,pad=1)
                        with m.phase('loss'):loss=(y*y).sum()
                        with m.phase('backward'):w.cleargrad();loss.backward()
                        with m.phase('optimizer'):w.data-=np.float32(.001)*w.grad.data
            cp.cuda.get_current_stream().synchronize()
            results.append([cp.asnumpy(v) for v in (w.data,w.grad.data,loss.data)])
            if enabled:
                assert all(r['step_wall_s'] is not None for r in m.steps) if mode=='latency' else all(r['step_wall_s'] is None for r in m.steps)
    for a,b in zip(*results):assert a.tobytes()==b.tobytes()


def test_torch_events_and_allocator(cp,tmp_path):
    import torch
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream), TrainingMonitor(enabled=True,cuda_backend='torch',resources=False,
            output_dir=tmp_path,allocator_peaks=True) as m:
        with m.step(0):
            with m.phase('forward'):x=torch.ones((32,32),device='cuda');y=x+x
    assert m.steps[0]['forward_gpu_event_s']>0
    assert m.summary()['allocator_tracked_peak']['active_bytes']>=x.nelement()*x.element_size()
    assert y.cpu().numpy().sum()==2048
