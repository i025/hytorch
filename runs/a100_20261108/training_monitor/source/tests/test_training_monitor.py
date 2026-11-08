"""CPU-safe telemetry contracts, fake runtime mapping, and failure cleanup."""
import csv
import json
import sys
import threading
import time
from types import SimpleNamespace
import pytest
import hytorch.training_monitor as tm
from hytorch.profiling import TrainingMonitor


def monitor(tmp_path, **kw):
    return TrainingMonitor(enabled=True, cuda_backend=None, gpu_index=None,
                           resources=False, output_dir=tmp_path, **kw)


def test_disabled_is_noop(tmp_path, monkeypatch):
    monkeypatch.setattr(tm.importlib, 'import_module', lambda name: pytest.fail('disabled import'))
    m = TrainingMonitor(output_dir=tmp_path/'absent')
    with m:
        with m.step(1):
            with m.phase('forward'):
                m.set_step_metadata(batch_size=4)
    m.stop(); m.export(); m.flush_events()
    assert not (tmp_path/'absent').exists()
    assert m.thread is None and m.steps == []


def test_lifecycle_export_and_reuse(tmp_path):
    m = monitor(tmp_path)
    for run in range(2):
        with m:
            with m.step(1, epoch=2, batch_size=3):
                with m.phase('data_load', gpu=False):
                    sum(range(20))
                with m.phase('custom'):
                    pass
                with m.phase('custom'):
                    pass
                m.set_step_metadata(batch_size=2, data_bytes=64)
            assert m.steps[0]['step_wall_s'] > 0
        m.stop()
        rows = list(csv.DictReader((m.run_dir/'STEP_TIMINGS.csv').open()))
        assert len(rows) == 1 and rows[0]['data_bytes'] == '64'
        assert json.loads((m.run_dir/'STEP_TIMINGS.json').read_text())[0]['custom_cpu_s'] > 0
        assert m.summary()['sample_count'] == 0
    assert len(list(tmp_path.glob('run_*'))) == 2


def test_empty_run(tmp_path):
    with monitor(tmp_path) as m:
        pass
    assert m.steps == [] and (m.run_dir/'STEP_TIMINGS.csv').read_text().startswith('step_id,')
    assert m.summary()['sampled_peaks_bytes']['gpu_used_bytes'] is None


def test_misuse(tmp_path):
    m = monitor(tmp_path)
    with pytest.raises(RuntimeError):
        with m.step(): pass
    with m:
        with pytest.raises(RuntimeError): m.start()
        with pytest.raises(RuntimeError):
            with m.phase('forward'): pass
        with pytest.raises(ValueError): m.phase('illegal space')
        with pytest.raises(RuntimeError): m.export()
        with m.step(3):
            with pytest.raises(RuntimeError): m.stop()
            with pytest.raises(RuntimeError): m.flush_events()
            with pytest.raises(RuntimeError):
                with m.step(4): pass
            with m.phase('forward'):
                with pytest.raises(RuntimeError):
                    with m.phase('nested'): pass
        with pytest.raises(ValueError):
            with m.step(3): pass
        with pytest.raises(RuntimeError): m.set_step_metadata(batch_size=4)


@pytest.mark.parametrize('kwargs', [dict(sample_interval=0), dict(sample_interval=float('nan')),
                                    dict(mode='bad'), dict(cuda_backend='bad'), dict(gpu_index=-1),
                                    dict(gpu_uuid='A', gpu_pci_bus_id='B'), dict(gpu_uuid='A')])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError): TrainingMonitor(**kwargs)


def test_exception_preserved_and_saved(tmp_path):
    m = monitor(tmp_path)
    with pytest.raises(ValueError, match='model failed'):
        with m:
            with m.step(0):
                with m.phase('backward'):
                    raise ValueError('model failed')
    assert not m.running and m._active is None and m._phase_name is None
    assert m.steps[0]['status'] == 'ERROR'
    assert 'model failed' in m.steps[0]['error']
    with m: pass


def test_missing_optional_dependencies(tmp_path, monkeypatch):
    original = tm.importlib.import_module
    def missing(name):
        if name in ('psutil', 'pynvml'): raise ImportError(name+' unavailable')
        return original(name)
    monkeypatch.setattr(tm.importlib, 'import_module', missing)
    m = TrainingMonitor(enabled=True, output_dir=tmp_path, cuda_backend=None,
                        gpu_uuid='GPU-test', sample_interval=.005)
    with m:
        with m.step(1):
            with m.phase('forward'): time.sleep(.02)
    assert not m.thread.is_alive()
    assert m.gpu_samples and m.cpu_samples
    assert all(r['gpu_utilization_pct'] is None for r in m.gpu_samples)
    assert all(r['system_cpu_pct'] is None for r in m.cpu_samples)
    assert m.steps[0]['forward_cpu_s'] > 0
    assert len(m.errors) == 2


def test_nvml_mapping_missing_metric_and_cpu_priming(monkeypatch):
    calls=[]
    class NVML:
        NVML_TEMPERATURE_GPU=0;NVML_CLOCK_SM=1;NVML_CLOCK_MEM=2
        def nvmlInit(self): calls.append('init')
        def nvmlShutdown(self): calls.append('close')
        def nvmlDeviceGetHandleByPciBusId(self, bus):
            assert bus=='0000:E1:00.0';return 'physical1'
        def nvmlDeviceGetUUID(self,h):return b'GPU-one'
        def nvmlDeviceGetIndex(self,h):return 1
        def nvmlDeviceGetName(self,h):return b'A100'
        def nvmlDeviceGetPciInfo(self,h):return SimpleNamespace(busId=b'0000:E1:00.0')
        def nvmlDeviceGetUtilizationRates(self,h):return SimpleNamespace(gpu=45,memory=12)
        def nvmlDeviceGetMemoryInfo(self,h):return SimpleNamespace(used=123,total=456)
        def nvmlDeviceGetTemperature(self,h,v):return 42
        def nvmlDeviceGetPowerUsage(self,h):raise RuntimeError('NOT_SUPPORTED')
        def nvmlDeviceGetClockInfo(self,h,v):return 1000
        def nvmlDeviceGetComputeRunningProcesses(self,h):return [SimpleNamespace(pid=tm.os.getpid(),usedGpuMemory=2**64-1)]
    proc=SimpleNamespace(cpu_percent=lambda _:150,memory_info=lambda:SimpleNamespace(rss=512),num_threads=lambda:4)
    ps=SimpleNamespace(Process=lambda pid:proc,cpu_percent=lambda _:25)
    monkeypatch.setattr(tm.importlib,'import_module',lambda name:NVML() if name=='pynvml' else ps)
    r=tm._Resources(('pci','0000:E1:00.0'))
    g,c,status=r.sample();assert r.identity['physical_gpu_index']==1
    assert g['gpu_power_w'] is None and 'NOT_SUPPORTED' in status['power']
    assert g['process_gpu_used_bytes'] is None and c['system_cpu_pct'] is None
    g,c,status=r.sample();assert c['process_cpu_pct']==150 and c['system_cpu_pct']==25
    r.close();assert calls==['init','close']


def test_runtime_pci_mapping_not_visible_index(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','1')
    fake=SimpleNamespace(cuda=SimpleNamespace(Device=lambda i:SimpleNamespace(pci_bus_id='0000:E1:00.0')),get_default_memory_pool=lambda:None)
    monkeypatch.setattr(tm.importlib,'import_module',lambda _:fake)
    timer=tm._CudaTimer('cupy',0)
    assert timer.selector==('pci','0000:E1:00.0')


class FakeTimer:
    def __init__(self,*_):
        self.selector=('pci','actual');self.backend='fake';self.synced=0;self.which=1
    def stream(self):return SimpleNamespace(ptr=self.which,synchronize=self.sync)
    def sync(self):self.synced+=1
    def stream_id(self,s):return s.ptr
    def event(self):return SimpleNamespace(record=lambda stream:None)
    def elapsed(self,a,b):return .001


def test_events_deferred_and_cross_stream_invalid(tmp_path,monkeypatch):
    monkeypatch.setattr(tm,'_CudaTimer',FakeTimer)
    m=TrainingMonitor(enabled=True,resources=False,output_dir=tmp_path)
    with m:
        with m.step(1):
            with m.phase('forward'):pass
            with m.phase('overlapping'):
                m._timer.which=2
        assert m._timer.synced==0
        assert m.steps[0]['forward_gpu_event_s'] is None
        assert m.steps[0]['step_wall_s'] is None
    assert m._timer.synced==2
    assert m.steps[0]['forward_gpu_event_s']==.001
    assert m.steps[0]['overlapping_gpu_event_s'] is None
    assert m.steps[0]['gpu_event_status']=='UNSUPPORTED_CROSS_STREAM_INTERVAL'
    assert not m._pending


def test_latency_sync_only_step_end(tmp_path,monkeypatch):
    monkeypatch.setattr(tm,'_CudaTimer',FakeTimer)
    with TrainingMonitor(enabled=True,resources=False,mode='latency',output_dir=tmp_path) as m:
        with m.step(1):
            with m.phase('forward'):pass
            assert m._timer.synced==0
        assert m._timer.synced==1 and m.steps[0]['step_wall_s']>0


def test_event_failure_keeps_host_metrics(tmp_path,monkeypatch):
    class Broken(FakeTimer):
        def event(self):raise RuntimeError('event failed')
    monkeypatch.setattr(tm,'_CudaTimer',Broken)
    with TrainingMonitor(enabled=True,resources=False,output_dir=tmp_path) as m:
        with m.step(0):
            with m.phase('forward'):pass
    assert m.steps[0]['host_iteration_s']>0 and m.steps[0]['gpu_event_status']=='UNAVAILABLE'


def test_wrong_thread_rejected(tmp_path):
    with monitor(tmp_path) as m:
        errors=[]
        def work():
            try:
                with m.step(0):pass
            except RuntimeError:errors.append(True)
        t=threading.Thread(target=work);t.start();t.join();assert errors==[True]


def test_resources_only_no_steps(tmp_path,monkeypatch):
    class Provider:
        identity={};errors=[]
        def __init__(self, selector):pass
        def sample(self):return {'gpu_used_bytes':None},{'system_cpu_pct':None},{}
        def close(self):pass
    monkeypatch.setattr(tm,'_Resources',Provider)
    with TrainingMonitor(enabled=True,timing=False,cuda_backend=None,output_dir=tmp_path,sample_interval=.005) as m:
        with m.step(0):
            with m.phase('forward'):time.sleep(.025)
    assert not m.steps and len(m.gpu_samples)>=2 and not m.thread.is_alive()
    times=[r['monotonic_ns'] for r in m.gpu_samples]
    assert times==sorted(times) and all(r['schedule_delay_s']>=0 for r in m.gpu_samples)


def test_loop_covers_next_and_matches_unmonitored_cpu(tmp_path):
    import numpy as np
    from hytorch.monitor_loop import monitored_epochs
    class Dataset:
        def __len__(self):return 4
        def __getitem__(self,index):return np.array([index],dtype='f'),np.array([index],dtype='f')
    calls=[]
    with monitor(tmp_path) as m:
        losses,epochs=monitored_epochs(Dataset(),2,1,monitor=m,transfer=lambda x:x,
            predict=lambda x:x,loss_fn=lambda x,t:np.sum(x-t),backward=lambda loss:calls.append('backward'),
            update=lambda:calls.append('update'))
    assert calls==['backward','update']*2
    assert len(losses)==2 and len(epochs)==1
    assert all(r['data_load_cpu_s']>0 and r['data_bytes']==16 for r in m.steps)


def test_nonzero_sampler_allocator_device_selected_once(tmp_path,monkeypatch):
    calls=[]
    class Timer(FakeTimer):
        def __init__(self,*_):
            super().__init__();self.backend='cupy';self.device=1
            self.module=SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(setDevice=lambda d:calls.append((d,threading.get_ident())))))
        def allocator(self):return {'cupy_active_bytes':32,'cupy_reserved_bytes':64}
    class Provider:
        identity={};errors=[]
        def __init__(self,selector):pass
        def sample(self):return {},{},{}
        def close(self):pass
    monkeypatch.setattr(tm,'_CudaTimer',Timer);monkeypatch.setattr(tm,'_Resources',Provider)
    with TrainingMonitor(enabled=True,timing=False,gpu_index=1,output_dir=tmp_path,sample_interval=.005) as m:
        time.sleep(.02)
    assert len(calls)==1 and calls[0][0]==1 and calls[0][1]!=threading.get_ident()
    assert m.gpu_samples[0]['cupy_active_bytes']==32
