"""Host contracts only. These tests do not establish CUDA correctness or speed."""
import json
from types import SimpleNamespace
import numpy as np
import pytest
from hytorch import Variable, no_grad
import hytorch.functions as F
import hytorch.layers as L
from hytorch.optimizers import SGD, Adam, AdamW, FreezeParam
from hytorch.backends.cuda import backend, current_config, dispatch_stats
from hytorch.profiling import StepProfiler, ResourceMonitor, diagnose
from hytorch.model_zoo import ImageNetCNN, preprocess
from benchmarks.bench_cuda_training import TinyNet, initialize, torch_model, parameters
from scripts.build_cuda import build_command


def test_context_restores_and_cpu_fallback():
    original=current_config()
    dispatch_stats(reset=True)
    x=Variable(np.arange(6,dtype=np.float32).reshape(2,3))
    w=Variable(np.ones((3,2),np.float32))
    with pytest.raises(RuntimeError):
        with backend(matmul="shared"):
            y=F.matmul(x,w)
            np.testing.assert_array_equal(y.data,x.data@w.data)
            y.sum().backward()
            np.testing.assert_array_equal(x.grad.data,np.full((2,3),2.))
            raise RuntimeError("intentional")
    assert current_config()==original
    assert dispatch_stats()["matmul_fallback_cpu_or_dtype"]==1


@pytest.mark.parametrize("optimizer",[SGD,Adam,AdamW])
def test_freeze_hook_excludes_update(optimizer):
    model=TinyNet()
    initialize(model,(3,16,16),7)
    before=model.conv1.W.data.copy()
    head=model.fc1.W.data.copy()
    opt=optimizer().setup(model)
    opt.add_hook(FreezeParam(model.conv1))
    x=np.ones((2,3,16,16),np.float32)
    model(x).sum().backward(); opt.update()
    np.testing.assert_array_equal(model.conv1.W.data,before)
    assert not np.array_equal(model.fc1.W.data,head)


def test_initializers_preserve_float32():
    for model,x in [(L.Linear(3),np.ones((2,5),np.float32)),
                    (L.Conv2d(3,3),np.ones((2,2,5,5),np.float32))]:
        assert model(x).dtype==np.float32
        assert model.W.dtype==np.float32


def test_profiler_denominator_and_throughput(tmp_path):
    p=StepProfiler(tmp_path/"steps.jsonl")
    p.begin()
    for phase in ("data","transfer","forward","backward","update"):
        with p.phase(phase):
            sum(range(100))
    row=p.end(4,1,1)
    assert row["step_s"]>0
    assert sum(p.summary()["phase_fraction"].values())==pytest.approx(1.)
    assert p.summary()["samples_per_sec"]==pytest.approx(4/row["step_s"])
    p=StepProfiler(tmp_path/"throughput.jsonl",mode="throughput")
    p.begin()
    with p.phase("forward"):
        pass
    assert p.end(2,1,1)["forward_s"] is None


def test_monitor_parsing_and_missing_fields(tmp_path,monkeypatch):
    monitor=ResourceMonitor(tmp_path/"hardware.jsonl",gpu_id="GPU-test")
    def fake(*args,**kwargs):
        monitor.stop_event.set()  # One sample, no wall wait.
        return SimpleNamespace(stdout="65, 123, 8192, 70, [N/A]\n")
    monkeypatch.setattr("hytorch.profiling.subprocess.run",fake)
    monitor._sample()
    row=monitor.rows[0]
    assert row["gpu_utilization_pct"]==65
    assert row["gpu_memory_used_mib"]==123
    assert row["gpu_memory_total_mib"]==8192
    assert row["gpu_temperature_c"]==70
    assert row["gpu_power_w"] is None
    assert "sampled" in monitor.summary()["memory_peak_kind"]
    assert diagnose({"steps":0}, {})


@pytest.mark.parametrize("name,convs",[("alexnet",5),("vgg11",8),("vgg16",13)])
def test_pretrained_model_schema(name,convs):
    model=ImageNetCNN(name)
    assert len(model.weight_map)==2*(convs+3)
    assert len([o for o in model.feature_ops if o[0]=="conv"])==convs
    with pytest.raises(ValueError,match="224"):
        model(np.zeros((1,3,32,32),np.float32))
    model.freeze_features()
    assert model.frozen_features


def test_preprocess_shapes_and_channels():
    from PIL import Image
    x=preprocess(Image.new("RGB",(301,400),color=(255,0,0)))
    assert x.shape==(3,224,224) and x.dtype==np.float32 and x.flags.c_contiguous
    np.testing.assert_allclose(x[:,0,0],[(1-.485)/.229,-.456/.224,-.406/.225],rtol=1e-6)


def test_build_arch_validation():
    assert "-gencode=arch=compute_89,code=sm_89" in build_command("nvcc","test.so","89")
    with pytest.raises(ValueError):
        build_command("nvcc","test.so","89;echo")


@pytest.mark.parametrize("kind,shape",[("tiny",(3,16,16)),("donkey",(3,66,200))])
def test_torch_parity_forward_backward_update(kind,shape):
    torch=pytest.importorskip("torch")
    from donkeycar_hy.model import DonkeyNet
    model=TinyNet() if kind=="tiny" else DonkeyNet(dropout_rate=0.)
    state=initialize(model,shape,42)
    reference=torch_model(kind,state)
    rng=np.random.default_rng(12)
    x=rng.normal(size=(2,*shape)).astype(np.float32)
    t=rng.normal(size=(2,1)).astype(np.float32)
    actual=model(x)
    expected=reference(torch.from_numpy(x))
    np.testing.assert_allclose(actual.data,expected.detach().numpy(),atol=1e-6,rtol=1e-5)
    F.mean_square_error(actual,t).backward()
    torch.nn.functional.mse_loss(expected,torch.from_numpy(t)).backward()
    for key,param in parameters(model).items():
        layer,name=key.split("/")
        ref=getattr(getattr(reference,layer),"weight" if name=="W" else "bias").grad.numpy()
        if ref.ndim==2:
            ref=ref.T
        np.testing.assert_allclose(param.grad.data,ref,atol=1e-5,rtol=1e-4)
    SGD(lr=.001).setup(model).update()
    torch.optim.SGD(reference.parameters(),lr=.001).step()
    for key,param in parameters(model).items():
        layer,name=key.split("/")
        ref=getattr(getattr(reference,layer),"weight" if name=="W" else "bias").detach().numpy()
        np.testing.assert_allclose(param.data,ref.T if ref.ndim==2 else ref,atol=1e-6,rtol=1e-5)


def test_missing_library_fails_not_silent_fallback(monkeypatch,tmp_path):
    from hytorch.backends.cuda import runtime
    monkeypatch.setattr(runtime,"_library",None)
    monkeypatch.setenv("HYTORCH_CUDA_LIBRARY",str(tmp_path/"missing.so"))
    with pytest.raises(RuntimeError,match="missing"):
        runtime.load_library()
