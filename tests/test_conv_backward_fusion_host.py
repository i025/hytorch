"""CPU routing and higher-order contracts; these do not validate CUDA execution."""
import numpy as np
import pytest

from hytorch import Variable, Config
import hytorch.functions as F
import hytorch.functions_conv as FC
from hytorch.backends.cuda import backend, current_config
from hytorch.backends.cuda import runtime
from scripts.build_cuda import build_command


def test_backend_context_and_build_source():
    original = current_config()
    assert original['conv_backward'] == 'original'
    with backend(conv_backward='fused'):
        assert current_config()['conv_backward'] == 'fused'
        with backend():
            assert current_config()['conv_backward'] == 'original'
        assert current_config()['conv_backward'] == 'fused'
    assert current_config() == original
    with pytest.raises(ValueError, match='conv_backward'):
        with backend(conv_backward='invalid'):
            pass
    assert any(s.endswith('/conv_dinput.cu') for s in build_command('nvcc','test.so','80'))


@pytest.mark.parametrize('pruned', [False, True])
def test_explicit_routing_keeps_parameter_gradients(monkeypatch, pruned):
    rng = np.random.default_rng(42)
    a = rng.normal(size=(2,2,7,8)).astype(np.float32)*.1
    w = rng.normal(size=(3,2,3,3)).astype(np.float32)*.1
    calls = []

    def routing_stub(gy, weight, shape, stride, pad):
        # CPU numerical substitute only to exercise dispatch, never called CUDA evidence.
        assert not Config.enable_backprop
        calls.append(shape)
        return FC.deconv2d(gy,weight,stride=stride,pad=pad,outsize=shape[2:]).data

    monkeypatch.setattr(runtime,'conv2d_dinput',routing_stub)
    gradients = []
    for mode in ('original','fused'):
        x, weight, bias = Variable(a,requires_grad=not pruned), Variable(w), Variable(np.zeros(3,np.float32))
        with backend(conv_backward=mode):
            y = FC.conv2d(x,weight,bias,stride=2,pad=1)
            F.sum(y*y).backward()
        gradients.append((weight.grad.data,bias.grad.data))
        assert (x.grad is None) == pruned
    assert calls == ([] if pruned else [a.shape])
    for a,b in zip(*gradients):
        np.testing.assert_array_equal(a,b)


def test_explicit_fused_rejects_cpu_arrays():
    with pytest.raises(TypeError, match='CuPy float32'):
        runtime.conv2d_dinput(np.ones((1,1,2,2),np.float32),np.ones((1,1,2,2),np.float32),(1,1,3,3))


def test_profile_wrapper_preserves_results_and_labels(monkeypatch):
    import importlib.util
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    path=root/'docs/experiments/conv-backward-fusion/scripts/profile_training.py'
    spec=importlib.util.spec_from_file_location('stage4d_profile',path)
    profile=importlib.util.module_from_spec(spec);spec.loader.exec_module(profile)
    rng=np.random.default_rng(42)
    a=rng.normal(size=(2,2,7,8)).astype(np.float32)*.1
    w=rng.normal(size=(3,2,3,3)).astype(np.float32)*.1
    def stub(gy,weight,shape,stride,pad):
        return FC.deconv2d(gy,weight,stride=stride,pad=pad,outsize=shape[2:]).data
    monkeypatch.setattr(runtime,'conv2d_dinput',stub)
    def run():
        x,weight=Variable(a),Variable(w)
        with backend(conv_backward='fused'):
            F.sum(FC.conv2d(x,weight)**2).backward()
        return x.grad.data,weight.grad.data
    reference=run()
    labels=[]
    with profile.FusionRanges(labels.append,lambda:None) as recorder:
        actual=run()
    for v,r in zip(actual,reference):np.testing.assert_array_equal(v,r)
    assert 'OP/internal:native.conv_dinput' in labels
    assert not recorder.stack and not recorder.patches
    assert runtime.conv2d_dinput is stub


@pytest.mark.parametrize('pruned', [False, True])
def test_higher_order_keeps_differentiable_path(monkeypatch, pruned):
    import torch
    torch.set_num_threads(4)
    rng = np.random.default_rng(42)
    a = rng.normal(size=(1,2,4,4)).astype(np.float32)*.1
    w = rng.normal(size=(2,2,2,2)).astype(np.float32)*.1
    x, weight = Variable(a,requires_grad=not pruned), Variable(w)
    tx, tw = torch.tensor(a,requires_grad=not pruned), torch.tensor(w,requires_grad=True)

    def forbidden(*args, **kwargs):
        raise AssertionError('create_graph must retain the original differentiable implementation')

    monkeypatch.setattr(runtime,'conv2d_dinput',forbidden)
    with backend(conv_backward='fused'):
        y = FC.conv2d(x,weight)
        F.sum(y*y).backward(create_graph=True)
        gw = weight.grad
        weight.cleargrad(); x.cleargrad()
        F.sum(gw*gw).backward(create_graph=True)
    ty = torch.nn.functional.conv2d(tx,tw)
    tgw, = torch.autograd.grad((ty*ty).sum(),tw,create_graph=True)
    (tgw*tgw).sum().backward()
    np.testing.assert_allclose(weight.grad.data,tw.grad.numpy(),atol=1e-5,rtol=1e-4)
    if pruned:
        assert x.grad is None
    else:
        np.testing.assert_allclose(x.grad.data,tx.grad.numpy(),atol=1e-5,rtol=1e-4)
