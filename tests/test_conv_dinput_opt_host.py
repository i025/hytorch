"""Routing and differentiable-path regression; CPU stubs are not GPU evidence."""
import numpy as np
import pytest

from hytorch import Variable,Config
import hytorch.functions as F
import hytorch.functions_conv as FC
from hytorch.backends.cuda import backend,current_config
from hytorch.backends.cuda import runtime
from scripts.build_cuda import build_command


def test_context_and_build_sources():
    with backend(conv_backward='fused_opt'):
        assert current_config()['conv_backward']=='fused_opt'
        with backend(conv_backward='auto'):
            assert current_config()['conv_backward']=='auto'
        assert current_config()['conv_backward']=='fused_opt'
    assert current_config()['conv_backward']=='original'
    command=build_command('nvcc','test.so','80')
    assert any(s.endswith('/conv_dinput_opt.cu') for s in command)
    assert any(s.endswith('/conv_dinput.cu') for s in command)


@pytest.mark.parametrize('pruned',[False,True])
def test_routing_and_parameter_gradients(monkeypatch,pruned):
    calls=[]
    def stub(gy,w,shape,stride,pad):
        assert not Config.enable_backprop
        calls.append(shape)
        return FC.deconv2d(gy,w,stride=stride,pad=pad,outsize=shape[2:]).data
    monkeypatch.setattr(runtime,'conv2d_dinput_opt',stub)
    rng=np.random.default_rng(42)
    a=rng.normal(size=(2,3,7,9)).astype(np.float32)*.1
    w=rng.normal(size=(5,3,3,3)).astype(np.float32)*.1
    values=[]
    for mode in ('original','fused_opt'):
        x,weight,bias=Variable(a,requires_grad=not pruned),Variable(w),Variable(np.zeros(5,np.float32))
        with backend(conv_backward=mode):F.sum(FC.conv2d(x,weight,bias,stride=2,pad=1)**2).backward()
        assert (x.grad is None)==pruned
        values.append((weight.grad.data,bias.grad.data))
    assert len(calls)==int(not pruned)
    for a,b in zip(*values):np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize('pruned',[False,True])
def test_higher_order(monkeypatch,pruned):
    import torch
    torch.set_num_threads(4)
    def forbidden(*args,**kwargs):raise AssertionError('Higher-order graph must use the differentiable path')
    monkeypatch.setattr(runtime,'conv2d_dinput_opt',forbidden)
    rng=np.random.default_rng(42)
    a=rng.normal(size=(1,3,5,7)).astype(np.float32)*.1
    w=rng.normal(size=(5,3,3,3)).astype(np.float32)*.1
    x,weight=Variable(a,requires_grad=not pruned),Variable(w)
    tx,tw=torch.tensor(a,requires_grad=not pruned),torch.tensor(w,requires_grad=True)
    with backend(conv_backward='fused_opt'):
        y=FC.conv2d(x,weight,pad=1);F.sum(y*y).backward(create_graph=True)
        gw=weight.grad;weight.cleargrad();x.cleargrad();F.sum(gw*gw).backward(create_graph=True)
    ty=torch.nn.functional.conv2d(tx,tw,padding=1)
    tgw,=torch.autograd.grad((ty*ty).sum(),tw,create_graph=True);(tgw*tgw).sum().backward()
    np.testing.assert_allclose(weight.grad.data,tw.grad.numpy(),atol=1e-5,rtol=1e-4)
    if not pruned:np.testing.assert_allclose(x.grad.data,tx.grad.numpy(),atol=1e-5,rtol=1e-4)
