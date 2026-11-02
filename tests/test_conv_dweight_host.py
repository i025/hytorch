"""CPU routing and higher-order regression; no GPU fallback or GPU claim."""
import numpy as np
import pytest

from hytorch import Variable,Config
import hytorch.functions as F
import hytorch.functions_conv as FC
from hytorch.backends.cuda import backend,current_config,runtime
from scripts.build_cuda import build_command


def test_context_and_build():
    original=current_config()
    with backend(conv_dweight='fused'):
        assert current_config()['conv_dweight']=='fused'
        with backend():assert current_config()['conv_dweight']=='original'
    assert current_config()==original
    with pytest.raises(ValueError,match='conv_dweight'):
        with backend(conv_dweight='unknown'):pass
    assert any(s.endswith('/conv_dweight.cu') for s in build_command('nvcc','test.so','80'))


@pytest.mark.parametrize('pruned',[False,True])
@pytest.mark.parametrize('bias',[False,True])
def test_routing(monkeypatch,pruned,bias):
    calls=[]
    def stub(x,gy,kernel,stride,pad):
        assert not Config.enable_backprop
        calls.append(x.shape)
        col=FC.im2col_array(x,kernel,stride,pad,to_matrix=False)
        return np.tensordot(gy,col,((0,2,3),(0,4,5)))
    monkeypatch.setattr(runtime,'conv2d_dweight',stub)
    rng=np.random.default_rng(42)
    a=rng.normal(size=(2,3,7,9)).astype(np.float32)*.1
    weight=rng.normal(size=(5,3,3,3)).astype(np.float32)*.1
    results=[]
    for mode in ('original','fused'):
        x,w=Variable(a,requires_grad=not pruned),Variable(weight)
        b=Variable(np.zeros(5,np.float32)) if bias else None
        with backend(conv_dweight=mode):F.sum(FC.conv2d(x,w,b,stride=2,pad=1)**2).backward()
        assert (x.grad is None)==pruned
        results.append((w.grad.data,None if b is None else b.grad.data,None if pruned else x.grad.data))
    assert len(calls)==1
    for a,b in zip(*results):
        if a is not None:np.testing.assert_allclose(a,b,atol=1e-5,rtol=1e-4)


@pytest.mark.parametrize('pruned',[False,True])
def test_higher_order(monkeypatch,pruned):
    import torch
    torch.set_num_threads(4)
    def forbidden(*args):raise AssertionError('A higher-order graph must retain original dWeight')
    monkeypatch.setattr(runtime,'conv2d_dweight',forbidden)
    rng=np.random.default_rng(42)
    a=rng.normal(size=(1,3,5,7)).astype(np.float32)*.1
    w=rng.normal(size=(5,3,3,3)).astype(np.float32)*.1
    x,weight=Variable(a,requires_grad=not pruned),Variable(w)
    tx,tw=torch.tensor(a,requires_grad=not pruned),torch.tensor(w,requires_grad=True)
    with backend(conv_dweight='fused'):
        y=FC.conv2d(x,weight,pad=1);F.sum(y*y).backward(create_graph=True)
        gw=weight.grad;weight.cleargrad();x.cleargrad();F.sum(gw*gw).backward(create_graph=True)
    ty=torch.nn.functional.conv2d(tx,tw,padding=1)
    tgw,=torch.autograd.grad((ty*ty).sum(),tw,create_graph=True);(tgw*tgw).sum().backward()
    np.testing.assert_allclose(weight.grad.data,tw.grad.numpy(),atol=1e-5,rtol=1e-4)
    if not pruned:np.testing.assert_allclose(x.grad.data,tx.grad.numpy(),atol=1e-5,rtol=1e-4)


def test_profile_ranges(monkeypatch):
    import importlib.util
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/'docs/experiments/conv-dweight-fusion/scripts/profile_training.py'
    spec=importlib.util.spec_from_file_location('weight_profile',path)
    profile=importlib.util.module_from_spec(spec);spec.loader.exec_module(profile)
    def stub(x,gy,kernel,stride,pad):
        return np.tensordot(gy,FC.im2col_array(x,kernel,stride,pad,to_matrix=False),((0,2,3),(0,4,5)))
    monkeypatch.setattr(runtime,'conv2d_dweight',stub)
    active=[];labels=[]
    def push(name):active.append(name);labels.append(name)
    with profile.WeightRanges(push,lambda:active.pop()),backend(conv_dweight='fused'):
        x=Variable(np.ones((1,2,5,7),np.float32));w=Variable(np.ones((3,2,3,3),np.float32))
        F.sum(FC.conv2d(x,w)**2).backward()
    assert not active and 'OP/internal:native.conv_dweight' in labels
    assert runtime.conv2d_dweight is stub


def test_controller_reset_symbol():
    import importlib.util
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/'docs/experiments/conv-dweight-fusion/scripts/run_window.py'
    spec=importlib.util.spec_from_file_location('weight_window',path)
    window=importlib.util.module_from_spec(spec);spec.loader.exec_module(window)
    # Resolve the real toolkit symbol on CPU; do not call it or create a context.
    assert window.reset_function().argtypes==[]
