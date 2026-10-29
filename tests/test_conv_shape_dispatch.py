"""Independent CNN reference and dispatch metadata contracts."""
import numpy as np
import pytest

from hytorch import Variable
import hytorch.functions as F
from benchmarks.bench_cuda_training import SpatialNet, initialize, parameters, torch_model
from hytorch.optimizers import SGD
from hytorch.backends.cuda import backend,dispatch_stats,current_config
from hytorch.backends.cuda import runtime
import hytorch.functions_conv as FC


@pytest.mark.parametrize('shape,wshape,stride,pad,dtype,contiguous,expected',[
    ((2,8,9,17),(24,8,1,1),1,0,'float32',True,'fused'),
    ((8,32,9,9),(56,32,1,1),1,0,'float32',True,'fused'),
    ((8,32,9,9),(57,32,1,1),1,0,'float32',True,'original'),
    ((17,3,3,5),(16,3,1,1),1,0,'float32',True,'original'),
    ((1,3,2,9),(16,3,1,1),1,0,'float32',True,'original'),
    ((1,3,9,9),(16,3,3,3),1,1,'float32',True,'original'),
    ((1,3,9,9),(16,3,5,5),1,0,'float32',True,'original'),
    ((1,3,9,9),(16,3,1,1),2,0,'float32',True,'original'),
    ((1,3,9,9),(16,3,1,1),1,1,'float32',True,'original'),
    ((1,3,9,9),(16,3,1,1),1,0,'float64',True,'original'),
    ((1,3,9,9),(16,3,1,1),1,0,'float32',False,'original'),
])
def test_operator_metadata_policy(shape,wshape,stride,pad,dtype,contiguous,expected):
    assert runtime.select_conv_dinput(shape,wshape,stride,pad,dtype,contiguous)[0]==expected


@pytest.mark.parametrize('pruned',[False,True])
def test_cpu_auto_and_input_gradient_contract(pruned):
    rng=np.random.default_rng(42)
    a=rng.normal(size=(2,3,5,7)).astype(np.float32)*.1
    w=rng.normal(size=(16,3,1,1)).astype(np.float32)*.1
    rows=[]
    for mode in ('original','auto'):
        dispatch_stats(reset=True)
        x,ww=Variable(a,requires_grad=not pruned),Variable(w)
        with backend(conv_backward=mode):F.sum(FC.conv2d(x,ww)**2).backward()
        assert (x.grad is None)==pruned
        rows.append(ww.grad.data)
        if mode=='auto':
            assert dispatch_stats().get('conv_dinput_auto_original',0)==(0 if pruned else 1)
            assert dispatch_stats().get('conv_dinput_auto_reason_cpu',0)==(0 if pruned else 1)
    np.testing.assert_array_equal(*rows)
    assert current_config()['conv_backward']=='original'


def test_auto_higher_order_original_path(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('Higher-order graph must not call native gather')
    monkeypatch.setattr(runtime,'conv2d_dinput',forbidden)
    rng=np.random.default_rng(42);a=rng.normal(size=(1,3,5,7)).astype(np.float32)*.1
    w=rng.normal(size=(16,3,1,1)).astype(np.float32)*.1
    rows=[]
    dispatch_stats(reset=True)
    for mode in ('original','auto'):
        x,ww=Variable(a),Variable(w)
        with backend(conv_backward=mode):
            F.sum(FC.conv2d(x,ww)**2).backward(create_graph=True)
            gw=ww.grad;ww.cleargrad();F.sum(gw**2).backward()
        rows.append(ww.grad.data)
    np.testing.assert_array_equal(*rows)
    assert dispatch_stats().get('conv_dinput_auto_reason_higher_order',0)>0


@pytest.mark.parametrize('input_gradient',[False,True])
def test_spatial_cnn_training_reference(input_gradient):
    import torch
    torch.set_num_threads(4)
    model=SpatialNet();state=initialize(model,(3,16,16),42)
    reference=torch_model('spatial',state)
    optimizer=SGD(.001).setup(model)
    other=torch.optim.SGD(reference.parameters(),lr=.001)
    rng=np.random.default_rng(42)
    for _ in range(3):
        a=rng.normal(size=(4,3,16,16)).astype(np.float32)*.1
        t=rng.normal(size=(4,1)).astype(np.float32)*.1
        x=Variable(a,requires_grad=input_gradient)
        tx=torch.tensor(a,requires_grad=input_gradient)
        model.cleargrads();other.zero_grad(set_to_none=True)
        loss=F.mean_square_error(model(x),t)
        expected=torch.nn.functional.mse_loss(reference(tx),torch.tensor(t))
        loss.backward();expected.backward()
        np.testing.assert_allclose(loss.data,expected.detach().numpy(),atol=1e-6,rtol=1e-5)
        assert (x.grad is None)==(not input_gradient)
        if input_gradient:np.testing.assert_allclose(x.grad.data,tx.grad.numpy(),atol=1e-5,rtol=1e-4)
        for name,p in parameters(model).items():
            assert p.grad.dtype==np.float32
            layer,attr=name.split('/')
            value=getattr(getattr(reference,layer),'weight' if attr=='W' else 'bias').grad.numpy()
            if value.ndim==2:value=value.T
            np.testing.assert_allclose(p.grad.data,value,atol=1e-5,rtol=1e-4)
        optimizer.update();other.step()
        for name,p in parameters(model).items():
            layer,attr=name.split('/')
            value=getattr(getattr(reference,layer),'weight' if attr=='W' else 'bias').detach().numpy()
            if value.ndim==2:value=value.T
            np.testing.assert_allclose(p.data,value,atol=1e-6,rtol=1e-5)
