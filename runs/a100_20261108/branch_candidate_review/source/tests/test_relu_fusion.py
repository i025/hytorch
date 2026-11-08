"""Imported idea from d58c581, integrated with current optimized runtime."""
import ctypes
import os
import numpy as np
import pytest
import torch  # Load this runtime before any CuPy device initialization.
import hytorch as hy
import hytorch.functions as F
from hytorch.backends.cuda import runtime as rt
from test_col2im_cuda import cp

@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_cpu_and_higher_order(dtype):
    x=hy.Variable(np.array([-2,0,3],dtype))
    with rt.backend(relu_backward='fused'):
        F.sum(F.relu(x)**2).backward(create_graph=True)
        gx=x.grad;x.cleargrad();F.sum(gx).backward()
    np.testing.assert_array_equal(x.grad.data,[0,0,2])
    ref=hy.Variable(np.array([-2,0,3],dtype))
    with rt.backend(relu_backward='original'):
        F.sum(F.relu(ref)**2).backward(create_graph=True)
        rg=ref.grad;ref.cleargrad();F.sum(rg).backward()
    assert x.grad.dtype==ref.grad.dtype
    np.testing.assert_array_equal(x.grad.data,ref.grad.data)

def test_invalid_configuration_and_flag():
    with pytest.raises(ValueError):
        with rt.backend(relu_backward='bad'):pass
    with pytest.raises(TypeError):hy.Variable(np.ones(1),requires_grad='false')

@pytest.mark.parametrize('shape',[(),(1,),(257,),(3,7,11),(4,24,31,98)])
@pytest.mark.parametrize('streamed',[False,True])
def test_gpu_alignment(cp,shape,streamed):
    rng=np.random.default_rng(42);a=rng.normal(size=shape).astype('f');g=rng.normal(size=shape).astype('f')
    stream=cp.cuda.Stream(non_blocking=True) if streamed else cp.cuda.get_current_stream()
    values=[]
    with stream:
        for mode in ['original','fused']:
            x=hy.Variable(cp.asarray(a));rt.dispatch_stats(reset=True)
            with rt.backend(relu_backward=mode):F.sum(F.relu(x)*cp.asarray(g)).backward()
            values.append(x.grad.data)
        assert rt.dispatch_stats()['relu_backward_fused']==1
    stream.synchronize()
    for v in values:np.testing.assert_array_equal(cp.asnumpy(v),g*(a>0))
    tx=torch.tensor(a,device='cuda',requires_grad=True)
    (torch.relu(tx)*torch.tensor(g,device='cuda')).sum().backward()
    np.testing.assert_array_equal(cp.asnumpy(values[1]),tx.grad.cpu().numpy())

@pytest.mark.parametrize('layout',['transpose','negative','broadcast','empty','fp64'])
def test_layout_fallback(cp,layout):
    x=cp.arange(77,dtype=cp.float32).reshape(7,11)-20
    if layout=='transpose':x=x.T
    if layout=='negative':x=x[::-1,::-2]
    if layout=='broadcast':x=cp.broadcast_to(x[0],(7,11))
    if layout=='empty':x=x[:0]
    if layout=='fp64':x=x.astype(cp.float64)
    g=cp.ones_like(x);rt.dispatch_stats(reset=True)
    with rt.backend(relu_backward='fused'):
        for _ in range(3):
            v=hy.Variable(x);F.sum(F.relu(v)*g).backward()
            np.testing.assert_array_equal(cp.asnumpy(v.grad.data),cp.asnumpy(g*(x>0)))
    assert bool(rt.dispatch_stats().get('relu_backward_fused'))==(layout not in ('empty','fp64'))

def test_special_values_and_validation(cp):
    x=cp.array([-cp.inf,-1.,-0.,0.,1.,cp.inf,cp.nan,-1.,1.],cp.float32)
    g=cp.array([cp.inf,-2.,-3.,4.,-0.,cp.nan,5.,cp.nan,cp.inf],cp.float32)
    a,b=cp.asnumpy(rt.relu_backward(x,g)),cp.asnumpy(g*(x>0))
    np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(np.signbit(a[~np.isnan(a)]),np.signbit(b[~np.isnan(b)]))
    with pytest.raises(TypeError):rt.relu_backward(x.astype(cp.float64),g)
    with pytest.raises(ValueError):rt.relu_backward(x,g[:2])
    # C ABI validates invalid calls before launch.
    assert rt.load_library().hy_relu_backward(None,None,None,0,0)!=0

def test_gpu_higher_and_mixed_dtype(cp):
    with rt.backend(relu_backward='fused'):
        x=hy.Variable(cp.array([-2.,0.,3.],cp.float32))
        F.sum(F.relu(x)**2).backward(create_graph=True)
        g=x.grad;x.cleargrad();F.sum(g).backward(create_graph=True)
        np.testing.assert_array_equal(cp.asnumpy(x.grad.data),[0,0,2])
        x=hy.Variable(cp.array([-2.,3.],cp.float32));y=F.relu(x)
        y.grad=hy.Variable(cp.ones(2,cp.float64));y.backward()
        assert x.grad.dtype==cp.float64
        np.testing.assert_array_equal(cp.asnumpy(x.grad.data),[0,1])

@pytest.mark.parametrize('width',[1,32,256])
@pytest.mark.parametrize('streamed',[False,True])
def test_heap_gpu_graph_order(cp,width,streamed):
    a=np.random.default_rng(42).normal(0,.1,(4,32)).astype(np.float32)
    stream=cp.cuda.Stream(non_blocking=True) if streamed else cp.cuda.get_current_stream()
    values=[]
    with stream:
        for mode in ['list','heap']:
            with hy.using_config('backward_schedule',mode):
                x=hy.Variable(cp.asarray(a));root=x*x
                branches=[F.tanh(root+np.float32(i/width)) for i in range(width)]
                y=branches[0]
                for value in branches[1:]:y=y+value
                F.sum(y).backward(create_graph=True)
                first=x.grad;x.cleargrad();F.sum(first).backward()
                values.append((first.data,x.grad.data))
    stream.synchronize()
    for old,new in zip(values[0],values[1]):np.testing.assert_array_equal(cp.asnumpy(old),cp.asnumpy(new))
