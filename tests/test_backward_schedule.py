"""Heap scheduling preserves generation priority and legacy last-added tie order."""
import numpy as np
import pytest

import hytorch
import hytorch.functions as F
from hytorch.core import Function,Config


class Traced(Function):
    def __init__(self,name,trace):
        self.name,self.trace=name,trace

    def forward(self,x):
        return x*np.float32(1.125)

    def backward(self,gy):
        self.trace.append(self.name)
        return gy*np.float32(1.125)


def branched(mode,width,retain=False):
    trace=[]
    x=hytorch.Variable(np.array([.1,-.2,.3],np.float32))
    with hytorch.using_config('backward_schedule',mode):
        shared=Traced('shared',trace)(x)
        branches=[Traced(str(i),trace)(shared+np.float32(i/width)) for i in range(width)]
        result=branches[0]
        for value in branches[1:]:result=result+value
        F.sum(result**2).backward(retain_grad=retain,create_graph=True)
        first=x.grad.data.copy();gradient=x.grad
        retained=[b.grad.data.copy() if b.grad is not None else None for b in branches]
        x.cleargrad();F.sum(gradient).backward()
        second=x.grad.data.copy()
    return trace,first,second,retained


@pytest.mark.parametrize('width',[1,2,7,32,256])
@pytest.mark.parametrize('retain',[False,True])
def test_order_and_higher_derivatives(width,retain):
    old,new=[branched(mode,width,retain) for mode in ('list','heap')]
    assert old[0]==new[0]
    for a,b in zip(old[1:3],new[1:3]):np.testing.assert_array_equal(a,b)
    if retain:
        for a,b in zip(old[3],new[3]):np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize('mode',['list','heap'])
def test_leaves_and_no_gradient_receipt(mode):
    with hytorch.using_config('backward_schedule',mode):
        leaf=hytorch.Variable(np.array(2,np.float32));leaf.backward()
        assert leaf.grad.data==1
        x=hytorch.Variable(np.array([1.,2.],np.float32))
        target=hytorch.Variable(np.array([.1,.2],np.float32),requires_grad=False)
        F.mean_square_error(x,target).backward()
        assert target.grad is None
        np.testing.assert_array_equal(x.grad.data,np.array([.9,1.8],np.float32))


@pytest.mark.parametrize('mode',['list','heap'])
def test_exception_restores_backprop_context(mode):
    class Broken(Function):
        def forward(self,x):return x.copy()
        def backward(self,gy):raise RuntimeError('intentional failure')
    with hytorch.using_config('backward_schedule',mode):
        x=hytorch.Variable(np.ones(3,np.float32))
        with pytest.raises(RuntimeError,match='intentional failure'):Broken()(x).backward()
    assert Config.enable_backprop
    assert Config.backward_schedule=='list'


def test_invalid_schedule_is_not_silently_ignored():
    x=hytorch.Variable(np.ones(3,np.float32))
    with hytorch.using_config('backward_schedule','unknown'):
        with pytest.raises(ValueError,match='list/heap'):F.sum(x).backward()
