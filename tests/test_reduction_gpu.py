"""Prepared GPU acceptance. Disabled unless a device is explicitly allocated."""
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

import hytorch
import hytorch.functions as F
from hytorch.backends.cuda import backend, dispatch_stats
from hytorch.backends.cuda.runtime import sum_array
from test_conv_shape_dispatch_gpu import device, compare

SCRIPTS = Path(__file__).resolve().parents[1]/'docs/experiments/cuda-reduction-foundation/scripts'
spec = importlib.util.spec_from_file_location('sum_cases', SCRIPTS/'cases.py')
cases = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cases)


@pytest.mark.parametrize('shape,axis', cases.CASES)
@pytest.mark.parametrize('keepdims', [False, True])
@pytest.mark.parametrize('nondefault', [False, True])
def test_sum_shapes(device, shape, axis, keepdims, nondefault, request):
    cp, torch = device
    rng = np.random.default_rng(42)
    a = rng.normal(size=shape).astype(np.float32)
    stream = cp.cuda.Stream(non_blocking=True) if nondefault else cp.cuda.Stream.null
    with stream:
        x = cp.asarray(a)
        old = x.sum(axis=axis, keepdims=keepdims)
        dispatch_stats(reset=True)
        with backend(reduction='native'):
            actual = sum_array(x, axis, keepdims)
        consumed = actual + cp.float32(.03125)
    stream.synchronize()
    assert dispatch_stats()['sum_native'] == 1
    tx = torch.tensor(a, device='cuda')
    # torch dim=() means ALL axes; NumPy/CuPy axis=() means NO axes.
    target = tx + 0. if axis == () else torch.sum(tx, dim=axis, keepdim=keepdims)
    compare(request.node.name+'_cupy', actual, old)
    compare(request.node.name+'_torch', actual, target.cpu().numpy())
    compare(request.node.name+'_stream_consumer', consumed, old+cp.float32(.03125))


@pytest.mark.parametrize('size', [65537, 1048577])
def test_retained_cancellation_inputs(device, size, request):
    cp, _ = device
    root = Path(os.environ['HYTORCH_REDUCTION_FAILURE_INPUTS'])
    case = 8 if size == 65537 else 23
    path = next(root.glob('failure-c{}-cancellation-False-*.npz'.format(case)))
    a = np.load(path)['x']
    with backend(reduction='native'):
        actual = sum_array(cp.asarray(a))
    compare(request.node.name+'_cupy', actual, cp.asarray(a).sum())
    compare(request.node.name+'_fp64', actual, a.astype(np.float64).sum().astype(np.float32))


@pytest.mark.parametrize('kind', ['noncontiguous', 'float64', 'empty', 'rank9'])
def test_original_and_materialized_layout(device, kind, request):
    cp, _ = device
    shape = (2,3,5) if kind not in ('empty','rank9') else ((2,0,3) if kind == 'empty' else (1,)*9)
    x = cp.arange(int(np.prod(shape)), dtype=cp.float64 if kind=='float64' else cp.float32).reshape(shape)
    if kind == 'noncontiguous':
        x = x[:, :, ::-1]
    dispatch_stats(reset=True)
    axis = None if kind == 'empty' else -1
    with backend(reduction='native'):
        actual = sum_array(x, axis=axis)
    compare(request.node.name, actual, x.sum(axis=axis), atol=0, rtol=0)
    assert dispatch_stats().get('sum_native',0) == int(kind == 'noncontiguous')


def test_autograd_mean_sumto_repeated_higher_order(device, request):
    cp, torch = device
    a = np.random.default_rng(42).normal(0,.1,(2,3,7)).astype(np.float32)
    for repeat in range(3):
        x = hytorch.Variable(cp.asarray(a))
        with backend(reduction='native'):
            y = F.mean(x*x, axis=(0,2), keepdims=True)
            F.sum(y).backward(create_graph=True)
            first = x.grad
            x.cleargrad()
            F.sum(first).backward()
            reduced = F.sum_to(x, (1,3,1))
        tx = torch.tensor(a,device='cuda',requires_grad=True)
        ty = (tx*tx).mean((0,2),keepdim=True)
        g, = torch.autograd.grad(ty.sum(),tx,create_graph=True)
        gg, = torch.autograd.grad(g.sum(),tx)
        for key, v, ref in [('forward',y.data,ty),('grad',first.data,g),('grad2',x.grad.data,gg),
                            ('sumto',reduced.data,tx.sum((0,2),keepdim=True))]:
            compare(request.node.name+str(repeat)+key,v,ref.detach().cpu().numpy())
