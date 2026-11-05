"""CPU-only sum planning, semantics and autograd; no native launch is executed."""
import ctypes as ct
import importlib.util
from pathlib import Path

import numpy as np
import pytest

import hytorch
import hytorch.functions as F
from hytorch.backends.cuda import backend, current_config, dispatch_stats, runtime
from hytorch.backends.cuda.reduction import SumLayout, plan_sum
from scripts.build_cuda import build_command

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('reduction_cases', ROOT/'docs/experiments/cuda-reduction-foundation/scripts/cases.py')
cases = importlib.util.module_from_spec(spec); spec.loader.exec_module(cases)


@pytest.mark.parametrize('shape,axis', cases.CASES)
@pytest.mark.parametrize('keepdims', [False, True])
def test_numpy_semantics_and_cpu_native_selection(shape, axis, keepdims, monkeypatch):
    def forbidden(*args):
        raise AssertionError('CPU tests must not load CUDA or launch a kernel')
    monkeypatch.setattr(runtime, 'load_library', forbidden)
    x = (np.arange(np.prod(shape, dtype=int), dtype=np.float32)%17).reshape(shape)
    layout, output, parts = plan_sum(shape, axis, keepdims)
    expected = x.sum(axis=axis, keepdims=keepdims)
    assert output == expected.shape
    assert layout.outputs == expected.size and layout.reduction * layout.outputs == x.size
    assert parts >= 1
    dispatch_stats(reset=True)
    with backend(reduction='native'):
        actual = F.sum(hytorch.Variable(x), axis=axis, keepdims=keepdims)
    np.testing.assert_array_equal(actual.data, expected)
    assert dispatch_stats()['reduction_fallback_cpu_or_dtype'] == 1


@pytest.mark.parametrize('axis', [(0,0), 3, -4])
def test_invalid_axes(axis):
    with pytest.raises((ValueError, np.AxisError)):
        plan_sum((2,3,4), axis)


def test_layout_abi_and_default():
    assert ct.sizeof(SumLayout) == 144
    assert current_config()['reduction'] == 'original'
    with pytest.raises(ValueError, match='reduction'):
        with backend(reduction='unknown'):
            pass
    assert any(p.endswith('/reduction.cu') for p in build_command('nvcc','test.so','80'))


@pytest.mark.parametrize('shape', [(0,), (2,0,3), (1,)*9])
def test_cpu_empty_and_rank_keep_original(shape):
    x = np.ones(shape,np.float32)
    with backend(reduction='native'):
        np.testing.assert_array_equal(F.sum(x).data, x.sum())


def test_mean_sum_to_broadcast_and_higher_order():
    x = hytorch.Variable(np.arange(24,dtype=np.float32).reshape(2,3,4)/20)
    with backend(reduction='native'):
        mean = F.mean(x,axis=(0,2),keepdims=True)
        np.testing.assert_allclose(mean.data,x.data.mean((0,2),keepdims=True),atol=1e-6,rtol=1e-5)
        reduced = F.sum_to(x, (1,3,1))
        np.testing.assert_array_equal(reduced.data,x.data.sum((0,2),keepdims=True))
        F.sum(x*x).backward(create_graph=True)
        first=x.grad
        np.testing.assert_array_equal(first.data,2*x.data)
        x.cleargrad();F.sum(first).backward()
        np.testing.assert_array_equal(x.grad.data,np.full(x.shape,2,np.float32))


def test_default_and_explicit_routing(monkeypatch):
    x=np.arange(12,dtype=np.float32).reshape(3,4)
    calls=[]
    monkeypatch.setattr(runtime,'selected',lambda op,x: 'native' if current_config()['reduction']=='native' else None)
    def modeled(x,axis=None,keepdims=False):
        calls.append((x.shape,axis,keepdims))
        return x.sum(axis=axis,keepdims=keepdims)
    monkeypatch.setattr(runtime,'reduce_sum',modeled)
    F.sum(x)
    assert not calls
    with backend(reduction='native'):
        F.sum(x,axis=0);F.sum_to(x,(1,4));F.mean_square_error(x,x+1)
    assert len(calls)==3
