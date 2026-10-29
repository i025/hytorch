"""CPU contracts for opt-in SGD; these tests do not establish GPU correctness."""
from types import SimpleNamespace

import numpy as np
import pytest

from hytorch import Variable, cuda
from hytorch.core import Parameter
from hytorch.optimizers import SGD, FreezeParam, WeightDecay
from hytorch.backends.cuda import runtime


def model_with(values):
    params = [Parameter(value.copy()) for value in values]
    return SimpleNamespace(params=lambda: iter(params)), params


def test_default_stays_original_loop(monkeypatch):
    model, params = model_with([np.arange(7, dtype=np.float32)])
    params[0].grad = Variable(np.ones(7, np.float32))
    calls = []
    opt = SGD(lr=.125).setup(model)
    original = opt.update_one
    monkeypatch.setattr(opt, "update_one", lambda p: (calls.append(p), original(p)))
    opt.update()
    assert opt.fused is False and calls == params
    np.testing.assert_array_equal(params[0].data, np.arange(7, dtype=np.float32)-.125)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_cpu_fused_retains_original_inplace_update(dtype, monkeypatch):
    def forbidden(*args):
        raise AssertionError("CPU SGD must not dispatch to CUDA")
    monkeypatch.setattr(runtime, "multi_tensor_sgd", forbidden, raising=False)
    initial = np.arange(20, dtype=dtype).reshape(4, 5)
    results = []
    for fused in (False, True):
        model, params = model_with([initial])
        # CPU retains the original implementation, including strided inputs.
        param = params[0]
        param.data = param.data[:, ::2]
        param.grad = Variable(np.full(param.shape, .75, dtype))
        pointer, gradient = param.data, param.grad.data.copy()
        opt = SGD(lr=.001, fused=fused).setup(model)
        for _ in range(5):
            opt.update()
        assert param.data is pointer
        np.testing.assert_array_equal(param.grad.data, gradient)
        results.append(param.data.copy())
    np.testing.assert_array_equal(*results)


@pytest.mark.parametrize("fused", [False, True])
def test_hooks_order_missing_gradient_and_freeze(fused):
    model, params = model_with([np.ones(3, np.float32) for _ in range(3)])
    params[0].grad = Variable(np.ones(3, np.float32))
    params[1].grad = Variable(np.ones(3, np.float32))
    seen = []
    opt = SGD(lr=.125, fused=fused).setup(model)
    opt.add_hook(lambda active: seen.append(tuple(active)))
    opt.add_hook(WeightDecay(.5))
    opt.add_hook(FreezeParam(params[1]))
    opt.add_hook(lambda active: seen.append((active[0].grad.data.copy(), active[1].grad)))
    opt.update()
    assert seen[0] == tuple(params[:2])
    np.testing.assert_array_equal(seen[1][0], np.full(3, 1.5, np.float32))
    assert seen[1][1] is None
    np.testing.assert_array_equal(params[0].data, np.full(3, .8125, np.float32))
    np.testing.assert_array_equal(params[1].data, np.ones(3, np.float32))
    np.testing.assert_array_equal(params[2].data, np.ones(3, np.float32))


def test_no_active_parameters_still_calls_hooks(monkeypatch):
    def forbidden(*args):
        raise AssertionError("Empty SGD update must not dispatch")
    monkeypatch.setattr(runtime, "multi_tensor_sgd", forbidden, raising=False)
    model, _ = model_with([np.ones(3, np.float32)])
    seen = []
    opt = SGD(fused=True).setup(model)
    opt.add_hook(lambda active: seen.append(active[:]))
    opt.update()
    assert seen == [[]]


def test_fused_passes_current_gradient_arrays(monkeypatch):
    model, params = model_with([np.ones(3, np.float32)])
    fake_cuda_module = object()
    monkeypatch.setattr(cuda, "get_array_module", lambda value: fake_cuda_module)
    calls = []
    monkeypatch.setattr(runtime, "multi_tensor_sgd",
                        lambda p, g, lr: calls.append((p, g, lr)), raising=False)
    opt = SGD(lr=.125, fused=True).setup(model)
    gradients = [np.full(3, i, np.float32) for i in (1, 2)]
    for gradient in gradients:
        params[0].grad = Variable(gradient)
        opt.update()
    assert len(calls) == 2
    for call, gradient in zip(calls, gradients):
        assert call[0][0] is params[0].data and call[1][0] is gradient
        assert call[2] == .125


def test_fused_rejects_mixed_parameter_devices_before_update(monkeypatch):
    model, params = model_with([np.ones(3, np.float32) for _ in range(2)])
    for param in params:
        param.grad = Variable(np.ones(3, np.float32))
    fake_cuda_module = object()
    monkeypatch.setattr(cuda, "get_array_module",
                        lambda value: np if value is params[0].data else fake_cuda_module)
    with pytest.raises(ValueError, match="all parameters"):
        SGD(fused=True).setup(model).update()
    for param in params:
        np.testing.assert_array_equal(param.data, np.ones(3, np.float32))


def test_hook_failure_propagates_without_updating_parameters():
    model, params = model_with([np.ones(3, np.float32)])
    params[0].grad = Variable(np.ones(3, np.float32))
    def bad_hook(active):
        raise RuntimeError("deliberate hook failure")
    opt = SGD(fused=True).setup(model)
    opt.add_hook(bad_hook)
    with pytest.raises(RuntimeError, match="deliberate hook failure"):
        opt.update()
    np.testing.assert_array_equal(params[0].data, np.ones(3, np.float32))
