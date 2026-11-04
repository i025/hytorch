"""Explicit real-GPU acceptance; preserve original mask multiplication semantics."""
import numpy as np
import pytest

import hytorch
import hytorch.functions as F
from hytorch.backends.cuda import backend, dispatch_stats
from hytorch.backends.cuda.runtime import relu_backward
from test_conv_shape_dispatch_gpu import device, compare


@pytest.mark.parametrize('shape', [(), (1,), (257,), (3, 7, 11), (4, 24, 31, 98), (16, 32, 33, 37)])
@pytest.mark.parametrize('nondefault', [False, True])
def test_relu_gradient(device, shape, nondefault, request):
    cp, torch = device
    rng = np.random.default_rng(42)
    a, g = [rng.normal(size=shape).astype(np.float32) for _ in range(2)]
    stream = cp.cuda.Stream(non_blocking=True) if nondefault else cp.cuda.Stream.null
    results = []
    with stream:
        for mode in ('original', 'fused'):
            x = hytorch.Variable(cp.asarray(a))
            dispatch_stats(reset=True)
            with backend(relu_backward=mode):
                F.sum(F.relu(x) * cp.asarray(g)).backward()
            results.append(x.grad.data)
        assert dispatch_stats()['relu_backward_fused'] == 1
    stream.synchronize()
    tx = torch.tensor(a, device='cuda', requires_grad=True)
    (torch.relu(tx) * torch.tensor(g, device='cuda')).sum().backward()
    compare(request.node.name + '_old', results[1], results[0])
    compare(request.node.name + '_torch', results[1], tx.grad.detach().cpu().numpy())
    np.testing.assert_array_equal(cp.asnumpy(results[1]), cp.asnumpy(results[0]))


def test_layout_and_repeat(device, request):
    cp, _ = device
    x = cp.arange(7 * 11, dtype=cp.float32).reshape(7, 11).T - 20
    gy = cp.ones((7, 11), cp.float32).T
    for repeat in range(5):
        actual = relu_backward(x, gy)
        compare(request.node.name + '_' + str(repeat), actual, gy * (x > 0))


def test_special_values_and_signed_zero(device):
    cp, _ = device
    x = cp.array([-cp.inf, -1., -0., 0., 1., cp.inf, cp.nan, -1., 1.], cp.float32)
    gy = cp.array([cp.inf, -2., -3., 4., -0., cp.nan, 5., cp.nan, cp.inf], cp.float32)
    actual, expected = cp.asnumpy(relu_backward(x, gy)), cp.asnumpy(gy * (x > 0))
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(np.signbit(actual[~np.isnan(actual)]),
                                  np.signbit(expected[~np.isnan(expected)]))


def test_higher_order_gpu(device):
    cp, _ = device
    x = hytorch.Variable(cp.array([-2., 0., 3.], cp.float32))
    dispatch_stats(reset=True)
    with backend(relu_backward='fused'):
        F.sum(F.relu(x)**2).backward(create_graph=True)
        gx = x.grad
        x.cleargrad()
        F.sum(gx).backward(create_graph=True)
    np.testing.assert_array_equal(cp.asnumpy(x.grad.data), [0., 0., 2.])
    assert dispatch_stats().get('relu_backward_fused', 0) == 0
