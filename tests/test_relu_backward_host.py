import numpy as np

import hytorch
import hytorch.functions as F
from hytorch.backends.cuda import backend, current_config, dispatch_stats, runtime
from scripts.build_cuda import build_command


def test_config_and_cpu_semantics():
    before = current_config()
    dispatch_stats(reset=True)
    x = hytorch.Variable(np.array([-2., 0., 3.], np.float32))
    with backend(relu_backward='fused'):
        F.sum(F.relu(x)**2).backward()
    np.testing.assert_array_equal(x.grad.data, [0., 0., 6.])
    assert dispatch_stats()['relu_backward_fallback_cpu_or_dtype'] == 1
    assert current_config() == before
    assert any(p.endswith('/relu_backward.cu') for p in build_command('nvcc', 'test.so', '80'))


def test_higher_order_preserves_graph(monkeypatch):
    def forbidden(*args):
        raise AssertionError('Native gradient must not run when constructing a higher-order graph')
    monkeypatch.setattr(runtime, 'relu_backward', forbidden)
    x = hytorch.Variable(np.array([-2., 0., 3.], np.float32))
    with backend(relu_backward='fused'):
        F.sum(F.relu(x)**2).backward(create_graph=True)
        gx = x.grad
        x.cleargrad()
        F.sum(gx).backward()
    np.testing.assert_array_equal(x.grad.data, [0., 0., 2.])
