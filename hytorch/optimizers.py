from hytorch import cuda
from hytorch.core import Parameter
import math
import numpy as np


# =============================================================================
# Optimizer (base class)
# =============================================================================
class Optimizer:
    def __init__(self):
        self.target = None
        self.hooks = []

    def setup(self, target):
        self.target = target    # model
        return self

    def update(self):
        params = [p for p in self.target.params() if p.grad is not None]
        # 将None之外的参数汇总到列表
        # 预处理（可选）
        for f in self.hooks:
            f(params)

        # FreezeParam hooks can clear gradients after the first filter.
        params = [p for p in params if p.grad is not None]

        # 更新参数
        for param in params:
            self.update_one(param)

    def update_one(self, param):
        raise NotImplementedError

    def add_hook(self, f):
        self.hooks.append(f)


# =============================================================================
# Hook functions
# =============================================================================
class WeightDecay:
    def __init__(self, rate):
        self.rate = rate

    def __call__(self, params):
        for param in params:
            param.grad.data += self.rate * param.data


class ClipGrad:
    def __init__(self, max_norm):
        self.max_norm = max_norm

    def __call__(self, params):
        total_norm = 0
        for param in params:
            total_norm += (param.grad.data ** 2).sum()
        total_norm = math.sqrt(float(total_norm))

        rate = self.max_norm / (total_norm + 1e-6)
        if rate < 1:
            for param in params:
                param.grad.data *= rate


class FreezeParam:
    def __init__(self, *layers):
        self.freeze_params = []
        for l in layers:
            if isinstance(l, Parameter):
                self.freeze_params.append(l)
            else:
                for p in l.params():
                    self.freeze_params.append(p)

    def __call__(self, params):
        for p in self.freeze_params:
            p.grad = None


# =============================================================================
# SGD / MomentumSGD / AdaGrad / AdaDelta / Adam
# =============================================================================
# 随机梯度下降法
class SGD(Optimizer):
    """SGD; fused=True opts into contiguous CUDA FP32 multi-tensor updates.

    CPU parameters retain the original loop. Mixed CPU/CUDA parameter sets and
    unsupported CUDA arrays raise an error instead of silently changing backend.
    """
    def __init__(self, lr=0.01, fused=False):
        super().__init__()
        self.lr = lr
        self.fused = fused

    def update(self):
        if not self.fused:
            return super().update()

        params = [p for p in self.target.params() if p.grad is not None]
        for hook in self.hooks:
            hook(params)
        params = [p for p in params if p.grad is not None]
        if not params:
            return

        modules = [cuda.get_array_module(p.data) for p in params]
        if all(module is np for module in modules):
            for param in params:
                self.update_one(param)
            return
        if any(module is np for module in modules):
            raise ValueError("Fused SGD requires all parameters on CPU or all on CUDA")

        from hytorch.backends.cuda.runtime import multi_tensor_sgd
        multi_tensor_sgd([p.data for p in params],
                         [p.grad.data for p in params], self.lr)

    def update_one(self, param):  # 更新权重和偏移
        param.data -= self.lr * param.grad.data


class MomentumSGD(Optimizer):
    def __init__(self, lr=0.01, momentum=0.9):
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.vs = {}

    def update_one(self, param):
        v_key = id(param)
        if v_key not in self.vs:
            xp = cuda.get_array_module(param.data)
            self.vs[v_key] = xp.zeros_like(param.data)

        v = self.vs[v_key]
        v *= self.momentum
        v -= self.lr * param.grad.data
        param.data += v


class AdaGrad(Optimizer):
    def __init__(self, lr=0.001, eps=1e-8):
        super().__init__()
        self.lr = lr
        self.eps = eps
        self.hs = {}

    def update_one(self, param):
        xp = cuda.get_array_module(param.data)

        h_key = id(param)
        if h_key not in self.hs:
            self.hs[h_key] = xp.zeros_like(param.data)

        lr = self.lr
        eps = self.eps
        grad = param.grad.data
        h = self.hs[h_key]

        h += grad * grad
        param.data -= lr * grad / (xp.sqrt(h) + eps)


class AdaDelta(Optimizer):
    def __init__(self, rho=0.95, eps=1e-6):
        super().__init__()
        self.rho = rho
        self.eps = eps
        self.msg = {}
        self.msdx = {}

    def update_one(self, param):
        xp = cuda.get_array_module(param.data)

        key = id(param)
        if key not in self.msg:
            self.msg[key] = xp.zeros_like(param.data)
            self.msdx[key] = xp.zeros_like(param.data)

        msg, msdx = self.msg[key], self.msdx[key]
        rho = self.rho
        eps = self.eps
        grad = param.grad.data

        msg *= rho
        msg += (1 - rho) * grad * grad
        dx = xp.sqrt((msdx + eps) / (msg + eps)) * grad
        msdx *= rho
        msdx += (1 - rho) * dx * dx
        param.data -= dx


class Adam(Optimizer):
    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__()
        self.t = 0
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.ms = {}
        self.vs = {}

    def update(self, *args, **kwargs):
        self.t += 1
        super().update(*args, **kwargs)

    @property
    def lr(self):
        fix1 = 1. - math.pow(self.beta1, self.t)
        fix2 = 1. - math.pow(self.beta2, self.t)
        return self.alpha * math.sqrt(fix2) / fix1

    def update_one(self, param):
        xp = cuda.get_array_module(param.data)

        key = id(param)
        if key not in self.ms:
            self.ms[key] = xp.zeros_like(param.data)
            self.vs[key] = xp.zeros_like(param.data)

        m, v = self.ms[key], self.vs[key]
        beta1, beta2, eps = self.beta1, self.beta2, self.eps
        grad = param.grad.data

        m += (1 - beta1) * (grad - m)
        v += (1 - beta2) * (grad * grad - v)
        param.data -= self.lr * m / (xp.sqrt(v) + eps)

class AdamW(Optimizer):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999,
                 eps=1e-8, weight_decay=0.01):
        super().__init__()
        self.t = 0
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.ms = {}
        self.vs = {}

    def update(self, *args, **kwargs):
        self.t += 1
        super().update(*args, **kwargs)

    def update_one(self, param):
        xp = cuda.get_array_module(param.data)
        key = id(param)

        if key not in self.ms:
            self.ms[key] = xp.zeros_like(param.data)
            self.vs[key] = xp.zeros_like(param.data)

        m, v = self.ms[key], self.vs[key]
        beta1, beta2, eps = self.beta1, self.beta2, self.eps
        grad = param.grad.data

        # 一阶 / 二阶动量
        m += (1 - beta1) * (grad - m)
        v += (1 - beta2) * (grad * grad - v)

        # bias correction
        m_hat = m / (1 - beta1 ** self.t)
        v_hat = v / (1 - beta2 ** self.t)

        # Adam 更新
        param.data -= self.lr * m_hat / (xp.sqrt(v_hat) + eps)

        # AdamW：decoupled weight decay
        if self.weight_decay > 0:
            param.data -= self.lr * self.weight_decay * param.data

