import numpy as np
from numpy import transpose

import hytorch
from hytorch import cuda
from hytorch.core import Function
from hytorch import as_variable
from hytorch import utils
from hytorch import as_array


# =============================================================================
# Basic functions: sin / cos / tanh / exp / log
# =============================================================================
class Sin(Function):
    def forward(self, x):   # 变量是ndarray实例，用的是Variable.data->gpu化，用get_array_module获取数据类型
        xp = cuda.get_array_module(x)
        y = xp.sin(x)
        return y

    def backward(self, gy):  # 变量是Variable实例
        x = self.inputs[0]  # 取出 Variable
        gx = gy * cos(x)  # 用cos(x) ,构建计算图 NumPy 只认识 ndarray
        return gx


def sin(x):
    return Sin()(x)


class Cos(Function):
    def forward(self, x):   # 变量是ndarray实例，用的是Variable.data->gpu化，用get_array_module获取数据类型
        xp = cuda.get_array_module(x)
        y = xp.cos(x)
        return y

    def backward(self, gy):  # 变量是Variable实例
        x = self.inputs[0]
        gx = -gy*sin(x)     # 构建计算图
        return gx


def cos(x):
    return Cos()(x)


class Tanh(Function):
    def forward(self, x):   # 变量是ndarray实例，用的是Variable.data->gpu化，用get_array_module获取数据类型
        xp = cuda.get_array_module(x)
        y = xp.tanh(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()       # weakref()
        gx = gy*(1 - y*y)
        return gx


def tanh(x):
    return Tanh()(x)


class Exp(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.exp(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()  # weakref
        gx = gy * y
        return gx


def exp(x):
    return Exp()(x)


class Log(Function):
    def forward(self, x):           # 变量是ndarray实例，用的是Variable.data->gpu化，用get_array_module获取数据类型
        xp = cuda.get_array_module(x)
        y = xp.log(x)

        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy / x
        return gx


def log(x):
    return Log()(x)


# =============================================================================
# Tensor operations: reshape / transpose / get_item / expand_dims / flatten
# =============================================================================
class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y

    def backward(self, gy):
        return reshape(gy, self.x_shape)


def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)


# 矩阵的转置
class Transpose(Function):
    def __init__(self, axes=None):
        self.axes = axes

    def forward(self, x):
        y = x.transpose(self.axes)
        return y

    def backward(self, gy):
        if self.axes is None:
            return transpose(gy)

        axes_len = len(self.axes)
        inv_axes = tuple(np.argsort([ax % axes_len for ax in self.axes]))
        '''先把所有轴索引转成非负数，再求逆映射，防止负数轴索引导致排序错误。
        axes = [2, 0, 1]
        原始轴顺序：0, 1, 2
        转置后轴顺序：2, 0, 1
        逆映射是：[1, 2, 0]
        '''
        return transpose(gy, inv_axes)


def transpose(x, axes=None):
    return Transpose(axes)(x)


class GetItem(Function):
    def __init__(self, slices):
        self.slices = slices

    def forward(self, x):
        y = x[self.slices]
        return y

    def backward(self, gy):
        x, = self.inputs
        f = GetItemGrad(self.slices, x.shape)
        return f(gy)


class GetItemGrad(Function):
    def __init__(self, slices, in_shape):
        self.slices = slices
        self.in_shape = in_shape

    def forward(self, gy):
        xp = cuda.get_array_module(gy)
        gx = xp.zeros(self.in_shape, dtype=gy.dtype)
        if xp is np:
            np.add.at(gx, self.slices, gy)
        else:
            gx.scatter_add(self.slices, gy)   # ?

        return gx

    def backward(self, ggx):
        return get_item(ggx, self.slices)


def get_item(x, slices):
    f = GetItem(slices)
    return f(x)




def flatten(x):
    """Flattens the input. Does not affect the batch size."""
    return reshape(x, (x.shape[0], -1))


# =============================================================================
# sum / sum_to / broadcast_to / average / matmul / linear
# =============================================================================
class Sum(Function):
    def __init__(self, axis, keepdims):   # axis=0,行相加，axis=1， 列相加。keepdims=False，y为标量，反之保留轴的数量
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy):
        gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        gx = broadcast_to(gy, self.x_shape)    # 广播机制，使x.shape = gx.shape
        return gx


def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)


class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        xp = cuda.get_array_module(x)
        y = xp.broadcast_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = sum_to(gy, self.x_shape)    # broadcast_to的反向传播是sum_to
        return gx


def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)


class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = utils.sum_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = broadcast_to(gy, self.x_shape)     # sum_to的反向传播是broadcast_to,两者相互依赖
        return gx


def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)


def average(x, axis=None, keepdims=False):
    x = as_variable(x)
    y = sum(x, axis, keepdims)
    return y * (y.data.size / x.data.size)

mean = average


class MatMul(Function):
    def __init__(self, native=True):
        self.native = native

    def forward(self, x, w):
        from hytorch.backends.cuda.runtime import selected, gemm, current_config
        mode = selected("matmul", x) if self.native else None
        if mode is not None:
            return gemm(x, w, variant=mode, tile=current_config()["tile"])
        y = x.dot(w)   # 矩阵的乘法运算，其中包括：向量内积、多维矩阵乘法、矩阵与向量的乘法。
        return y

    def backward(self, gy):
        x, w = self.inputs
        gx = MatMul(native=False)(gy, w.T)
        gw = MatMul(native=False)(x.T, gy)    # Preserve existing backward backend.
        return gx, gw


def matmul(x, w):
    return MatMul()(x, w)


# =============================
# Batch MatMul (BMM): (B,M,K) @ (B,K,N) -> (B,M,N)
# =============================
def _swap_last2_axes(ndim):
    # (..., i, j) -> (..., j, i)
    axes = list(range(ndim))
    axes[-1], axes[-2] = axes[-2], axes[-1]
    return tuple(axes)

class BatchMatMul(Function):
    def forward(self, x, w):
        # x: (B,M,K), w: (B,K,N)  (不支持广播；Transformer 里我们会显式 reshape 成同 batch)
        xp = cuda.get_array_module(x)
        y = xp.matmul(x, w)  # 关键：matmul 才是 batch 语义
        return y

    def backward(self, gy):
        x, w = self.inputs

        # 转置只交换最后两维（不能用 .T，因为 .T 会反转所有维度）
        wT = transpose(w, _swap_last2_axes(w.ndim))
        xT = transpose(x, _swap_last2_axes(x.ndim))

        gx = bmm(gy, wT)
        gw = bmm(xT, gy)
        return gx, gw

def bmm(x, w):
    return BatchMatMul()(x, w)


def linear_simple(x, W, b=None):
    t = matmul(x, W)
    if b is None:
        return t
    y = b + t
    t.data = None       # matmul的反向传播并不需要t的值，引用次数变成0，被python解释器删除
    return y


class Linear(Function):
    def forward(self, x, W, b):
        from hytorch.backends.cuda.runtime import selected, gemm, current_config
        mode = selected("matmul", x)
        y = (x.dot(W) if mode is None else
             gemm(x, W, variant=mode, tile=current_config()["tile"]))
        if b is not None:
            y += b
        return y

    def backward(self, gy):
        x, W, b = self.inputs
        gb = None if b.data is None else sum_to(gy, b.shape)
        gx = MatMul(native=False)(gy, W.T)
        gW = MatMul(native=False)(x.T, gy)
        return gx, gW, gb


def linear(x, W, b=None):
    return Linear()(x, W, b)


# =============================================================================
# activation function: sigmoid / relu / softmax / log_softmax / leaky_relu
# =============================================================================

def sigmoid_simple(x):
    x = as_variable(x)
    return 1/(1 + exp(-x))


class Sigmoid(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        # y = 1 / (1 + xp.exp(-x))
        y = xp.tanh(x * 0.5) * 0.5 + 0.5  # Better implementation
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * y * (1 - y)
        return gx


def sigmoid(x):
    return Sigmoid()(x)


class ReLU(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.maximum(0.0, x)
        return y

    def backward(self, gy):
        x, = self.inputs
        mask = x.data > 0
        gx = gy * mask
        return gx


def relu(x):
    return ReLU()(x)


# 多分类
def softmax_simple(x, axis=1):
    x = as_variable(x)
    y = exp(x)
    sum_y = sum(y, axis=axis, keepdims=True)
    return y / sum_y


class Softmax(Function):
    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = x - x.max(axis=self.axis, keepdims=True)
        y = xp.exp(y)
        y /= y.sum(axis=self.axis, keepdims=True)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = y * gy
        sumdx = gx.sum(axis=self.axis, keepdims=True)
        gx -= y * sumdx
        return gx


def softmax(x, axis=1):
    return Softmax(axis)(x)

class LogSoftmax(Function):
    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, x):
        log_z = utils.logsumexp(x, self.axis)
        y = x - log_z
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy - exp(y) * gy.sum(axis=self.axis, keepdims=True)
        return gx


def log_softmax(x, axis=1):
    return LogSoftmax(axis)(x)


class LeakyReLU(Function):
    def __init__(self, slope):
        self.slope = slope

    def forward(self, x):
        y = x.copy()
        y[x <= 0] *= self.slope
        return y

    def backward(self, gy):
        x, = self.inputs
        mask = (x.data > 0).astype(gy.dtype)
        mask[mask <= 0] = self.slope
        gx = gy * mask
        return gx


def leaky_relu(x, slope=0.2):
    return LeakyReLU(slope)(x)


# =============================================================================
# loss function: mean_squared_error / softmax_cross_entropy / sigmoid_cross_entropy / binary_cross_entropy
# =============================================================================
def mean_square_error_simple(x0, x1):
    diff = as_variable(x0) - as_variable(x1)   # 不是继承Function的函数，手动Variable
    return sum(diff**2)/len(diff)


class MeanSquareError(Function):
    def forward(self, x0, x1):
        diff = x0 - x1
        y = (diff**2).sum()/len(diff)
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        diff = x0 - x1
        gx0 = gy * diff*(2./len(diff))
        gx1 = -gx0
        return gx0, gx1


def mean_square_error(x0, x1):
    return MeanSquareError()(x0, x1)


# softmax和交叉熵混合
def softmax_cross_entropy_simple(x, t):
    x, t = as_variable(x), as_variable(t)
    N = x.shape[0]

    p = softmax_simple(x)
    p = clip(p, 1e-15, 1.0)  # 防止log0的现象，p>max,返回max，p<min,返回min，中间就返回自己
    log_p = log(p)
    # print(log_p.shape)
    tlog_p = log_p[np.arange(N), t.data]  # np.arange(N) → array([0, 1, …, N-1]) ,不是array
    y = -1*sum(tlog_p) / N
    return y


def softmax_cross_entropy_soft_label(x, t):
    """Soft-label cross entropy for MixUp/CutMix targets.

    ``t`` is expected to have the same shape as ``x`` and each row should
    represent a probability distribution over classes.
    """
    x, t = as_variable(x), as_variable(t)
    if x.shape != t.shape:
        raise ValueError(f"soft labels must have shape {x.shape}, got {t.shape}")
    log_p = log_softmax(x, axis=1)
    return -sum(t * log_p) / x.shape[0]


class SoftmaxCrossEntropy(Function):
    def forward(self, x, t):
        N = x.shape[0]
        log_z = utils.logsumexp(x, axis=1)
        log_p = x - log_z
        log_p = log_p[np.arange(N), t.ravel()]
        y = -log_p.sum() / np.float32(N)
        return y

    def backward(self, gy):
        x, t = self.inputs
        N, CLS_NUM = x.shape

        gy *= 1/N
        y = softmax(x)
        # convert to one-hot
        xp = cuda.get_array_module(t.data)
        t_onehot = xp.eye(CLS_NUM, dtype=t.dtype)[t.data]
        y = (y - t_onehot) * gy
        return y


def softmax_cross_entropy(x, t):
    return SoftmaxCrossEntropy()(x, t)


def sigmoid_cross_entropy(x, t):
    if x.ndim != t.ndim:
        t = t.reshape(*x.shape)
    x, t = as_variable(x), as_variable(t)
    N = len(x)
    p = sigmoid(x)
    p = clip(p, 1e-15, 1.0)
    tlog_p = t * log(p) + (1 - t) * log(1 - p)
    y = -1 * sum(tlog_p) / N
    return y


def binary_cross_entropy(p, t):
    if p.ndim != t.ndim:
        t = t.reshape(*p.shape)
    N = len(t)
    p = clip(p, 1e-15, 0.999)
    tlog_p = t * log(p) + (1 - t) * log(1 - p)
    y = -1 * sum(tlog_p) / N
    return y


# =============================================================================
# accuracy / dropout / batch_norm / embed_id，这里的batch_norm暂时不用管
# =============================================================================
# 评估精度的函数，用的是ndarray，不能对其求导
def accuracy(y, t):
    y, t = as_variable(y), as_variable(t)

    pred = y.data.argmax(axis=1).reshape(t.shape)
    result = (pred == t.data)
    acc = result.mean()
    return as_variable(as_array(acc))


# Inverted Dropout,训练时以概率 dropout_ratio 随机将输入元素置 0，并放大幸存值 1/(1-r) 倍，保持期望值不变；测试时直接原样返回。
def dropout(x, dropout_ratio=0.5):
    x = as_variable(x)

    if hytorch.Config.train:
        xp = cuda.get_array_module(x)
        mask = xp.random.rand(*x.shape) > dropout_ratio
        scale = xp.array(1.0 - dropout_ratio).astype(x.dtype)
        y = x * mask / scale
        return y
    else:
        return x


def embed_id(x, W):
    return W[x]


class BatchNorm(Function):
    def __init__(self, mean, var, decay, eps):
        self.avg_mean = mean
        self.avg_var = var
        self.decay = decay
        self.eps = eps
        self.inv_std = None

    def forward(self, x, gamma, beta):
        assert x.ndim == 2 or x.ndim == 4

        x_ndim = x.ndim
        if x_ndim == 4:
            N, C, H, W = x.shape
            # (N, C, H, W) -> (N*H*W, C)
            x = x.transpose(0, 2, 3, 1).reshape(-1, C)

        xp = cuda.get_array_module(x)

        if hytorch.Config.train:
            mean = x.mean(axis=0)
            var = x.var(axis=0)
            inv_std = 1 / xp.sqrt(var + self.eps)
            xc = (x - mean) * inv_std

            m = x.size // gamma.size
            s = m - 1. if m - 1. > 1. else 1.
            adjust = m / s  # unbiased estimation
            self.avg_mean *= self.decay
            self.avg_mean += (1 - self.decay) * mean
            self.avg_var *= self.decay
            self.avg_var += (1 - self.decay) * adjust * var
            self.inv_std = inv_std
        else:
            inv_std = 1 / xp.sqrt(self.avg_var + self.eps)
            xc = (x - self.avg_mean) * inv_std
        y = gamma * xc + beta

        if x_ndim == 4:
            # (N*H*W, C) -> (N, C, H, W)
            y = y.reshape(N, H, W, C).transpose(0, 3, 1, 2)
        return y

    def backward(self, gy):
        gy_ndim = gy.ndim
        if gy_ndim == 4:
            N, C, H, W = gy.shape
            gy = gy.transpose(0, 2, 3, 1).reshape(-1, C)

        x, gamma, beta = self.inputs
        batch_size = len(gy)

        if x.ndim == 4:
            N, C, H, W = x.shape
            x = x.transpose(0, 2, 3, 1).reshape(-1, C)
        mean = x.sum(axis=0) / batch_size
        xc = (x - mean) * self.inv_std

        gbeta = sum(gy, axis=0)
        ggamma = sum(xc * gy, axis=0)
        gx = gy - gbeta / batch_size - xc * ggamma / batch_size
        gx *= gamma * self.inv_std

        if gy_ndim == 4:
            gx = gx.reshape(N, H, W, C).transpose(0, 3, 1, 2)
        return gx, ggamma, gbeta


def batch_norm(x, gamma, beta, mean, var, decay=0.9, eps=2e-5):
    return BatchNorm(mean, var, decay, eps)(x, gamma, beta)


# =============================================================================
# max / min / clip
# =============================================================================
class Max(Function):
    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        y = x.max(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy):
        x = self.inputs[0]
        y = self.outputs[0]()  # weakref

        shape = utils.max_backward_shape(x, self.axis)
        gy = reshape(gy, shape)
        y = reshape(y, shape)
        cond = (x.data == y.data)
        gy = broadcast_to(gy, cond.shape)
        return gy * cond


class Min(Max):
    def forward(self, x):
        y = x.min(axis=self.axis, keepdims=self.keepdims)
        return y


def max(x, axis=None, keepdims=False):
    return Max(axis, keepdims)(x)


def min(x, axis=None, keepdims=False):
    return Min(axis, keepdims)(x)


class Clip(Function):
    def __init__(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max

    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.clip(x, self.x_min, self.x_max)
        return y

    def backward(self, gy):
        x, = self.inputs
        mask = (x.data >= self.x_min) * (x.data <= self.x_max)
        gx = gy * mask
        return gx


def clip(x, x_min, x_max):
    return Clip(x_min, x_max)(x)


class Concat(Function):
    def __init__(self, axis=0):
        self.axis = axis

    def forward(self, *xs):
        # 将多个 Variable 的 data 提取出来
        xp = cuda.get_array_module(xs[0].data)
        xs_data = [x.data for x in xs]
        y = xp.concatenate(xs_data, axis=self.axis)
        return y

    def backward(self, gy):
        xs = self.inputs
        axis = self.axis

        # 计算每个输入在 axis 维度上的大小
        sizes = np.array([x.shape[axis] for x in xs], dtype=np.int32)
        indices = np.cumsum(sizes)

        gxs = []
        start = 0
        for i, x in enumerate(xs):
            end = indices[i]
            slices = [slice(None)] * gy.ndim
            slices[axis] = slice(start, end)
            gx = gy[tuple(slices)]
            gxs.append(gx)
            start = end

        return tuple(gxs)

def concat(xs, axis=0):
    if not isinstance(xs, (list, tuple)):
        xs = [xs]
    return Concat(axis)(*xs)
# =============================================================================
# conv2d / col2im / im2col / basic_math
# =============================================================================
from hytorch.functions_conv import conv2d
from hytorch.functions_conv import deconv2d
from hytorch.functions_conv import conv2d_simple
from hytorch.functions_conv import im2col
from hytorch.functions_conv import col2im
from hytorch.functions_conv import pooling_simple
from hytorch.functions_conv import pooling
from hytorch.functions_conv import average_pooling
from hytorch.core import add
from hytorch.core import sub
from hytorch.core import rsub
from hytorch.core import mul
from hytorch.core import div
from hytorch.core import neg
from hytorch.core import pow
