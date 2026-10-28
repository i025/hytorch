from hytorch.core import Parameter
import weakref
import numpy as np
import hytorch.functions as F
from hytorch.core import Parameter
from hytorch import cuda
import os
from hytorch.utils import pair
from hytorch import as_variable
import hytorch


# 与Function类似，持有参数并使用这些参数进行变换
class Layer:
    def __init__(self):
        self._params = set()   # 保存Layer实例拥有的实例

    # __setattr__ 是属性赋值的“总闸门”，能在赋值瞬间插入任何自定义逻辑.
    def __setattr__(self, name, value):
        if isinstance(value, (Parameter, Layer)):    # 添加了Layer层，使Layer能嵌套Layer
            self._params.add(name)     # 添加的是name而不是value，保存到外部文件更方便
        super().__setattr__(name, value)
# __setattr__ 里再次出现 self.attr = value 这种“点号赋值”语法，就会再次触发 __setattr__，从而无限递归。super 可以绕过本层的钩子

    def __call__(self, *inputs,**kwargs):
        outputs = self.forward(*inputs,**kwargs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, inputs):
        raise NotImplementedError()

    def params(self):
        for name in self._params:
            obj = self.__dict__[name]

            if isinstance(obj, Layer):
                yield from obj.params()      # 通过yield from来使用一个生成器来创建另一个生成器,一层层找到最终的参数，找到一个就递给你一个
            else:
                yield obj       # 边干活边吐数据”，内存里永远不会一次性囤完所有结果。暂停处理并返回值。而return是结束处理并返回值

    def cleargrads(self):
        for param in self.params():
            param.cleargrad()        # 对Layer拥有的所有参数调用cleargrad（）,如果参数还是 Layer，会继续深挖，直到Parameter再吐出来

    def to_cpu(self):
        for param in self.params():   # 同理
            param.to_cpu()

    def to_gpu(self):
        for param in self.params():   # 同理
            param.to_gpu()

    def _flatten_params(self, params_dict, parent_key=""):    # 以扁平的结构取出Parameter，返回字典
        for name in self._params:
            obj = self.__dict__[name]
            key = parent_key + '/' + name if parent_key else name

            if isinstance(obj, Layer):
                obj._flatten_params(params_dict, key)    # 如果是Layer就递归深挖到Parameter
            else:
                params_dict[key] = obj

    def save_weights(self, path):
        self.to_cpu()  # 确保数据在主内存中，数据是np.ndarray

        params_dict = {}
        self._flatten_params(params_dict)
        array_dict = {key: param.data for key, param in params_dict.items()
                      if param is not None}
        try:
            np.savez_compressed(path, **array_dict)
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(path):
                os.remove(path)
            raise
        # 处理由用户发起的键盘中断，此时正在保存的文件会被删除，防止创建或加载不完整的文件

    def load_weights(self, path):   # 载入自定义参数
        npz = np.load(path)
        params_dict = {}
        self._flatten_params(params_dict)
        for key, param in params_dict.items():
            param.data = npz[key]


class Linear(Layer):
    def __init__(self, out_size, nobias=False, dtype=np.float32, in_size=None):
        super().__init__()    # Linear.__init__ 里显式调用 Layer.__init__，让父类有机会初始化 _params 集合
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype

        self.W = Parameter(None, name='W')
        if self.in_size is not None:     # 如果没有指定in_size，则延后处理
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_size, dtype=dtype), name='b')

    def _init_W(self, xp=np):
        I, O = self.in_size, self.out_size
        W_data = xp.random.randn(I, O).astype(self.dtype)*np.dtype(self.dtype).type(np.sqrt(1/I))
        # “把权重初始化为均值为 0、方差为 1/I 的高斯随机数，保证前向传播时方差恒为 1，避免梯度爆炸/消失。”
        self.W.data = W_data

    def forward(self, x):
        # 在传播数据是初始化权重
        if self.W.data is None:
            self.in_size = x.shape[1]
            xp = cuda.get_array_module(x)
            self._init_W(xp)

        y = F.linear_simple(x, self.W, self.b)
        return y


class TwoLayerNet(Layer):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = Linear(hidden_size)
        self.l2 = Linear(out_size)

    def forward(self, x):
        y = F.sigmoid_simple(self.l1(x))
        y = self.l2(y)
        return y


class Conv2d(Layer):
    def __init__(self, out_channels, kernel_size, stride=1,
                 pad=0, nobias=False, dtype=np.float32, in_channels=None):
        """Two-dimensional convolutional layer.

        Args:
            out_channels (int): Number of channels of output arrays.
            kernel_size (int or (int, int)): Size of filters.
            stride (int or (int, int)): Stride of filter applications.
            pad (int or (int, int)): Spatial padding width for input arrays.
            nobias (bool): If `True`, then this function does not use the bias.
            in_channels (int or None): Number of channels of input arrays. If
            `None`, parameter initialization will be deferred until the first
            forward data pass at which time the size will be determined.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.dtype = dtype

        self.W = Parameter(None, name='W')
        if in_channels is not None:
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_channels, dtype=dtype), name='b')

    def _init_W(self, xp=np):
        C, OC = self.in_channels, self.out_channels
        KH, KW = pair(self.kernel_size)
        scale = np.dtype(self.dtype).type(np.sqrt(1 / (C * KH * KW)))
        W_data = xp.random.randn(OC, C, KH, KW).astype(self.dtype) * scale
        self.W.data = W_data

    def forward(self, x):
        if self.W.data is None:
            self.in_channels = x.shape[1]
            xp = cuda.get_array_module(x)
            self._init_W(xp)

        y = F.conv2d(x, self.W, self.b, self.stride, self.pad)
        return y


class Deconv2d(Layer):
    def __init__(self, out_channels, kernel_size, stride=1,
                 pad=0, nobias=False, dtype=np.float32, in_channels=None):
        """Two-dimensional deconvolutional (transposed convolution)layer.

        Args:
            out_channels (int): Number of channels of output arrays.
            kernel_size (int or (int, int)): Size of filters.
            stride (int or (int, int)): Stride of filter applications.
            pad (int or (int, int)): Spatial padding width for input arrays.
            nobias (bool): If `True`, then this function does not use the bias.
            in_channels (int or None): Number of channels of input arrays. If
            `None`, parameter initialization will be deferred until the first
            forward data pass at which time the size will be determined.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.dtype = dtype

        self.W = Parameter(None, name='W')
        if in_channels is not None:
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_channels, dtype=dtype), name='b')

    def _init_W(self, xp=np):
        C, OC = self.in_channels, self.out_channels
        KH, KW = pair(self.kernel_size)
        scale = np.sqrt(1 / (C * KH * KW))
        W_data = xp.random.randn(C, OC, KH, KW).astype(self.dtype) * scale
        self.W.data = W_data

    def forward(self, x):
        if self.W.data is None:
            self.in_channels = x.shape[1]
            xp = cuda.get_array_module(x)
            self._init_W(xp)

        y = F.deconv2d(x, self.W, self.b, self.stride, self.pad)
        return y


# dropout_layers.py

def _get_rng(x, seed=None):
    xp = cuda.get_array_module(x)
    if seed is None:
        return xp.random
    # 与 numpy/cupy 兼容的 RandomState
    try:
        return xp.random.RandomState(seed)
    except AttributeError:
        # 旧版本 fallback
        return xp.random


def dropout(x, p=0.5, seed=None):
    """Inverted Dropout（适合全连接/通用张量）
    训练: 以概率 p 置零，其余放大 1/(1-p)
    推理: 原样返回
    """
    x = as_variable(x)
    if not hytorch.Config.train or p <= 0:
        return x
    if p >= 1.0:
        # 全丢弃避免除零；与 PyTorch 行为对齐，直接返回 0
        return x * 0

    xp = cuda.get_array_module(x)
    rng = _get_rng(x, seed)
    mask = rng.rand(*x.shape) > p
    scale = xp.array(1.0 - p, dtype=x.dtype)
    return x * mask / scale


class Dropout(Layer):
    """通用 Dropout（MLP/任意张量）"""
    def __init__(self, p=0.5, seed=None):
        super().__init__()
        self.p = float(p)
        self.seed = seed

    def forward(self, x):
        return dropout(x, self.p, self.seed)


class Dropout2d(Layer):
    """Spatial Dropout / Dropout2d（CNN）
    按通道整块丢弃：对每个样本的每个通道采一个 0/1，广播到 H,W
    输入形状: (N, C, H, W)
    """
    def __init__(self, p=0.5, seed=None):
        super().__init__()
        self.p = float(p)
        self.seed = seed

    def forward(self, x):
        x = as_variable(x)
        if not hytorch.Config.train or self.p <= 0:
            return x
        if self.p >= 1.0:
            return x * 0

        xp = cuda.get_array_module(x)
        rng = _get_rng(x, self.seed)

        N, C = x.shape[0], x.shape[1]
        # (N,C,1,1) -> 广播到 (N,C,H,W)
        mask = rng.rand(N, C, 1, 1) > self.p
        scale = xp.array(1.0 - self.p, dtype=x.dtype)
        return x * mask / scale


from hytorch.functions_conv import batch_norm


class BatchNorm2d(Layer):
    """2D Batch Normalization"""
    def __init__(self, out_channels, eps=1e-3, momentum=0.9):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.gamma = Parameter(np.ones((1, out_channels, 1, 1), dtype=np.float32))
        self.beta = Parameter(np.zeros((1, out_channels, 1, 1), dtype=np.float32))
        # 先用 numpy 初始化，第一次 forward 再迁移到 x 的设备
        self.running_mean = np.zeros((1, out_channels, 1, 1), dtype=np.float32)
        self.running_var = np.ones((1, out_channels, 1, 1), dtype=np.float32)

    def forward(self, x):
        # ---- 设备对齐：把滑动均值方差迁移到与输入 x 相同的设备 ----
        xp = cuda.get_array_module(x.data)
        is_mean_numpy = isinstance(self.running_mean, np.ndarray)
        is_var_numpy = isinstance(self.running_var,  np.ndarray)

        if xp is np:
            # 目标设备是 CPU：若目前在 GPU，则搬回 CPU
            if not is_mean_numpy:
                self.running_mean = cuda.as_numpy(self.running_mean)
                self.running_var = cuda.as_numpy(self.running_var)
        else:
            # 目标设备是 GPU：若目前在 CPU，则搬到 GPU
            if is_mean_numpy:
                self.running_mean = cuda.as_cupy(self.running_mean)
                self.running_var = cuda.as_cupy(self.running_var)

        return batch_norm(
            x, self.gamma, self.beta,
            self.running_mean, self.running_var,
            eps=self.eps, momentum=self.momentum
        )


# =============================================================================
# EmbedID / BatchNorm
# =============================================================================
class EmbedID(Layer):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.W = Parameter(np.random.randn(in_size, out_size), name='W')

    def __call__(self, x):
        y = self.W[x]
        return y


class BatchNorm(Layer):
    def __init__(self):
        super().__init__()
        # `.avg_mean` and `.avg_var` are `Parameter` objects, so they will be
        # saved to a file (using `save_weights()`).
        # But they don't need grads, so they're just used as `ndarray`.
        self.avg_mean = Parameter(None, name='avg_mean')
        self.avg_var = Parameter(None, name='avg_var')
        self.gamma = Parameter(None, name='gamma')
        self.beta = Parameter(None, name='beta')

    def _init_params(self, x):
        xp = cuda.get_array_module(x)
        D = x.shape[1]
        if self.avg_mean.data is None:
            self.avg_mean.data = xp.zeros(D, dtype=x.dtype)
        if self.avg_var.data is None:
            self.avg_var.data = xp.ones(D, dtype=x.dtype)
        if self.gamma.data is None:
            self.gamma.data = xp.ones(D, dtype=x.dtype)
        if self.beta.data is None:
            self.beta.data = xp.zeros(D, dtype=x.dtype)

    def __call__(self, x):
        if self.avg_mean.data is None:
            self._init_params(x)
        return F.batch_norm(x, self.gamma, self.beta, self.avg_mean.data,
                            self.avg_var.data)



#层归一化
class LayerNorm(Layer):
    """
    Layer Normalization.

    - 对每个样本、在最后若干维(=normalized_shape)上做归一化
    - 公式：y = (x - mean) / sqrt(var + eps) * gamma + beta
    """
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, dtype=np.float32):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.dtype = dtype

        if elementwise_affine:
            self.gamma = Parameter(np.ones(self.normalized_shape, dtype=dtype), name='gamma')
            self.beta  = Parameter(np.zeros(self.normalized_shape, dtype=dtype), name='beta')
        else:
            self.gamma = None
            self.beta = None

    def forward(self, x):
        x = as_variable(x)

        # 检查最后几维是否匹配 normalized_shape
        if len(self.normalized_shape) > x.ndim:
            raise ValueError(f"LayerNorm: normalized_shape={self.normalized_shape} "
                             f"has more dims than input ndim={x.ndim}.")
        if x.shape[-len(self.normalized_shape):] != self.normalized_shape:
            raise ValueError(f"LayerNorm: input shape tail {x.shape[-len(self.normalized_shape):]} "
                             f"!= normalized_shape {self.normalized_shape}")

        # LN：对最后 len(normalized_shape) 个维度求均值方差
        axes = tuple(range(x.ndim - len(self.normalized_shape), x.ndim))
        mu = F.mean(x, axis=axes, keepdims=True)                 # :contentReference[oaicite:2]{index=2}
        xc = x - mu
        var = F.mean(xc * xc, axis=axes, keepdims=True)

        # inv_std = 1 / sqrt(var + eps)  -> 用幂运算实现 sqrt
        inv_std = (var + self.eps) ** (-0.5)
        x_hat = xc * inv_std

        if not self.elementwise_affine:
            return x_hat

        # gamma/beta reshape 成可广播形状：(1,1,..., normalized_shape...)
        prefix = (1,) * (x.ndim - len(self.normalized_shape))
        affine_shape = prefix + self.normalized_shape
        gamma = F.reshape(self.gamma, affine_shape)
        beta  = F.reshape(self.beta,  affine_shape)
        return x_hat * gamma + beta



####tran_layer
class LinearND(Layer):
    """
    支持 (N,D) / (N,T,D) / (...,D) 的 Linear
    - 只把最后一维当特征维 D
    - 前面维度全部展平做一次线性，再 reshape 回来
    """
    def __init__(self, in_size, out_size, nobias=False, dtype=np.float32):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype

        self.W = Parameter(None, name='W')
        self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_size, dtype=dtype), name='b')

    def _init_W(self, xp=np):
        I, O = self.in_size, self.out_size
        W_data = xp.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)
        self.W.data = W_data

    def forward(self, x):
        x = as_variable(x)
        xp = cuda.get_array_module(x.data)

        # 保证权重在同设备
        if cuda.get_array_module(self.W.data) is not xp:
            self.W.data = cuda.as_cupy(self.W.data) if xp is not np else cuda.as_numpy(self.W.data)
            if self.b is not None:
                self.b.data = cuda.as_cupy(self.b.data) if xp is not np else cuda.as_numpy(self.b.data)

        orig_shape = x.shape
        D = orig_shape[-1]
        if D != self.in_size:
            raise ValueError(f"LinearND: last dim {D} != in_size {self.in_size}")

        x2 = F.reshape(x, (-1, D))               # (N*T, D)
        y2 = F.linear_simple(x2, self.W, self.b) # (N*T, O)
        y  = F.reshape(y2, orig_shape[:-1] + (self.out_size,))
        return y
