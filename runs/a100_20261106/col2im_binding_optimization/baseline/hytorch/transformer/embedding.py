import numpy as np
from hytorch import Layer, cuda, as_variable
from hytorch.core import Parameter


class TokenEmbedding(Layer):
    def __init__(self, vocab_size, d_model, dtype=np.float32):
        super().__init__()
        self.W = Parameter(
            (np.random.randn(vocab_size, d_model).astype(dtype) * 0.02),
            name="W"
        )

    def forward(self, token_ids):
        """
        token_ids: np/cupy array 或 Variable（里面是 int 索引）
        return: (N, T, D)
        """
        # 先包装成 Variable 以统一接口
        token_ids = as_variable(token_ids)

        # ✅关键修复：索引必须是“纯 array”，不能是 Variable
        idx = token_ids.data  # np.ndarray 或 cupy.ndarray

        # ✅Embedding 索引必须是整型
        xp = cuda.get_array_module(idx)
        if getattr(idx, "dtype", None) is None:
            idx = xp.asarray(idx, dtype=xp.int64)
        else:
            if idx.dtype.kind not in ("i", "u", "b"):
                idx = idx.astype(xp.int64)
            else:
                idx = idx.astype(xp.int64)

        # ✅确保 idx 与 W 在同一设备（W 在 GPU 时 idx 也要是 cupy）
        W_xp = cuda.get_array_module(self.W.data)
        if W_xp is not xp:
            if W_xp is np:
                # idx 在 cupy -> 转 numpy
                idx = cuda.as_numpy(idx).astype(np.int64, copy=False)
            else:
                # idx 在 numpy -> 转 cupy
                idx = cuda.as_cupy(idx).astype(W_xp.int64, copy=False)

        return self.W[idx]   # (N,T,D)


def _build_sinusoidal_pe(max_len, d_model, dtype=np.float32):
    pe = np.zeros((max_len, d_model), dtype=dtype)
    pos = np.arange(max_len, dtype=dtype)[:, None]
    div = np.exp(np.arange(0, d_model, 2, dtype=dtype) * (-(np.log(10000.0) / d_model)))
    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div)
    return pe


class SinusoidalPositionalEncoding(Layer):
    def __init__(self, d_model, max_len=2048, dtype=np.float32):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.dtype = dtype
        self.pe = _build_sinusoidal_pe(max_len, d_model, dtype=dtype)  # numpy 缓存

    def forward(self, x, start_pos=0):
        x = as_variable(x)
        T = x.shape[1]
        if start_pos + T > self.max_len:
            raise ValueError("PosEnc: sequence too long")

        xp = cuda.get_array_module(x.data)
        pe = self.pe[start_pos:start_pos + T]  # (T,D) numpy
        if xp is not np:
            pe = cuda.as_cupy(pe)
        pe = pe.reshape((1, T, self.d_model))
        return as_variable(pe)
