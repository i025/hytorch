"""ctypes CUDA ABI with explicit current-stream and device ownership."""
import ctypes as ct
import os
from pathlib import Path
from contextlib import contextmanager
from contextvars import ContextVar
from collections import Counter
import numpy as np

_config = ContextVar("hytorch_cuda_backend", default={
    "matmul": "cupy", "conv": "cupy", "pool": "cupy", "tile": 16})
_counts = Counter()
_library = None
_dll_directories = []


def current_config():
    return dict(_config.get())


def dispatch_stats(reset=False):
    result = dict(_counts)
    if reset:
        _counts.clear()
    return result


@contextmanager
def backend(matmul="cupy", conv="cupy", pool="cupy", tile=16):
    if matmul not in ("cupy", "basic", "blocked", "shared"):
        raise ValueError("matmul must be cupy/basic/blocked/shared")
    if conv not in ("cupy", "direct", "im2col") or pool not in ("cupy", "native"):
        raise ValueError("conv must be cupy/direct/im2col; pool must be cupy/native")
    if tile not in (16, 32):
        raise ValueError("tile must be 16 or 32")
    token = _config.set(dict(matmul=matmul, conv=conv, pool=pool, tile=tile))
    try:
        yield
    finally:
        _config.reset(token)


def library_path():
    name = "hytorch_cuda.dll" if os.name == "nt" else "libhytorch_cuda.so"
    return Path(os.environ.get("HYTORCH_CUDA_LIBRARY", Path(__file__).parent / "build" / name))


def load_library():
    global _library
    if _library is None:
        path = library_path()
        if not path.is_file():
            raise RuntimeError(f"CUDA library missing: {path}. Run python scripts/build_cuda.py on NVIDIA host.")
        if os.name == "nt" and os.environ.get("CUDA_PATH"):
            _dll_directories.append(os.add_dll_directory(str(Path(os.environ["CUDA_PATH"]) / "bin")))
        lib = ct.CDLL(str(path))
        p, i = ct.c_void_p, ct.c_int
        signatures = {
            "hy_cuda_abi_version": [], "hy_cuda_error": [i],
            "hy_gemm": [p]*3 + [i]*5 + [p],
            "hy_im2col": [p]*2 + [i]*12 + [p],
            "hy_conv": [p]*4 + [i]*13 + [p],
            "hy_conv_pack": [p]*3 + [i]*3 + [p],
            "hy_pool": [p]*3 + [i]*12 + [p],
        }
        for name, args in signatures.items():
            fn = getattr(lib, name)
            fn.argtypes, fn.restype = args, ct.c_char_p if name == "hy_cuda_error" else i
        if lib.hy_cuda_abi_version() != 1:
            raise RuntimeError("Incompatible CUDA library ABI; rebuild library")
        _library = lib
    return _library


def _call(name, *args):
    import cupy as cp
    lib = load_library()
    status = getattr(lib, name)(*args, cp.cuda.get_current_stream().ptr)
    if status:
        raise RuntimeError(f"{name}: {lib.hy_cuda_error(status).decode()}")


def _ptr(x):
    return None if x is None else x.data.ptr


def _arrays(*arrays):
    import cupy as cp
    valid = [x for x in arrays if x is not None]
    if any(not isinstance(x, cp.ndarray) or x.dtype != cp.float32 for x in valid):
        raise TypeError("Native CUDA kernels require CuPy float32 arrays")
    device = cp.cuda.runtime.getDevice()
    if any(x.device.id != device for x in valid):
        raise ValueError("All arrays must belong to the current CUDA device")
    if any(x.size == 0 or x.size > 2**31-1 for x in valid):
        raise ValueError("Native CUDA arrays must have 1..INT_MAX elements")
    return tuple(None if x is None else cp.ascontiguousarray(x) for x in arrays)


def _shape_ok(shape):
    if any(v <= 0 for v in shape) or int(np.prod(shape, dtype=np.int64)) > 2**31-1:
        raise ValueError(f"Unsupported native output/workspace shape: {shape}")


def gemm(a, b, variant="shared", tile=16, out=None):
    import cupy as cp
    if variant not in ("basic", "blocked", "shared") or tile not in (16, 32):
        raise ValueError("Invalid GEMM variant or tile")
    a, b = _arrays(a, b)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("GEMM expects (M,K) @ (K,N)")
    m, k = a.shape
    n = b.shape[1]
    _shape_ok((m, n))
    if (m+tile-1)//tile > 65535:
        raise ValueError("GEMM row grid exceeds CUDA grid.y limit")
    if out is None:
        out = cp.empty((m, n), dtype=cp.float32)
    else:
        _arrays(out)
        if out.shape != (m, n) or not out.flags.c_contiguous:
            raise ValueError("out must be contiguous with shape (M,N)")
        if cp.may_share_memory(out, a) or cp.may_share_memory(out, b):
            raise ValueError("out must not overlap inputs")
    _call("hy_gemm", _ptr(a), _ptr(b), _ptr(out), m, n, k,
          ("basic", "blocked", "shared").index(variant), tile)
    _counts["gemm_" + variant] += 1
    return out


def _geometry(x, kernel, stride, pad):
    from hytorch.utils import pair
    if x.ndim != 4:
        raise ValueError("Expected NCHW input")
    kh, kw = pair(kernel); sh, sw = pair(stride); ph, pw = pair(pad)
    if min(kh, kw, sh, sw) <= 0 or min(ph, pw) < 0:
        raise ValueError("Invalid kernel/stride/padding")
    n, c, h, w = x.shape
    oh, ow = (h+2*ph-kh)//sh+1, (w+2*pw-kw)//sw+1
    _shape_ok((n, c, oh, ow))
    return n, c, h, w, kh, kw, sh, sw, ph, pw, oh, ow


def conv2d(x, w, b=None, stride=1, pad=0, variant="im2col", tile=16):
    import cupy as cp
    if variant not in ("direct", "im2col") or tile not in (16, 32):
        raise ValueError("Invalid convolution variant or tile")
    x, w, b = _arrays(x, w, b)
    if w.ndim != 4 or x.ndim != 4 or x.shape[1] != w.shape[1]:
        raise ValueError("Expected NCHW input and OIHW weights with equal channels")
    g = _geometry(x, w.shape[2:], stride, pad)
    n, c, h, wi, kh, kw, sh, sw, ph, pw, oh, ow = g
    oc = w.shape[0]
    if b is not None and b.shape != (oc,):
        raise ValueError("Bias must have shape (out_channels,)")
    _shape_ok((n, oc, oh, ow))
    y = cp.empty((n, oc, oh, ow), dtype=cp.float32)
    if variant == "direct":
        _call("hy_conv", _ptr(x), _ptr(w), _ptr(b), _ptr(y),
              n, c, h, wi, oc, kh, kw, sh, sw, ph, pw, oh, ow)
    else:
        shape = (n*oh*ow, c*kh*kw)
        _shape_ok(shape)
        col = cp.empty(shape, dtype=cp.float32)
        _call("hy_im2col", _ptr(x), _ptr(col), *g)
        result = gemm(col, w.reshape(oc, -1).T, tile=tile)
        _call("hy_conv_pack", _ptr(result), _ptr(b), _ptr(y), n, oc, oh*ow)
    _counts["conv_" + variant] += 1
    return y


def max_pool2d(x, kernel, stride=1, pad=0):
    import cupy as cp
    x, = _arrays(x)
    g = _geometry(x, kernel, stride, pad)
    n, c, _, _, _, _, _, _, _, _, oh, ow = g
    y = cp.empty((n, c, oh, ow), dtype=cp.float32)
    indexes = cp.empty(y.shape, dtype=cp.int64)
    _call("hy_pool", _ptr(x), _ptr(y), _ptr(indexes), *g)
    _counts["pool_native"] += 1
    return y, indexes


def selected(op, x):
    """CPU / non-FP32 paths retain existing semantics, with visible counters."""
    mode = _config.get()[op]
    if mode == "cupy":
        return None
    from hytorch import cuda
    if cuda.get_array_module(x) is np or x.dtype != np.float32:
        _counts[op + "_fallback_cpu_or_dtype"] += 1
        return None
    return mode
