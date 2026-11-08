"""ctypes CUDA ABI with explicit current-stream and device ownership."""
import ctypes as ct
import os
from pathlib import Path
from contextlib import contextmanager
from contextvars import ContextVar
from collections import Counter
from functools import lru_cache
import operator
import struct
import numpy as np

_config = ContextVar("hytorch_cuda_backend", default={
    "matmul": "cupy", "conv": "cupy", "pool": "cupy", "tile": 16, "col2im": "cupy", "dweight": "cupy", "metadata_cache": False})
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
def backend(matmul="cupy", conv="cupy", pool="cupy", tile=16, col2im="cupy", dweight="cupy", metadata_cache=False):
    if matmul not in ("cupy", "basic", "blocked", "shared"):
        raise ValueError("matmul must be cupy/basic/blocked/shared")
    if conv not in ("cupy", "direct", "im2col") or pool not in ("cupy", "native"):
        raise ValueError("conv must be cupy/direct/im2col; pool must be cupy/native")
    if col2im not in ("cupy", "native"):
        raise ValueError("col2im must be cupy/native")
    if dweight not in ("cupy", "packed"):
        raise ValueError("dweight must be cupy/packed")
    if tile not in (16, 32):
        raise ValueError("tile must be 16 or 32")
    if type(metadata_cache) is not bool:
        raise ValueError("metadata_cache must be bool")
    token = _config.set(dict(matmul=matmul, conv=conv, pool=pool, tile=tile, col2im=col2im, dweight=dweight, metadata_cache=metadata_cache))
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
            "hy_col2im": [p]*2 + [ct.c_int64]*18 + [p],
            "hy_col2im_fast": [p, p, ct.c_char_p, p],
            "hy_sgd_multi": [p, ct.c_int64, ct.c_float, p],
        }
        for name, args in signatures.items():
            # Optional extension preserves loading older ABI-1 libraries.
            if name in ("hy_col2im", "hy_col2im_fast", "hy_sgd_multi") and not hasattr(lib, name):
                continue
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
    # Only immutable built-in integer tuples can share validation results.
    # Float/NumPy scalar equality must not let other metadata bypass its checks.
    if (_config.get()["metadata_cache"] and type(shape) is tuple and
            all(type(v) is int for v in shape)):
        return _shape_ok_cached(shape)
    return _shape_ok_uncached(shape)


def _shape_ok_uncached(shape):
    if any(v <= 0 for v in shape) or int(np.prod(shape, dtype=np.int64)) > 2**31-1:
        raise ValueError(f"Unsupported native output/workspace shape: {shape}")


# Values only; no array, pointer, stream, device or numerical workspace is retained.
# Failed validations are not cached. Dtype/device/alias checks remain per call.
_shape_ok_cached = lru_cache(maxsize=256)(_shape_ok_uncached)


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


@lru_cache(maxsize=128)
def _col2im_metadata(img_shape, col_shape, byte_strides, stride, pad):
    """Bounded, value-keyed immutable metadata only; no buffers, devices or streams."""
    n,c,h,w = img_shape
    cn,cc,kh,kw,oh,ow = col_shape
    sh,sw = stride
    ph,pw = pad
    if min(n,c,h,w,kh,kw,oh,ow,sh,sw) <= 0 or min(ph,pw) < 0:
        raise ValueError("Invalid col2im geometry")
    if (n,c) != (cn,cc) or oh != (h+2*ph-kh)//sh+1 or ow != (w+2*pw-kw)//sw+1:
        raise ValueError("col shape does not match image/kernel/stride/padding")
    if any(v % 4 for v in byte_strides):
        raise NotImplementedError("col strides must be integral element offsets")
    strides = tuple(v // 4 for v in byte_strides)
    limit = 2**63-1
    if (max(n,c,h,w,kh,kw,oh,ow,sh,sw,ph,pw) > 2**31-1 or
        n*c*h*w > limit or sum((d-1)*abs(s) for d,s in zip(col_shape,strides)) > limit):
        raise NotImplementedError("col2im indexing exceeds supported int64 geometry")
    values = (n,c,h,w,kh,kw,oh,ow,sh,sw,ph,pw,*strides)
    # bytes are immutable, CPU-only, and kept alive throughout the synchronous ABI call.
    return values, struct.pack("=18q", *values)


def col2im(col, img_shape, stride=1, pad=0, out=None):
    """FP32 six-axis col view -> contiguous NCHW, preserving real signed strides.

    Unsupported indexing metadata raises NotImplementedError for caller fallback.
    This deliberately does not use _arrays(), which would pack the input view.
    """
    import cupy as cp
    import operator
    from hytorch.utils import pair
    if not isinstance(col, cp.ndarray) or col.dtype != cp.float32:
        raise TypeError("Native col2im requires a CuPy float32 array")
    if col.ndim != 6 or len(img_shape) != 4:
        raise ValueError("col2im expects (N,C,KH,KW,OH,OW) and NCHW output")
    # Normalize every call: cache equality must never admit float geometry keys.
    img_shape = tuple(operator.index(v) for v in img_shape)
    stride = tuple(operator.index(v) for v in pair(stride))
    pad = tuple(operator.index(v) for v in pair(pad))
    if col.device.id != cp.cuda.runtime.getDevice():
        raise ValueError("col must belong to the current CUDA device")
    metadata, packed = _col2im_metadata(img_shape, col.shape, col.strides, stride, pad)
    n,c,h,w = img_shape
    if not hasattr(load_library(), "hy_col2im"):
        raise NotImplementedError("CUDA library has no col2im extension; rebuild to enable it")
    if out is None:
        out = cp.empty((n,c,h,w), dtype=cp.float32)
    elif (not isinstance(out, cp.ndarray) or out.dtype != cp.float32 or
          out.shape != (n,c,h,w) or not out.flags.c_contiguous or out.device.id != col.device.id):
        raise ValueError("out must be contiguous FP32 NCHW on the input device")
    if cp.may_share_memory(col, out):
        raise ValueError("out must not overlap col")
    lib = load_library()
    if hasattr(lib, "hy_col2im_fast"):
        _call("hy_col2im_fast", _ptr(col), _ptr(out), packed)
    else:
        _call("hy_col2im", _ptr(col), _ptr(out), *metadata)
    _counts["col2im_native"] += 1
    return out


def packed_dweight(x, gy, kernel, stride=1, pad=0):
    """Fresh backward-only workspace; no numerical cache or dtype conversion."""
    import cupy as cp
    if (not isinstance(x, cp.ndarray) or not isinstance(gy, cp.ndarray) or
            x.dtype != cp.float32 or gy.dtype != cp.float32):
        raise NotImplementedError("packed dWeight requires FP32 x and gy")
    device = cp.cuda.runtime.getDevice()
    if x.device.id != device or gy.device.id != device:
        raise NotImplementedError("packed dWeight requires the current device")
    if x.ndim != 4 or gy.ndim != 4:
        raise NotImplementedError("packed dWeight requires NCHW")
    try:
        g = _geometry(x, kernel, stride, pad)
        if any(not isinstance(v, (int, np.integer)) or v > 2**31-1 for v in g):
            raise ValueError("packed dWeight geometry exceeds the int32 CUDA ABI")
        n,c,h,w,kh,kw,sh,sw,ph,pw,oh,ow = g
        _shape_ok(x.shape)
        _shape_ok((n*oh*ow,c*kh*kw))
        if gy.shape[0] != n or gy.shape[2:] != (oh,ow):
            raise ValueError("gy geometry mismatch")
    except ValueError as exc:
        raise NotImplementedError(str(exc)) from exc
    x = cp.ascontiguousarray(x)
    col = cp.empty((n*oh*ow,c*kh*kw), dtype=cp.float32)
    _call("hy_im2col", _ptr(x), _ptr(col), *g)
    gm = gy.transpose(1,0,2,3).reshape(gy.shape[1],-1)
    result = (gm @ col).reshape(gy.shape[1],c,kh,kw)
    _counts["dweight_packed"] += 1
    return result


class _SgdTensor(ct.Structure):
    _fields_ = [("p",ct.c_uint64),("g",ct.c_uint64),("n",ct.c_uint64)]


def _sgd_multi_candidate(params, lr):
    """Return False before any update if the whole sequence needs legacy semantics."""
    import cupy as cp
    # Python float is weak FP32 for this verified range; explicit FP64 stays legacy.
    if type(lr) not in (float, np.float32) or not np.isfinite(lr) or abs(lr)>np.finfo(np.float32).max:
        return False
    device = cp.cuda.runtime.getDevice()
    arrays = []
    intervals = []
    for param in params:
        p,g = param.data,param.grad.data
        if (not isinstance(p,cp.ndarray) or not isinstance(g,cp.ndarray) or
            p.dtype!=cp.float32 or g.dtype!=cp.float32 or p.shape!=g.shape or
            not p.flags.c_contiguous or not g.flags.c_contiguous or
            p.device.id!=device or g.device.id!=device or p.size>2**63-1):
            return False
        if not p.size:
            continue
        pp,gp,n = p.data.ptr,g.data.ptr,p.size
        intervals.extend(((pp,pp+n*4,True),(gp,gp+n*4,False)))
        arrays.append((pp,gp,n))
    # All write/write and write/read overlaps force ordered fallback, even across groups.
    write_end = read_end = -1
    for start,end,write in sorted(intervals):
        if start < write_end or (write and start < read_end):
            return False
        if write:write_end=max(write_end,end)
        else:read_end=max(read_end,end)
    if not arrays:
        return True
    lib = load_library()
    if not hasattr(lib,"hy_sgd_multi"):
        return False
    descriptors = (_SgdTensor*len(arrays))(*(_SgdTensor(*v) for v in arrays))
    _call("hy_sgd_multi", ct.cast(descriptors,ct.c_void_p), len(arrays), float(lr))
    _counts["sgd_multi"] += 1
    return True


def sgd_multi(params, lr):
    # Rejected by exact subnormal acceptance: CuPy 14.2 uses FTZ; this kernel must not.
    # Preserve ordered legacy semantics until a compatible policy is explicitly chosen.
    return False
