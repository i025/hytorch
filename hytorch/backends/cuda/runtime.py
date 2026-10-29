"""ctypes CUDA ABI with explicit current-stream and device ownership."""
import ctypes as ct
import os
from pathlib import Path
from contextlib import contextmanager
from contextvars import ContextVar
from collections import Counter
import numpy as np

_config = ContextVar("hytorch_cuda_backend", default={
    "matmul": "cupy", "conv": "cupy", "pool": "cupy", "tile": 16,
    "conv_backward": "original"})
_counts = Counter()
_library = None
_dll_directories = []
_SGD_MAX_TENSORS = 64


def current_config():
    return dict(_config.get())


def dispatch_stats(reset=False):
    result = dict(_counts)
    if reset:
        _counts.clear()
    return result


@contextmanager
def backend(matmul="cupy", conv="cupy", pool="cupy", tile=16, *, conv_backward="original"):
    if matmul not in ("cupy", "basic", "blocked", "shared"):
        raise ValueError("matmul must be cupy/basic/blocked/shared")
    if conv not in ("cupy", "direct", "im2col") or pool not in ("cupy", "native"):
        raise ValueError("conv must be cupy/direct/im2col; pool must be cupy/native")
    if tile not in (16, 32):
        raise ValueError("tile must be 16 or 32")
    if conv_backward not in ("original", "fused"):
        raise ValueError("conv_backward must be original/fused")
    token = _config.set(dict(matmul=matmul, conv=conv, pool=pool, tile=tile,
                            conv_backward=conv_backward))
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


def _sgd_metadata(parameters, gradients):
    """Refresh host descriptors; never copy or make parameter data contiguous."""
    import cupy as cp
    if len(parameters) != len(gradients):
        raise ValueError("Fused SGD needs one gradient entry per parameter")
    active = [(p, g) for p, g in zip(parameters, gradients) if g is not None]
    count = len(active)
    if count > _SGD_MAX_TENSORS:
        raise ValueError(f"Fused SGD supports at most {_SGD_MAX_TENSORS} active tensors")
    parameter_ptrs = (ct.c_uint64 * count)()
    gradient_ptrs = (ct.c_uint64 * count)()
    sizes = (ct.c_int64 * count)()
    if not count:
        return parameter_ptrs, gradient_ptrs, sizes, count
    device = cp.cuda.runtime.getDevice()
    ranges = []
    for index, (parameter, gradient) in enumerate(active):
        for value in (parameter, gradient):
            if not isinstance(value, cp.ndarray) or value.dtype != cp.float32:
                raise TypeError("Fused SGD requires CuPy float32 parameters and gradients")
            if value.device.id != device:
                raise ValueError("Fused SGD arrays must belong to the current CUDA device")
            if not value.flags.c_contiguous:
                raise ValueError("Fused SGD requires C-contiguous arrays")
        if parameter.shape != gradient.shape:
            raise ValueError("Fused SGD parameter and gradient shapes must match")
        if not 0 < parameter.size <= 2**31 - 1:
            raise ValueError("Fused SGD arrays must have 1..INT_MAX elements")
        parameter_ptrs[index], gradient_ptrs[index] = parameter.data.ptr, gradient.data.ptr
        sizes[index] = parameter.size
        ranges.extend(((parameter.data.ptr, parameter.data.ptr + parameter.nbytes, True),
                       (gradient.data.ptr, gradient.data.ptr + gradient.nbytes, False)))
    # Concurrent tensor updates cannot overlap another parameter or any gradient.
    all_end = parameter_end = 0
    for start, end, is_parameter in sorted(ranges):
        if start < parameter_end or (is_parameter and start < all_end):
            raise ValueError("Fused SGD parameter storage must not overlap parameters or gradients")
        all_end = max(all_end, end)
        if is_parameter:
            parameter_end = end
    return parameter_ptrs, gradient_ptrs, sizes, count


def _submit_sgd(metadata, lr):
    """Submit host descriptors through the CUDA ABI without synchronizing."""
    lib = load_library()
    if not hasattr(lib, "hy_multi_tensor_sgd"):
        raise RuntimeError("Fused SGD CUDA symbol missing; rebuild with python scripts/build_cuda.py")
    fn = lib.hy_multi_tensor_sgd
    if fn.argtypes is None:
        fn.argtypes = [ct.POINTER(ct.c_uint64), ct.POINTER(ct.c_uint64),
                       ct.POINTER(ct.c_int64), ct.c_int, ct.c_float, ct.c_void_p]
        fn.restype = ct.c_int
    _call("hy_multi_tensor_sgd", *metadata, lr)


def multi_tensor_sgd(parameters, gradients, lr):
    """Update up to 64 CUDA FP32 tensor pairs in place on the current stream.

    Missing gradients are skipped. Arrays must be C-contiguous on the current
    device; parameter storage must not overlap other parameters or gradients.
    Pointers are refreshed on every call. No device descriptor allocation,
    explicit metadata copy, synchronization, or contiguous parameter packing is
    performed: CUDA transports the small descriptor POD as launch arguments.
    """
    metadata = _sgd_metadata(parameters, gradients)
    if not metadata[3]:
        return
    _submit_sgd(metadata, lr)
    _counts["sgd_fused"] += 1


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


def conv2d_dinput(gy, w, input_shape, stride=1, pad=0):
    """Gather dInput on the current stream; contiguous CUDA FP32 NCHW/OIHW only.

    Stride 1/2, padding 0/1, groups=1, dilation=1. Unsupported inputs raise;
    no implicit copies, workspace, synchronization or alternative algorithm.
    """
    import cupy as cp
    from hytorch.utils import pair
    for value in (gy, w):
        if not isinstance(value, cp.ndarray) or value.dtype != cp.float32:
            raise TypeError("Fused dInput requires CuPy float32 arrays")
        if value.ndim != 4 or not value.flags.c_contiguous:
            raise ValueError("Fused dInput requires contiguous NCHW/OIHW arrays")
        if value.device.id != cp.cuda.runtime.getDevice():
            raise ValueError("Fused dInput arrays must belong to the current CUDA device")
        _shape_ok(value.shape)
    n, c, h, wi = input_shape
    oc, wc, kh, kw = w.shape
    sh, sw = pair(stride); ph, pw = pair(pad)
    if sh not in (1, 2) or sw not in (1, 2) or ph not in (0, 1) or pw not in (0, 1):
        raise ValueError("Fused dInput supports stride 1/2 and padding 0/1")
    oh, ow = (h+2*ph-kh)//sh+1, (wi+2*pw-kw)//sw+1
    _shape_ok(input_shape)
    if c != wc or gy.shape != (n, oc, oh, ow):
        raise ValueError("Fused dInput input/weight/gradient shapes do not match")
    fn = load_library().hy_conv_dinput
    if fn.argtypes is None:
        fn.argtypes, fn.restype = [ct.c_void_p]*3 + [ct.c_int]*13 + [ct.c_void_p], ct.c_int
    gx = cp.empty(input_shape, dtype=cp.float32)
    _call("hy_conv_dinput", _ptr(gy), _ptr(w), _ptr(gx),
          n, c, h, wi, oc, kh, kw, sh, sw, ph, pw, oh, ow)
    _counts["conv_dinput_fused"] += 1
    return gx


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
