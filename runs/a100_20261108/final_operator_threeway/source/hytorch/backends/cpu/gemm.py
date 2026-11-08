import ctypes
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np


_THIS_DIR = Path(__file__).resolve().parent
_C_SRC = _THIS_DIR / "gemm_kernels.c"
_C_LIB_NAME = "hytorch_gemm_kernel_v2.dll"
_C_HANDLE = None
_C_LOAD_ERROR = None


def _candidate_build_roots():
    roots = []
    env_root = os.getenv("HYTORCH_GEMM_BUILD_DIR")
    if env_root:
        roots.append(Path(env_root))
    roots.append(Path(tempfile.gettempdir()) / "hytorch_gemm_build")
    roots.append(Path.cwd() / ".hytorch_build")
    return roots


def _try_compile_with_cl(src, out_lib):
    cmd = ["cl", "/nologo", "/O2", "/LD", str(src), "/Fe:" + str(out_lib)]
    try:
        return subprocess.run(cmd, check=False, capture_output=True, text=True)
    except FileNotFoundError:
        return SimpleNamespace(returncode=127, stdout="", stderr="cl not found")


def _try_compile_with_gcc(src, out_lib):
    candidates = []
    gcc_in_path = shutil.which("gcc")
    if gcc_in_path:
        candidates.append(gcc_in_path)

    for p in [
        r"C:\mingw64\bin\gcc.exe",
        r"C:\msys64\mingw64\bin\gcc.exe",
        r"C:\msys64\ucrt64\bin\gcc.exe",
    ]:
        if os.path.exists(p):
            candidates.append(p)

    if not candidates:
        return SimpleNamespace(returncode=127, stdout="", stderr="gcc not found")

    last_result = None
    for gcc_bin in candidates:
        cmd = [gcc_bin, "-O3", "-shared", "-o", str(out_lib), str(src)]
        try:
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)
            last_result = result
            if result.returncode == 0:
                return result
        except FileNotFoundError:
            last_result = SimpleNamespace(returncode=127, stdout="", stderr="gcc not found: {}".format(gcc_bin))

    return last_result or SimpleNamespace(returncode=127, stdout="", stderr="gcc not found")


def _load_c_kernel():
    global _C_HANDLE, _C_LOAD_ERROR
    if _C_HANDLE is not None:
        return _C_HANDLE

    if not _C_SRC.exists():
        _C_LOAD_ERROR = "Missing C source: {}".format(_C_SRC)
        return None

    c_lib = None
    compile_result = None

    for root in _candidate_build_roots():
        try:
            root.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue

        candidate_lib = root / _C_LIB_NAME
        if candidate_lib.exists():
            c_lib = candidate_lib
            break

        tmp_lib = root / ("tmp_" + _C_LIB_NAME)
        result = _try_compile_with_cl(_C_SRC, tmp_lib)
        if result.returncode != 0:
            result = _try_compile_with_gcc(_C_SRC, tmp_lib)
        compile_result = result
        if result.returncode != 0:
            continue

        try:
            os.replace(str(tmp_lib), str(candidate_lib))
            c_lib = candidate_lib
            break
        except OSError:
            continue

    if c_lib is None:
        _C_LOAD_ERROR = "Failed to compile C GEMM kernel (stderr: {})".format(getattr(compile_result, "stderr", ""))
        return None

    try:
        lib = ctypes.CDLL(str(c_lib))
    except OSError as e:
        _C_LOAD_ERROR = "Failed to load C GEMM shared library: {}".format(e)
        return None

    lib.gemm4x4_f32.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ]
    lib.gemm4x4_f32.restype = ctypes.c_int

    lib.gemm4x4_f64.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
    ]
    lib.gemm4x4_f64.restype = ctypes.c_int

    lib.gemm_naive_f32.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ]
    lib.gemm_naive_f32.restype = ctypes.c_int

    lib.gemm_naive_f64.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
    ]
    lib.gemm_naive_f64.restype = ctypes.c_int

    lib.gemm_blocked_packed_f32.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ]
    lib.gemm_blocked_packed_f32.restype = ctypes.c_int

    lib.gemm_blocked_packed_f64.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
    ]
    lib.gemm_blocked_packed_f64.restype = ctypes.c_int

    _C_HANDLE = lib
    return _C_HANDLE


def c_backend_available():
    return _load_c_kernel() is not None


def _gemm_call_c(a, b, f32_func_name, f64_func_name):
    lib = _load_c_kernel()
    if lib is None:
        raise RuntimeError("C GEMM kernel unavailable: {}".format(_C_LOAD_ERROR))

    dtype = np.result_type(a.dtype, b.dtype)
    m, k = a.shape
    n = b.shape[1]

    if dtype == np.float32:
        a_cont = np.ascontiguousarray(a, dtype=np.float32)
        b_cont = np.ascontiguousarray(b, dtype=np.float32)
        c = np.empty((m, n), dtype=np.float32)
        fn = getattr(lib, f32_func_name)
        ret = fn(
            int(m), int(n), int(k),
            a_cont.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            b_cont.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
    elif dtype == np.float64:
        a_cont = np.ascontiguousarray(a, dtype=np.float64)
        b_cont = np.ascontiguousarray(b, dtype=np.float64)
        c = np.empty((m, n), dtype=np.float64)
        fn = getattr(lib, f64_func_name)
        ret = fn(
            int(m), int(n), int(k),
            a_cont.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            b_cont.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
    else:
        c = np.matmul(a, b)
        ret = 0

    if ret != 0:
        raise RuntimeError("C GEMM kernel returned non-zero status: {}".format(ret))
    return c


def cpu_gemm(a, b, bias=None, use_blocked_packed=True, kernel=None):
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("gemm expects 2D matrices, got a.ndim={}, b.ndim={}".format(a.ndim, b.ndim))
    if a.shape[1] != b.shape[0]:
        raise ValueError("gemm shape mismatch: a.shape={}, b.shape={}".format(a.shape, b.shape))

    selected_kernel = kernel if kernel is not None else os.getenv(
        "HYTORCH_CPU_GEMM_KERNEL",
        "blocked_packed_c" if use_blocked_packed else "4x4_c",
    )

    if selected_kernel not in ("naive_c", "4x4_c", "blocked_packed_c"):
        raise ValueError(
            "Unsupported GEMM kernel: {}. Use 'naive_c', '4x4_c' or 'blocked_packed_c'.".format(selected_kernel)
        )

    try:
        if selected_kernel == "naive_c":
            c = _gemm_call_c(a, b, "gemm_naive_f32", "gemm_naive_f64")
        elif selected_kernel == "blocked_packed_c":
            c = _gemm_call_c(a, b, "gemm_blocked_packed_f32", "gemm_blocked_packed_f64")
        else:
            c = _gemm_call_c(a, b, "gemm4x4_f32", "gemm4x4_f64")
    except Exception:
        # Keep framework usable even when local C toolchain is unavailable.
        c = np.matmul(a, b)

    if bias is not None:
        c = c + bias
    return c


def gemm(a, b, bias=None, kernel=None):
    return cpu_gemm(a, b, bias=bias, use_blocked_packed=True, kernel=kernel)
