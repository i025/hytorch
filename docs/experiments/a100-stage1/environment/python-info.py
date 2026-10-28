"""Read package and runtime versions with all CUDA devices hidden."""
import ctypes
from importlib.metadata import version
import json
import os
import sys

assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
runtime = ctypes.CDLL("/usr/local/cuda-12.2/lib64/libcudart.so")
runtime.cudaRuntimeGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
runtime.cudaRuntimeGetVersion.restype = ctypes.c_int
runtime_version = ctypes.c_int()
status = runtime.cudaRuntimeGetVersion(ctypes.byref(runtime_version))
if status:
    raise RuntimeError("cudaRuntimeGetVersion returned {}".format(status))
print(json.dumps(dict(
    python=sys.version, executable=sys.executable, prefix=sys.prefix,
    base_prefix=sys.base_prefix, cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
    cuda_path=os.environ.get("CUDA_PATH"), runtime_version=runtime_version.value,
    packages={name: version(name) for name in
              ("numpy", "cupy-cuda12x", "pytest", "Pillow", "matplotlib", "psutil")},
    selected_physical_gpu=None, selected_logical_gpu=None,
    gpu_status="NOT_EXECUTED_WAITING_FOR_IDLE_DEVICE"), indent=2))
import cupy
cupy.show_config()
