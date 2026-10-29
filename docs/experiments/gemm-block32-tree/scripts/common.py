"""Shared setup for the standalone accuracy and timing experiments."""
import csv
import ctypes as ct
import json
import os
from pathlib import Path
import sys
import numpy as np
import cupy as cp
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'docs/experiments/gemm-fp32-accuracy/scripts'))
from hytorch.backends.cuda.runtime import load_library
from experiments.gemm_block32_tree.bindings import Block32Tree


def setup(library):
    assert cp.cuda.runtime.getDeviceCount() == torch.cuda.device_count() == 1
    cp.cuda.Device(0).use()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert cp.cuda.get_current_stream().ptr == torch.cuda.current_stream().cuda_stream == 0
    return Block32Tree(library)


def metadata():
    return dict(gpu=torch.cuda.get_device_name(0), uuid=os.environ['CUDA_VISIBLE_DEVICES'], logical_device=0,
        numpy=np.__version__, cupy=cp.__version__, torch=torch.__version__, torch_cuda=torch.version.cuda,
        cuda_runtime=cp.cuda.runtime.runtimeGetVersion(), cuda_driver=cp.cuda.runtime.driverGetVersion(),
        cupy_compute_type=int(cp._core.get_compute_type(cp.float32)),
        cublas_math_mode=cp.cuda.cublas.getMathMode(cp.cuda.device.get_cublas_handle()),
        CUPY_TF32=os.environ.get('CUPY_TF32'), NVIDIA_TF32_OVERRIDE=os.environ.get('NVIDIA_TF32_OVERRIDE'),
        torch_matmul_tf32=torch.backends.cuda.matmul.allow_tf32, torch_cudnn_tf32=torch.backends.cudnn.allow_tf32,
        stream=int(cp.cuda.get_current_stream().ptr))


def shared_call(a, b, out):
    function = load_library().hy_gemm
    m, k = a.shape
    arguments = (a.data.ptr, b.data.ptr, out.data.ptr, m, b.shape[1], k, 2, 32, cp.cuda.get_current_stream().ptr)

    def launch():
        status = function(*arguments)
        if status:
            raise RuntimeError(f'original hy_gemm CUDA status {status}')
        return out
    return launch


def inputs(n, seed_base):
    rng = np.random.default_rng(seed_base + n)
    return rng.standard_normal((n,n), dtype=np.float32), rng.standard_normal((n,n), dtype=np.float32)


def write_csv(path, rows):
    with Path(path).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
