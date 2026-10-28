"""Actual Torch/CuPy CUDA smoke on the one visible device."""
import json
import os
import numpy as np
import torch
import cupy as cp
assert torch.cuda.is_available(), 'PyTorch CUDA is unavailable'
assert torch.cuda.device_count()==cp.cuda.runtime.getDeviceCount()==1
x=np.arange(16,dtype=np.float32).reshape(4,4)
tx=torch.from_numpy(x).cuda(0)
cx=cp.asarray(x)
actual_t=(tx@tx).cpu().numpy()
actual_c=cp.asnumpy(cx@cx)
np.testing.assert_array_equal(actual_t,x@x)
np.testing.assert_array_equal(actual_c,x@x)
print(json.dumps(dict(status='PASS',cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],logical_device=0,
    torch=torch.__version__,torch_cuda=torch.version.cuda,torch_cuda_available=torch.cuda.is_available(),
    gpu=torch.cuda.get_device_name(0),cupy=cp.__version__,cupy_runtime=cp.cuda.runtime.runtimeGetVersion(),
    torch_result=actual_t.tolist(),cupy_result=actual_c.tolist()),indent=2))
