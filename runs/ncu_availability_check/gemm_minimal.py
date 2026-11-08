"""Future one-kernel NCU probe only; not executed during the busy-GPU check."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch  # Match the already verified local CUDA library loading order.
import cupy as cp
from hytorch.backends.cuda import runtime as rt
n=1024
a=cp.ones((n,n),dtype=cp.float32)
b=cp.ones_like(a)
out=cp.empty_like(a)
rt.gemm(a,b,variant='shared',tile=16,out=out)
cp.cuda.get_current_stream().synchronize()
assert bool(cp.all(out==cp.float32(n)))
print('FP32 1024x1024 GEMM correct; source dispatch selects gemm_register<16>, grid=(16,16), block=(16,16). Confirm kernel name in NCU report.')
