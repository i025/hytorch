from pathlib import Path
import sys,csv,json,time,contextlib,hashlib,os
R=Path(__file__).resolve().parents[4];sys.path.insert(0,os.environ.get("HYTORCH_EXPERIMENT_SOURCE",str(R)))
import torch
import cupy as cp
import numpy as np
import hytorch
import hytorch.functions as F
from hytorch import functions_conv as FC
from hytorch.backends.cuda import runtime as rt
O=R/'runs/a100_20261107/conv_pool_framework_optimization'
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
sync=cp.cuda.get_current_stream().synchronize

def write(path,rows):
 path=Path(path)
 if not rows:return
 with path.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)

@contextlib.contextmanager
def config(mode='B0'):
 opts=dict(matmul='shared',conv='im2col',pool='native',col2im='native',dweight='packed',metadata_cache=True)
 ctx=hytorch.core.using_config('compact_function',mode=='O1') if hasattr(hytorch.core.Config,'compact_function') else contextlib.nullcontext()
 with rt.backend(**opts),ctx:yield

def host(x):
 if isinstance(x,hytorch.Variable):x=x.data
 if isinstance(x,torch.Tensor):return x.detach().cpu().numpy()
 return cp.asnumpy(x)

def error(a,b,atol=2e-5,rtol=2e-4):
 aa,bb=host(a),host(b);bitwise=aa.dtype==bb.dtype and aa.shape==bb.shape and aa.tobytes()==bb.tobytes()
 a,b=aa.astype(np.float64),bb.astype(np.float64);d=np.abs(a-b)
 return dict(bitwise_equal=bitwise,max_abs=float(d.max(initial=0)),rel_l2=float(np.linalg.norm(d)/max(np.linalg.norm(b),1e-30)),failed=int(np.count_nonzero(d>atol+rtol*np.abs(b))))

@contextlib.contextmanager
def phase(name,enabled):
 if enabled:cp.cuda.nvtx.RangePush(name)
 try:yield
 finally:
  if enabled:cp.cuda.nvtx.RangePop()
