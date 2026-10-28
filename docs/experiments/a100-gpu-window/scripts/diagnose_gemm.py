"""Numerical diagnosis only; no performance claims and no tolerance changes."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import cupy as cp
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
from hytorch.backends.cuda.runtime import gemm
from benchmarks.bench_cuda_gemm import errors
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
rows=[];samples=[]
for n in (1024,2048,4096):
    rng=np.random.default_rng(42+n);x=rng.standard_normal((n,n),dtype=np.float32);y=rng.standard_normal((n,n),dtype=np.float32)
    reference=x@y;dx,dy=cp.asarray(x),cp.asarray(y)
    cupy=cp.asnumpy(cp.matmul(dx,dy));cp.cuda.get_current_stream().synchronize()
    for method,tile in [('cupy',None),('basic',16),('blocked',16),('blocked',32),('shared',16),('shared',32)]:
        actual=cupy.copy() if method=='cupy' else cp.asnumpy(gemm(dx,dy,method,tile))
        cp.cuda.get_current_stream().synchronize()
        error=np.abs(actual.astype(np.float64)-reference.astype(np.float64));tol=1e-4+1e-4*np.abs(reference.astype(np.float64));failed=error>tol
        absolute,relative=errors(actual,reference)
        rows.append(dict(n=n,method=method,tile=tile,max_abs_error=absolute,relative_l2_error=relative,
            failed_elements=int(failed.sum()),max_tolerance_ratio=float(np.max(error/tol)),max_abs_vs_cupy=float(np.max(np.abs(actual-cupy))),correct=not bool(failed.any())))
        indices=np.argpartition((error/tol).ravel(),-3)[-3:]
        for index in indices:
            i,j=np.unravel_index(index,error.shape);ref64=float(np.dot(x[i,:].astype(np.float64),y[:,j].astype(np.float64)))
            samples.append(dict(n=n,method=method,tile=tile,row=int(i),col=int(j),numpy_fp32=float(reference[i,j]),gpu_fp32=float(actual[i,j]),
                dot_fp64=ref64,gpu_abs_vs_fp64=abs(float(actual[i,j])-ref64),numpy_abs_vs_fp64=abs(float(reference[i,j])-ref64),allowed_error=float(tol[i,j])))
        print(json.dumps(rows[-1]),flush=True)
    (a.output/'errors.json').write_text(json.dumps(rows,indent=2));(a.output/'fp64-dot-samples.json').write_text(json.dumps(samples,indent=2))
