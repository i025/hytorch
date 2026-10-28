"""Separate matched FP32 CuPy/PyTorch supplement; original GEMM baseline unchanged."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
import torch
import cupy as cp
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
from benchmarks.bench_cuda_gemm import errors,gpu_time
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
args.output.mkdir(parents=True,exist_ok=False)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
rows=[]
for n in (1024,2048,4096):
    rng=np.random.default_rng(42+n)
    a=rng.standard_normal((n,n),dtype=np.float32);b=rng.standard_normal((n,n),dtype=np.float32)
    reference=a@b
    ca,cb=cp.asarray(a),cp.asarray(b);cy=cp.empty((n,n),dtype=cp.float32)
    ta,tb=torch.from_numpy(a).cuda(),torch.from_numpy(b).cuda();ty=torch.empty((n,n),device='cuda',dtype=torch.float32)
    _,cupy_times=gpu_time(lambda:cp.matmul(ca,cb,out=cy),cp,10,5)
    for _ in range(5): torch.matmul(ta,tb,out=ty)
    torch.cuda.synchronize();torch_times=[]
    for _ in range(10):
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        start.record();torch.matmul(ta,tb,out=ty);end.record();end.synchronize();torch_times.append(start.elapsed_time(end))
    for method,actual,times in [('cupy',cp.asnumpy(cy),cupy_times),('torch',ty.cpu().numpy(),torch_times)]:
        absolute,relative=errors(actual,reference)
        rows.append(dict(size=n,implementation=method,mean_ms=float(np.mean(times)),std_ms=float(np.std(times,ddof=1)),
            median_ms=float(np.median(times)),p95_ms=float(np.percentile(times,95)),repeats_ms=json.dumps(times),
            gflops=2*n**3/(np.mean(times)*1e6),speedup_cupy=float(np.mean(cupy_times)/np.mean(times)),
            max_abs_error=absolute,relative_l2_error=relative,correct=bool(np.allclose(actual,reference,atol=1e-4,rtol=1e-4))))
    with (args.output/'gemm.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    print('Finished',n,flush=True)
    del ca,cb,cy,ta,tb,ty
metadata=dict(torch=torch.__version__,torch_cuda=torch.version.cuda,cupy=cp.__version__,torch_tf32=torch.backends.cuda.matmul.allow_tf32,
    cupy_fp32_compute_type=int(cp._core.get_compute_type(cp.float32)),gpu=torch.cuda.get_device_name(),dtype='float32',seed='42+size',warmup=5,repeats=10,
    timing='CUDA events, preallocated outputs, excludes H2D/D2H; separate session from seven-configuration baseline',
    atol=1e-4,rtol=1e-4,gpu_status='EXECUTED',all_correct=all(r['correct'] for r in rows))
(args.output/'summary.json').write_text(json.dumps(metadata,indent=2))
if not metadata['all_correct']: raise SystemExit('Numerical acceptance failed; original measurements retained')
