"""One-shot GPU numerical comparison, no timing benchmark or kernel changes."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import numpy as np
import cupy as cp
import torch
from accuracy import metrics,summarize,ulp_distance
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
from hytorch.backends.cuda.runtime import gemm,load_library,library_path
p=argparse.ArgumentParser();p.add_argument('--positions',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
assert cp.cuda.runtime.getDeviceCount()==torch.cuda.device_count()==1
cp.cuda.Device(0).use();load_library()
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
positions=json.loads(args.positions.read_text());sample_rows=[];comparisons=[]
meta=dict(status='RUNNING',gpu_status='EXECUTED',logical_device=0,uuid=os.environ['CUDA_VISIBLE_DEVICES'],
    gpu=torch.cuda.get_device_name(0),cupy=cp.__version__,torch=torch.__version__,torch_cuda=torch.version.cuda,
    cupy_runtime=cp.cuda.runtime.runtimeGetVersion(),cupy_fp32_compute_type=int(cp._core.get_compute_type(cp.float32)),
    cublas_math_mode_before=cp.cuda.cublas.getMathMode(cp.cuda.device.get_cublas_handle()),
    torch_matmul_tf32=torch.backends.cuda.matmul.allow_tf32,torch_cudnn_tf32=torch.backends.cudnn.allow_tf32,
    CUPY_TF32=os.environ.get('CUPY_TF32'),NVIDIA_TF32_OVERRIDE=os.environ.get('NVIDIA_TF32_OVERRIDE'),
    library=str(library_path()),full_matrix_fp64=False,full_matrix_comparisons='GPU methods versus each other and original NumPy FP32 only')
for n in (1024,2048,4096):
    rng=np.random.default_rng(42+n);a=rng.standard_normal((n,n),dtype=np.float32);b=rng.standard_normal((n,n),dtype=np.float32)
    reference=a@b;ca,cb=cp.asarray(a),cp.asarray(b)
    host={'cupy':cp.asnumpy(cp.matmul(ca,cb))}
    ta,tb=torch.from_numpy(a).cuda(0),torch.from_numpy(b).cuda(0)
    host['torch']=torch.matmul(ta,tb).cpu().numpy();del ta,tb
    for method,tile in [('basic',16),('blocked',16),('blocked',32),('shared',16),('shared',32)]:
        host[method if method=='basic' else f'{method}{tile}']=cp.asnumpy(gemm(ca,cb,method,tile))
    for method,actual in host.items():
        indices=np.argwhere(np.abs(actual.astype(np.float64)-reference.astype(np.float64))>1e-4+1e-4*np.abs(reference.astype(np.float64)))
        comparisons.append(dict(n=n,a=method,b='numpy_fp32',scope='FULL_MATRIX',elements=n*n,
            unequal_elements=int(np.count_nonzero(actual!=reference)),max_abs_difference=float(np.max(np.abs(actual.astype(np.float64)-reference.astype(np.float64)))),
            fail_original_tolerance=len(indices)))
        with (args.output/f'{method}-{n}-failure-coordinates.csv').open('x',newline='') as f:
            w=csv.writer(f,lineterminator='\n');w.writerow(['row','col','actual','numpy_fp32'])
            for i,j in indices:w.writerow([int(i),int(j),float(actual[i,j]),float(reference[i,j])])
        for record in [r for r in positions if r['n']==n]:
            i,j=record['row'],record['col'];assert reference[i,j]==record['numpy_fp32']
            sample_rows.append(dict(**record,strategy=method,**metrics(actual[i,j],record['reference_fp64'],reference[i,j])))
    names=list(host)
    for index,left in enumerate(names):
        for right in names[index+1:]:
            a0,b0=host[left],host[right]
            comparisons.append(dict(n=n,a=left,b=right,scope='FULL_MATRIX',elements=n*n,
                unequal_elements=int(np.count_nonzero(a0!=b0)),max_abs_difference=float(np.max(np.abs(a0.astype(np.float64)-b0.astype(np.float64)))),
                fail_original_tolerance=int(np.count_nonzero(np.abs(a0.astype(np.float64)-b0.astype(np.float64))>1e-4+1e-4*np.abs(b0.astype(np.float64))))))
    with (args.output/'full-matrix-comparisons.json').open('w') as f:json.dump(comparisons,f,indent=2)
    print(json.dumps(dict(n=n,comparisons=[r for r in comparisons if r['n']==n and r['b']=='numpy_fp32'])),flush=True)
    del host,reference,ca,cb,a,b
with (args.output/'gpu-samples.csv').open('x',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(sample_rows[0]),lineterminator='\n');w.writeheader();w.writerows(sample_rows)
meta.update(status='COMPLETED',cublas_math_mode_after=cp.cuda.cublas.getMathMode(cp.cuda.device.get_cublas_handle()),groups=summarize(sample_rows))
with (args.output/'gpu-summary.json').open('x') as f:json.dump(meta,f,indent=2)
print(json.dumps(dict(status='COMPLETED',full_comparison_rows=len(comparisons),sample_rows=len(sample_rows))),flush=True)
