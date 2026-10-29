"""Sample the gather indexing and FP32 fmaf equation on CPU, not the CUDA kernel."""
import argparse
import csv
import ctypes as ct
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
from test_conv_backward_fusion_gpu import CASES
from hytorch.utils import pair
from hytorch.functions_conv import deconv2d

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
fma=ct.CDLL('libm.so.6').fmaf
fma.argtypes=[ct.c_float]*3;fma.restype=ct.c_float
torch.set_num_threads(4)
rows=[]
for case,(shape,wshape,stride,pad) in enumerate(CASES):
    n,c,h,w=shape;oc,_,kh,kw=wshape
    sh,sw=pair(stride);ph,pw=pair(pad)
    oh,ow=(h+2*ph-kh)//sh+1,(w+2*pw-kw)//sw+1
    rng=np.random.default_rng(42)
    weight=rng.normal(size=wshape).astype(np.float32)*.1
    gy=rng.normal(size=(n,oc,oh,ow)).astype(np.float32)*.1
    old=deconv2d(gy,weight,stride=stride,pad=pad,outsize=shape[2:]).data
    reference=torch.nn.grad.conv2d_input(shape,torch.from_numpy(weight),torch.from_numpy(gy),stride=stride,padding=pad).numpy()
    size=int(np.prod(shape))
    positions=sorted(set([0,size-1,*rng.choice(size,min(size,128),replace=False).tolist()]))
    for flat in positions:
        bn,ci,ih,iw=np.unravel_index(flat,shape)
        value=np.float32(0)
        for ky in range(kh):
            oy=ih+ph-ky
            if oy<0 or oy>=oh*sh or oy%sh:continue
            for kx in range(kw):
                ox=iw+pw-kx
                if ox<0 or ox>=ow*sw or ox%sw:continue
                partial=np.float32(0)
                for co in range(oc):
                    partial=fma(gy[bn,co,oy//sh,ox//sw],weight[co,ci,ky,kx],partial)
                value=np.float32(value+np.float32(partial))
        for name,tensor in [('original_numpy',old),('pytorch_cpu',reference)]:
            expected=float(tensor[bn,ci,ih,iw]);error=abs(float(value)-expected)
            rows.append(dict(case=case,shape=str(shape),weight_shape=str(wshape),index=flat,
                reference=name,actual=float(value),expected=expected,abs_error=error,
                passed=error<=1e-5+1e-4*abs(expected)))
    np.savez(a.output/f'case{case}.npz',weight=weight,gy=gy,original=old,torch=reference,positions=np.array(positions))
with (a.output/'samples.csv').open('x',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
summary=dict(status='PASSED' if all(r['passed'] for r in rows) else 'FAILED',
    scope='CPU equation sample only, not full matrix or CUDA correctness',comparisons=len(rows),
    max_abs=max(r['abs_error'] for r in rows),failures=sum(not r['passed'] for r in rows),atol=1e-5,rtol=1e-4)
groups=[]
for case in range(len(CASES)):
    for name in ('original_numpy','pytorch_cpu'):
        rr=[r for r in rows if r['case']==case and r['reference']==name]
        delta=np.array([r['actual']-r['expected'] for r in rr])
        expected=np.array([r['expected'] for r in rr])
        groups.append(dict(case=case,reference=name,samples=len(rr),max_abs=max(r['abs_error'] for r in rr),
            relative_l2=float(np.linalg.norm(delta)/np.linalg.norm(expected))))
summary['per_case']=groups
(a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2));assert summary['status']=='PASSED'
