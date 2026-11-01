"""Sample actual CUDA bodies compiled as serial host C++; not GPU execution."""
import argparse
import csv
import ctypes as ct
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from cases import cases
from hytorch.utils import pair


def host_source(source,kernel):
    # Use the actual indexing/accumulation body. Only CUDA launch/built-ins are
    # replaced; fmaf and float additions retain FP32 rounding on this x86 host.
    body=source.split('API int ')[0].replace('#include <cuda_runtime.h>','#include <cmath>')
    body=body.replace('__global__ void','extern "C" void').replace('__fmaf_rn','std::fma').replace('__fadd_rn','host_add')
    prefix='struct Dim { long long x; };\nstatic Dim blockIdx,blockDim,threadIdx;\ninline float host_add(float a,float b) { return a+b; }\n'
    suffix='''
extern "C" void samples(const float* gy,const float* weight,float* gx,
 int N,int C,int H,int W,int OC,int KH,int KW,int SH,int SW,int PH,int PW,int OH,int OW,
 const int64_t* positions,int count) {
 blockDim.x=256;
 for(int p=0;p<count;++p) {
  blockIdx.x=positions[p]/256;threadIdx.x=positions[p]%256;
  KERNEL(gy,weight,gx,N,C,H,W,OC,KH,KW,SH,SW,PH,PW,OH,OW);
 }
}
'''.replace('KERNEL',kernel)
    return prefix+body+suffix


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    functions={};commands=[]
    for mode,file,kernel in [('fused','conv_dinput.cu','conv_dinput_gather'),('fused_opt','conv_dinput_opt.cu','conv_dinput_gather_opt')]:
        path=a.output/(mode+'.cpp');path.write_text(host_source((ROOT/'hytorch/backends/cuda'/file).read_text(),kernel))
        library=a.output/(mode+'.so')
        command=['g++','-O2','-std=c++14','-ffp-contract=off','-fPIC','-shared',str(path),'-o',str(library)]
        done=subprocess.run(command,text=True,capture_output=True);commands.append(dict(command=command,returncode=done.returncode,stdout=done.stdout,stderr=done.stderr));done.check_returncode()
        fn=ct.CDLL(str(library.resolve())).samples
        fn.argtypes=[ct.c_void_p]*3+[ct.c_int]*13+[ct.c_void_p,ct.c_int];fn.restype=None;functions[mode]=fn
    (a.output/'build.json').write_text(json.dumps(commands,indent=2)+'\n')
    rows=[]
    for case in cases('history')+cases('fresh'):
        shape,wshape=case['shape'],case['weight_shape'];n,c,h,w=shape;co,_,kh,kw=wshape
        sh,sw=pair(case['stride']);ph,pw=pair(case['pad']);oh,ow=(h+2*ph-kh)//sh+1,(w+2*pw-kw)//sw+1
        rng=np.random.default_rng(42+int(case['case'][1:]));weight=rng.normal(size=wshape).astype(np.float32)*.1
        gy=rng.normal(size=(n,co,oh,ow)).astype(np.float32)*.1
        size=int(np.prod(shape));positions=np.array(sorted(set([0,size-1,*rng.choice(size,min(size,128),replace=False)])),dtype=np.int64)
        outputs={}
        for mode,fn in functions.items():
            gx=np.empty(shape,dtype=np.float32)
            fn(gy.ctypes.data,weight.ctypes.data,gx.ctypes.data,n,c,h,w,co,kh,kw,sh,sw,ph,pw,oh,ow,positions.ctypes.data,len(positions))
            outputs[mode]=gx.ravel()[positions].copy()
        expected=torch.nn.grad.conv2d_input(shape,torch.from_numpy(weight),torch.from_numpy(gy),
            stride=(sh,sw),padding=(ph,pw)).numpy().ravel()[positions]
        outputs['torch_cpu']=expected
        np.savez(a.output/(case['case']+'.npz'),weight=weight,gy=gy,positions=positions,**outputs)
        for reference in ('fused','torch_cpu'):
            delta=outputs['fused_opt'].astype(np.float64)-outputs[reference]
            for i,pos in enumerate(positions):
                row=dict(case=case['case'],split=case['split'],position=int(pos),reference=reference,
                    actual=float(outputs['fused_opt'][i]),expected=float(outputs[reference][i]),abs_error=float(abs(delta[i])),
                    passed=bool(abs(delta[i])<=1e-5+1e-4*abs(outputs[reference][i])))
                rows.append(row)
        print(case['case']+' sampled CPU bodies complete',flush=True)
    with (a.output/'samples.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=dict(status='PASSED' if all(r['passed'] for r in rows) else 'FAILED',cases=124,comparisons=len(rows),
        max_abs=max(r['abs_error'] for r in rows),atol=1e-5,rtol=1e-4,
        scope='Sampled CPU compilation of actual CUDA bodies; NOT CUDA execution or full-tensor GPU validation')
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
    assert summary['status']=='PASSED'


if __name__=='__main__':main()
