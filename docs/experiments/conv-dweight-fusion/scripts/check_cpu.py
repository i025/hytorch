"""CPU FP32 split/reduction arithmetic model; explicitly not CUDA execution."""
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
import hytorch.functions_conv as FC

SOURCE=r'''
#include <cmath>
#include <cstdint>
extern "C" void samples(const float* x,const float* gy,float* dw,
 int N,int C,int H,int W,int OC,int KH,int KW,int SH,int SW,int PH,int PW,int OH,int OW,
 const int64_t* positions,int count) {
 for(int i=0;i<count;++i) {
  int64_t index=positions[i];int o=index/(C*KH*KW),c=index/(KH*KW)%C,kh=index/KW%KH,kw=index%KW;
  float groups[8]={};int R=N*OH*OW;
  for(int split=0;split<(R+127)/128;++split) {
   float value=0;
   for(int k=0;k<128;++k) {
    int r=split*128+k,n=r/(OH*OW),ih=r/OW%OH*SH+kh-PH,iw=r%OW*SW+kw-PW;
    float a=0,b=0;
    if(n<N) a=gy[((int64_t)n*OC+o)*OH*OW+r%(OH*OW)];
    if(n<N && ih>=0 && ih<H && iw>=0 && iw<W) b=x[((int64_t)n*C+c)*H*W+ih*W+iw];
    value=std::fma(a,b,value);
   }
   groups[split%8]=groups[split%8]+value;
  }
  for(int step=4;step;step/=2) for(int j=0;j<step;++j) groups[j]=groups[j]+groups[j+step];
  dw[i]=groups[0];
 }
}
'''


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    cpp,lib=a.output/'arithmetic.cpp',a.output/'arithmetic.so';cpp.write_text(SOURCE)
    command=['g++','-O2','-std=c++14','-ffp-contract=off','-fPIC','-shared',str(cpp),'-o',str(lib)]
    done=subprocess.run(command,text=True,capture_output=True)
    (a.output/'build.json').write_text(json.dumps(dict(command=command,returncode=done.returncode,stdout=done.stdout,stderr=done.stderr),indent=2)+'\n')
    done.check_returncode();fn=ct.CDLL(str(lib.resolve())).samples
    fn.argtypes=[ct.c_void_p]*3+[ct.c_int]*13+[ct.c_void_p,ct.c_int];fn.restype=None;rows=[]
    for case in cases():
        n,c,h,w=case['shape'];oc,_,kh,kw=case['weight_shape'];sh,sw=pair(case['stride']);ph,pw=pair(case['pad'])
        oh,ow=(h+2*ph-kh)//sh+1,(w+2*pw-kw)//sw+1
        rng=np.random.default_rng(42);x=rng.normal(size=case['shape']).astype(np.float32)*.1
        gy=rng.normal(size=(n,oc,oh,ow)).astype(np.float32)*.1
        size=oc*c*kh*kw;positions=np.array(sorted(set([0,size-1,*rng.choice(size,min(size,128),replace=False)])),np.int64)
        actual=np.empty(len(positions),np.float32)
        fn(x.ctypes.data,gy.ctypes.data,actual.ctypes.data,n,c,h,w,oc,kh,kw,sh,sw,ph,pw,oh,ow,positions.ctypes.data,len(positions))
        col=FC.im2col_array(x,(kh,kw),(sh,sw),(ph,pw),to_matrix=False)
        original=np.tensordot(gy,col,((0,2,3),(0,4,5))).ravel()[positions]
        reference=torch.nn.grad.conv2d_weight(torch.from_numpy(x),case['weight_shape'],torch.from_numpy(gy),stride=(sh,sw),padding=(ph,pw)).numpy().ravel()[positions]
        np.savez(a.output/(case['case']+'.npz'),x=x,gy=gy,positions=positions,actual=actual,original=original,torch_cpu=reference)
        for label,ref in [('numpy',original),('torch_cpu',reference)]:
            delta=actual.astype(np.float64)-ref
            rows.append(dict(case=case['case'],reference=label,positions=len(positions),max_abs=float(abs(delta).max()),
                relative_l2=float(np.linalg.norm(delta)/np.linalg.norm(ref.astype(np.float64))),
                mismatches=int((abs(delta)>1e-5+1e-4*abs(ref)).sum())))
        print(case['case']+' CPU samples complete',flush=True)
    with (a.output/'comparisons.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=dict(status='PASSED' if not sum(r['mismatches'] for r in rows) else 'FAILED',cases=len(cases()),
        comparisons=sum(r['positions'] for r in rows),max_abs=max(r['max_abs'] for r in rows),scope='CPU arithmetic model samples, not GPU execution or full-tensor GPU acceptance')
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2));assert summary['status']=='PASSED'


if __name__=='__main__':main()
