from pathlib import Path
import sys,time,csv,json
import numpy as np
import torch
import cupy as cp
O=Path(__file__).resolve().parents[1];sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
import hytorch as hy
import hytorch.functions as F
from hytorch.backends.cuda import runtime as rt

def write(name,rows):
 with (O/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
shapes=[(),(1,),(257,),(4,24,31,98),(4,36,14,47),(4,48,5,22),(4,64,3,20),(4,64,1,18),(4,100),(4,50),(1024,1024),(7,11)]
rows=[]
for si,shape in enumerate(shapes):
 rng=np.random.default_rng(42);x=cp.asarray(rng.normal(size=shape).astype('f'));g=cp.asarray(rng.normal(size=shape).astype('f'))
 for layout in (['contiguous','transpose','negative'] if len(shape)==2 else ['contiguous']):
  xx=x.T if layout=='transpose' else x[::-1] if layout=='negative' else x;gg=g.T if layout=='transpose' else g[::-1] if layout=='negative' else g
  fs={'original':lambda:gg*(xx>0),'fused':lambda:rt.relu_backward(xx,gg)}
  expected=cp.asnumpy(fs['original']());np.testing.assert_array_equal(cp.asnumpy(fs['fused']()),expected)
  for f in fs.values():
   for _ in range(20):f()
  cp.cuda.get_current_stream().synchronize()
  for rep in range(20):
   for name in (['original','fused'] if rep%2==0 else ['fused','original']):
    f=fs[name];a,b=cp.cuda.Event(),cp.cuda.Event();cp.cuda.get_current_stream().synchronize()
    a.record();t=time.perf_counter_ns()
    for _ in range(30):result=f()
    submit=(time.perf_counter_ns()-t)/30/1e3;b.record();b.synchronize();full=(time.perf_counter_ns()-t)/30/1e3
    rows.append(dict(shape=str(shape),layout=layout,mode=name,repeat=rep,calls=30,event_interval_us=cp.cuda.get_elapsed_time(a,b)*1000/30,cpu_submit_us=submit,synchronized_wall_us=full,bitwise_equal=True))
write('RELU_BENCHMARK.csv',rows)
print('RELU COMPLETE',len(rows))
