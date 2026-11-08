from pathlib import Path
import sys,csv,time,gc
import numpy as np
O=Path(__file__).resolve().parents[1];sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
import hytorch as hy
import hytorch.functions as F
rows=[]
def run(width,mode):
 x=hy.Variable(np.array([.1,-.2,.3],np.float32))
 branches=[F.tanh(x*np.float32(.001*i+1)) for i in range(width)]
 # balanced tree accumulates a wide frontier unlike a long sum chain.
 while len(branches)>1:branches=[branches[i]+branches[i+1] if i+1<len(branches) else branches[i] for i in range(0,len(branches),2)]
 y=F.sum(branches[0]**2)
 with hy.using_config('backward_schedule',mode):
  start=time.perf_counter_ns();y.backward();wall=(time.perf_counter_ns()-start)/1e6
 return wall,x.grad.data
for width in [1,8,64,256]:
 for m in ['list','heap']:
  for _ in range(5):run(width,m)
 for rep in range(20):
  rr={}
  for m in (['list','heap'] if rep%2==0 else ['heap','list']):
   elapsed,g=run(width,m);rr[m]=g;rows.append(dict(width=width,mode=m,repeat=rep,backward_cpu_ms=elapsed))
  np.testing.assert_array_equal(rr['list'],rr['heap'])
with (O/'HEAP_CPU.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('CPU heap complete',len(rows))
