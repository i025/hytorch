from common import *
import argparse
from benchmarks.bench_cuda_training import initialize,parameters
from donkeycar_hy.model import DonkeyNet
from hytorch.optimizers import SGD
p=argparse.ArgumentParser();p.add_argument('--profile',action='store_true');a=p.parse_args();rows=[]
stream=cp.cuda.Stream(non_blocking=True);stream.use();sync=stream.synchronize
if a.profile:cp.cuda.runtime.profilerStart()
for i,spec in enumerate(specs+holdout):
 for layout in ['contiguous','negative','slice']:
  x,g=arrays(spec,2026,layout);k,s,pad=spec[3:];fns={'A':lambda:original(x,g,k,s,pad),'B':lambda:rt.packed_dweight(x,g,k,s,pad)}
  for f in fns.values():
   for _ in range(10):f()
  for rep in range(3 if a.profile else 15):
   for mode in (['A','B'] if rep%2==0 else ['B','A']):
    f=fns[mode]
    if a.profile:
     cp.cuda.nvtx.RangePush(f'dwbench:{spec[0]}:{layout}:{mode}:{rep}');f();cp.cuda.nvtx.RangePop();continue
    sync();t=time.perf_counter_ns();y=f();cpu=(time.perf_counter_ns()-t)/1000;sync()
    t=time.perf_counter_ns();y=f();sync();wall=(time.perf_counter_ns()-t)/1000
    start,end=cp.cuda.Event(),cp.cuda.Event();start.record();y=f();end.record();end.synchronize();envelope=cp.cuda.get_elapsed_time(start,end)*1000
    rows.append(dict(operator='dweight',case=spec[0],layout=layout,mode=mode,repeat=rep,cpu_submit_us=cpu,synchronized_us=wall,event_envelope_us=envelope,x_bytes=x.nbytes,gy_bytes=g.nbytes,col_payload_bytes=int(x.shape[0]*g.shape[2]*g.shape[3]*x.shape[1]*k[0]*k[1]*4)))
model=DonkeyNet(dropout_rate=0.);initialize(model,(3,66,200),42);model.to_gpu();ps=list(parameters(model).values());rng=np.random.default_rng(3)
from hytorch import Variable
for p in ps:p.grad=Variable(cp.asarray(rng.normal(0,.1,p.shape).astype(np.float32)))
opt=SGD(.001).setup(model)
fns={'legacy':opt.update,'prototype_rejected':lambda:rt._sgd_multi_candidate(ps,.001)}
for f in fns.values():
 for _ in range(10):f()
for rep in range(3 if a.profile else 31):
 for mode in (list(fns) if rep%2==0 else list(fns)[::-1]):
  f=fns[mode]
  if a.profile:
   cp.cuda.nvtx.RangePush(f'sgdbench:{mode}:{rep}');f();cp.cuda.nvtx.RangePop();continue
  sync();t=time.perf_counter_ns();f();cpu=(time.perf_counter_ns()-t)/1000;sync();t=time.perf_counter_ns();f();sync();wall=(time.perf_counter_ns()-t)/1000
  rows.append(dict(operator='sgd',case='DonkeyNet',layout='contiguous',mode=mode,repeat=rep,cpu_submit_us=cpu,synchronized_us=wall,event_envelope_us='',tensor_count=len(ps),status='diagnostic only; prototype rejected by subnormal gate' if mode!='legacy' else 'baseline'))
if a.profile:sync();cp.cuda.runtime.profilerStop()
else:write(O/'OPERATORS.csv',rows)
