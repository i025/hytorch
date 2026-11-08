"""Matched observer-overhead experiment; diagnostic only, never production changes."""
import sys,time,json,csv,argparse
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path.cwd()))
import cupy as cp
import hytorch.functions as F
from hytorch.backends.cuda.runtime import backend
from hytorch.optimizers import SGD
from hytorch.dataloaders import DataLoader
from hytorch.profiling import StepProfiler
from benchmarks.bench_cuda_training import SyntheticDataset,initialize,parameters,snapshot,state_hash
from donkeycar_hy.model import DonkeyNet
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
cp.cuda.Device(0).use();sync=cp.cuda.get_current_stream().synchronize
model=DonkeyNet(dropout_rate=0.);initial=initialize(model,(3,66,200),42);model.to_gpu();opt=SGD(.001).setup(model)
ds=SyntheticDataset(64,(3,66,200),42)
def restore():
 for k,v in parameters(model).items():v.data=cp.asarray(initial[k].copy())
 model.cleargrads();sync()
def compute(x,t):
 loss=F.mean_square_error(model(x),t);model.cleargrads();loss.backward();opt.update();return loss
rows=[]
with backend(matmul='shared',conv='im2col',pool='native'):
 batch=next(iter(DataLoader(ds,4,shuffle=False)));x,t=map(cp.asarray,batch)
 for _ in range(10):compute(x,t)
 sync()
 for repeat in range(5):
  modes=['logged','no_json','boundary_only']
  if repeat%2:modes.reverse()
  for mode in modes:
   restore();profile=StepProfiler(out/f'{mode}_{repeat}.jsonl',sync,mode='throughput')
   start_event,end_event=cp.cuda.Event(),cp.cuda.Event();start_event.record();start=time.perf_counter()
   for epoch in range(3):
    for i,(x,t) in enumerate(DataLoader(ds,4,shuffle=False)):
     if mode=='logged':profile.begin()
     elif mode=='no_json':sync()
     # Original loader construction/next occurs outside this diagnostic's StepProfiler;
     # wall interval still includes the same data loading in all three modes.
     if mode=='logged':
      with profile.phase('transfer'):x,t=cp.asarray(x),cp.asarray(t)
      with profile.phase('forward'):loss=F.mean_square_error(model(x),t)
      with profile.phase('backward'):model.cleargrads();loss.backward()
      with profile.phase('update'):opt.update()
      profile.end(len(x),epoch,i)
     else:
      x,t=cp.asarray(x),cp.asarray(t);loss=compute(x,t)
      if mode=='no_json':sync()
     if mode!='boundary_only':value=float(loss.data.item())
   end_event.record();sync();wall=time.perf_counter()-start
   value=float(loss.data.item());final=snapshot(model)
   rows.append(dict(mode=mode,repeat=repeat,steps=48,wall_ms=wall*1000,event_span_ms=cp.cuda.get_elapsed_time(start_event,end_event),final_hash=state_hash(final),loss=value))
   print(rows[-1],flush=True)
with (out/'timings.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
assert len({r['final_hash'] for r in rows})==1,'Observer modes changed the training result'
(out/'method.json').write_text(json.dumps(dict(seed=42,samples=64,batch=4,epochs=3,warmup=10,repeats=5,backend='shared/im2col/native tile16',monitor=False,logged='StepProfiler throughput + per-step loss read',no_json='same step boundary synchronization and loss read, no JSON/phase context managers',boundary_only='no per-step log, scalar read or explicit sync; final boundary sync; pageable H2D still exists',event_span='GPU stream elapsed span including CPU submission gaps, NOT cumulative kernel time'),indent=2))
