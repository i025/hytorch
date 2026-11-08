"""Same-process alternating old/new dispatch, fixed FP32 training configuration."""
import os,sys,time,json,csv,argparse,contextlib
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import numpy as np
import cupy as cp
from hytorch.backends.cuda import runtime as rt
import hytorch.functions as F
from hytorch.optimizers import SGD
from hytorch.dataloaders import DataLoader
from benchmarks.bench_cuda_training import SyntheticDataset,initialize,parameters,snapshot,state_hash
from donkeycar_hy.model import DonkeyNet
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--trace',choices=['cupy','native']);a=p.parse_args();O=Path(a.output);O.mkdir(parents=True,exist_ok=False)
model=DonkeyNet(dropout_rate=0.);initial=initialize(model,(3,66,200),42);model.to_gpu();opt=SGD(.001).setup(model);ds=SyntheticDataset(256,(3,66,200),42)
sync=cp.cuda.get_current_stream().synchronize
@contextlib.contextmanager
def phase(name):
 cp.cuda.nvtx.RangePush('phase:'+name)
 try:yield
 finally:cp.cuda.nvtx.RangePop()
def restore():
 for k,v in parameters(model).items():v.data=cp.asarray(initial[k].copy())
 model.cleargrads();sync()
def step(batch):
 with phase('transfer'):x,t=map(cp.asarray,batch)
 with phase('forward'):loss=F.mean_square_error(model(x),t)
 with phase('backward'):model.cleargrads();loss.backward()
 with phase('update'):opt.update()
 return loss
modes=[a.trace] if a.trace else ['cupy','native'];rows=[];finals={};gradstates={}
with rt.backend(matmul='shared',conv='im2col',pool='native'):
 for mode in modes:
  restore();batch=next(iter(DataLoader(ds,4,shuffle=False)))
  with rt.backend(matmul='shared',conv='im2col',pool='native',col2im=mode):
   for _ in range(10):step(batch)
  sync()
 for repeat in range(1 if a.trace else 7):
  for mode in (modes if repeat%2==0 else modes[::-1]):
   restore()
   token=rt._config.set(dict(matmul='shared',conv='im2col',pool='native',tile=16,col2im=mode))
   if a.trace:cp.cuda.runtime.profilerStart()
   start=time.perf_counter();count=0
   for epoch in range(1 if a.trace else 3):
    it=iter(DataLoader(ds,4,shuffle=False))
    for i in range(10 if a.trace else 64):
     cp.cuda.nvtx.RangePush(f'step:{count}')
     with phase('data'):batch=next(it)
     loss=step(batch);cp.cuda.nvtx.RangePop();count+=1
   with phase('final_sync'):sync()
   elapsed=time.perf_counter()-start
   if a.trace:cp.cuda.runtime.profilerStop()
   rt._config.reset(token)
   state=snapshot(model);value=float(loss.data.item());finals[(mode,repeat)]=state
   grads={k:cp.asnumpy(v.grad.data) for k,v in parameters(model).items()}
   gradstates[(mode,repeat)]=grads
   assert all(v.dtype==np.float32 and v.grad.dtype==np.float32 for v in parameters(model).values())
   row=dict(mode=mode,repeat=repeat,steps=count,samples=count*4,wall_s=elapsed,samples_per_s=count*4/elapsed,loss=value,final_hash=state_hash(state));rows.append(row);print(row,flush=True)
if not a.trace:
 for row in rows:
  old=finals[('cupy',row['repeat'])];own=finals[(row['mode'],row['repeat'])]
  row.update(grad_max_abs_vs_cupy=max(float(np.max(np.abs(gradstates[(row['mode'],row['repeat'])][k]-gradstates[('cupy',row['repeat'])][k]))) for k in own),gradients_bitwise_cupy=all(np.array_equal(gradstates[(row['mode'],row['repeat'])][k],gradstates[('cupy',row['repeat'])][k]) for k in own),weight_max_abs_vs_old=max(float(np.max(np.abs(own[k]-old[k]))) for k in own),weights_bitwise_old=all(np.array_equal(own[k],old[k]) for k in own))
 assert all(r['weights_bitwise_old'] and r['gradients_bitwise_cupy'] for r in rows)
with (O/'training.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(O/'method.json').write_text(json.dumps(dict(seed=42,samples=256,batch=4,epochs=3,steps=192,repeats=7,warmup=10,lr=.001,initial_hash=state_hash(initial),dtype='float32 activations/weights/gradients',timing='includes sequential host loader, H2D, compute, final sync; no per-step sync/loss read/log; all modes same phase NVTX',only_difference='col2im context setting, same native library/GEMM128/input/model/optimizer'),indent=2))
