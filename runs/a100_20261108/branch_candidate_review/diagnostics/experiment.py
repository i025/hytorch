"""Matched current-branch ablations and isolated pre-change worker; no timing profiler."""
from pathlib import Path
import os,sys,time,csv,json,argparse,contextlib,hashlib,gc
O=Path(__file__).resolve().parents[1]
R=Path(os.environ.get('HYTORCH_REVIEW_ROOT',str(Path(__file__).resolve().parents[4])))
sys.path.insert(0,str(R))
import torch
import numpy as np
import cupy as cp
import hytorch as hy
import hytorch.functions as F
from hytorch.backends.cuda import runtime as rt
from hytorch.optimizers import SGD
from hytorch.monitor_loop import monitored_epochs
from hytorch.profiling import TrainingMonitor
from benchmarks.bench_cuda_training import initialize,parameters,snapshot,SyntheticDataset,TinyNet,state_hash
from donkeycar_hy.model import DonkeyNet
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
MODES=['baseline','relu','prune','heap','relu_prune','all']
@contextlib.contextmanager
def config(mode):
 kw=dict(matmul='shared',conv='im2col',pool='native',col2im='native',dweight='packed',metadata_cache=True)
 if 'relu_backward' in rt.current_config():kw['relu_backward']='fused' if mode in ('relu','relu_prune','all') else 'original'
 with rt.backend(**kw),hy.using_config('compact_function',True):
  if hasattr(hy.Config,'backward_schedule'):
   with hy.using_config('backward_schedule','heap' if mode in ('heap','all') else 'list'):yield
  else:yield

def write(name,rows):
 if rows:
  with (O/name).open('w') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

class Experiment:
 def __init__(self,kind='donkey'):
  self.kind=kind;self.shape=(3,66,200) if kind=='donkey' else (3,16,16)
  self.model=DonkeyNet(dropout_rate=0) if kind=='donkey' else TinyNet()
  self.initial=initialize(self.model,self.shape,42);self.model.to_gpu();self.ds=SyntheticDataset(256,self.shape,42)
 def run(self,mode,rep=0,steps=0,trace=False):
  for k,p in parameters(self.model).items():p.data=cp.asarray(self.initial[k])
  self.model.cleargrads();opt=SGD(.001,backend='legacy').setup(self.model)
  np.random.seed(42);cp.random.seed(42)
  prune=mode in ('prune','relu_prune','all')
  def predict(x):return self.model(hy.Variable(x,requires_grad=False) if prune else x)
  def backward(loss):self.model.cleargrads();loss.backward()
  @contextlib.contextmanager
  def nvtx(name):
   cp.cuda.nvtx.RangePush(name)
   try:yield
   finally:cp.cuda.nvtx.RangePop()
  gc.collect();cp.cuda.get_current_stream().synchronize();rt.dispatch_stats(reset=True)
  pool=cp.get_default_memory_pool()
  with config(mode),TrainingMonitor(enabled=False) as monitor:
   if trace:cp.cuda.runtime.profilerStart()
   start=time.perf_counter_ns()
   losses,_=monitored_epochs(self.ds,4,1 if steps else 3,monitor=monitor,transfer=cp.asarray,predict=predict,
     loss_fn=F.mean_square_error,backward=backward,update=opt.update,retain_loss=lambda l:l.data,
     steps_per_epoch=steps,phase_context=nvtx if trace else None)
   cp.cuda.get_current_stream().synchronize();elapsed=(time.perf_counter_ns()-start)/1e9
   if trace:cp.cuda.runtime.profilerStop()
  state=snapshot(self.model);grads={k:cp.asnumpy(p.grad.data) for k,p in parameters(self.model).items()};ls=cp.asnumpy(cp.stack([v[2] for v in losses]))
  row=dict(model=self.kind,mode=mode,repeat=rep,steps=len(losses),wall_s=elapsed,samples_per_s=4*len(losses)/elapsed,loss=float(ls[-1]),weights_sha256=state_hash(state),gradients_sha256=state_hash(grads),losses_sha256=hashlib.sha256(ls.tobytes()).hexdigest(),pool_reserved_end_bytes=pool.total_bytes(),dispatch_json=json.dumps(rt.dispatch_stats(),sort_keys=True))
  return row,state,grads,ls

def compare(a,b,rows):
 for i,kind in [(1,'weight'),(2,'gradient'),(3,'loss')]:
  aa=a[i] if i!=3 else {'all_steps':a[i]};bb=b[i] if i!=3 else {'all_steps':b[i]}
  for k,x in aa.items():
   y=bb[k];equal=x.tobytes()==y.tobytes();d=float(np.max(np.abs(x.astype('d')-y.astype('d'))))
   rows.append(dict(model=a[0]['model'],mode=a[0]['mode'],repeat=a[0]['repeat'],kind=kind,name=k,bitwise_equal=equal,max_abs=d))
   assert equal,rows[-1]

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--worker',choices=MODES);p.add_argument('--trace',choices=MODES);p.add_argument('--output');a=p.parse_args()
 e=Experiment()
 if a.worker or a.trace:
  mode=a.worker or a.trace;e.run(mode,steps=10);result=e.run(mode,steps=20 if a.trace else 0,trace=bool(a.trace))
  if a.output:Path(a.output).write_text(json.dumps(result[0],indent=2))
  print(json.dumps(result[0]));sys.exit(0)
 for m in MODES:e.run(m,steps=10)
 rows=[];checks=[]
 for rep in range(12):
  base=MODES if rep%2==0 else MODES[::-1];order=base[rep%6:]+base[:rep%6];res={}
  for pos,m in enumerate(order):
   r=e.run(m,rep);r[0]['order']=pos;res[m]=r;rows.append(r[0]);print(json.dumps(r[0]),flush=True)
  for m in MODES:compare(res[m],res['baseline'],checks)
  write('TRAINING_COMPARISON.csv',rows);write('NUMERICAL_ALIGNMENT.csv',checks)
 # Distinct model, five balanced controls (not DonkeyNet-only acceptance).
 e=Experiment('tiny')
 for m in ['baseline','relu_prune','all']:e.run(m,steps=10)
 for rep in range(6):
  order=['baseline','relu_prune','all'];order=order[rep%3:]+order[:rep%3];res={}
  for pos,m in enumerate(order):
   r=e.run(m,rep);r[0]['order']=pos;res[m]=r;rows.append(r[0])
  for m in order:compare(res[m],res['baseline'],checks)
  write('TRAINING_COMPARISON.csv',rows);write('NUMERICAL_ALIGNMENT.csv',checks)
 print('EXPERIMENT COMPLETED',flush=True)
