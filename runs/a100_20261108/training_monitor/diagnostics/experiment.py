"""Fixed-seed paired monitoring experiment. No production kernels are modified."""
from pathlib import Path
import os,sys,time,json,csv,hashlib,contextlib,gc,subprocess
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='4'
os.environ['NVIDIA_TF32_OVERRIDE']='0'
R=Path(__file__).resolve().parents[4];O=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import torch
import cupy as cp
import numpy as np
import hytorch as hy
import hytorch.functions as F
from hytorch.backends.cuda import runtime as rt
from hytorch.profiling import TrainingMonitor
from hytorch.monitor_loop import monitored_epochs
from hytorch.optimizers import SGD
from benchmarks.bench_cuda_training import SyntheticDataset,initialize,parameters,snapshot,torch_model,TinyNet
from donkeycar_hy.model import DonkeyNet

torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True

def write(name,rows):
 if not rows:return
 with (O/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
@contextlib.contextmanager
def config():
 with rt.backend(matmul='shared',conv='im2col',pool='native',col2im='native',dweight='packed',metadata_cache=True),hy.core.using_config('compact_function',True):yield
@contextlib.contextmanager
def nvtx(name):
 cp.cuda.nvtx.RangePush(name)
 try:yield
 finally:cp.cuda.nvtx.RangePop()

def hashes(state):return {k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in state.items()}

class Experiment:
 def __init__(self,kind='donkey'):
  self.kind=kind;shape=(3,66,200) if kind=='donkey' else (3,16,16)
  self.hy=DonkeyNet(dropout_rate=0) if kind=='donkey' else TinyNet()
  self.initial=initialize(self.hy,shape,42);self.hy.to_gpu();self.ds=SyntheticDataset(256,shape,42);self.tm=None
 def run(self,mode='disabled',rep=0,tag='formal',epochs=3,steps=0,framework='hytorch',latency=False,peaks=False,trace=False):
  ist=framework=='torch'
  if ist:
   if self.tm is None:self.tm=torch_model(self.kind,self.initial).cuda();self.saved={k:v.clone() for k,v in self.tm.state_dict().items()}
   model=self.tm;model.load_state_dict(self.saved);optimizer=torch.optim.SGD(model.parameters(),lr=.001)
   transfer=lambda x:torch.from_numpy(x).cuda();loss_fn=torch.nn.functional.mse_loss
   backward=lambda loss:(optimizer.zero_grad(set_to_none=True),loss.backward());update=optimizer.step;retain=lambda loss:loss.detach()
   sync=lambda:torch.cuda.synchronize(0)
  else:
   model=self.hy
   for k,v in parameters(model).items():v.data=cp.asarray(self.initial[k])
   model.cleargrads();optimizer=SGD(.001,backend='legacy').setup(model)
   transfer=cp.asarray;loss_fn=F.mean_square_error;backward=lambda loss:(model.cleargrads(),loss.backward());update=optimizer.update;retain=lambda loss:loss.data
   sync=cp.cuda.get_current_stream().synchronize
  np.random.seed(42);cp.random.seed(42);torch.manual_seed(42)
  monitor=TrainingMonitor(enabled=mode!='disabled',timing=mode in ('timing','all'),resources=mode in ('resources','all'),
    gpu_index=0,cuda_backend='torch' if ist else 'cupy',sample_interval=.2,output_dir=O/'raw/runs'/f'{tag}_{self.kind}_{framework}_{rep}_{mode}',
    mode='latency' if latency else 'throughput',allocator_peaks=peaks)
  sync()
  # Matched outer completion includes monitor initialization, final event reads and export.
  with config() if not ist else contextlib.nullcontext():
   start_outer=time.perf_counter_ns()
   with monitor:
    if trace:cp.cuda.runtime.profilerStart()
    start_train=time.perf_counter_ns()
    losses,epoch_rows=monitored_epochs(self.ds,4,epochs,monitor=monitor,transfer=transfer,predict=model,loss_fn=loss_fn,backward=backward,update=update,
       retain_loss=retain,steps_per_epoch=steps,phase_context=nvtx if trace else None)
    sync();end_train=time.perf_counter_ns()
    if trace:cp.cuda.runtime.profilerStop()
   end_outer=time.perf_counter_ns()
  if ist:
   state={};grads={}
   for k in self.initial:
    layer,param=k.split('/');v=getattr(getattr(model,layer),'weight' if param=='W' else 'bias');x=v.detach().cpu().numpy();g=v.grad.detach().cpu().numpy()
    state[k]=x.T.copy() if x.ndim==2 else x.copy();grads[k]=g.T.copy() if g.ndim==2 else g.copy()
   ls=torch.stack([v[2] for v in losses]).cpu().numpy()
  else:state=snapshot(model);grads={k:cp.asnumpy(v.grad.data) for k,v in parameters(model).items()};ls=cp.asnumpy(cp.stack([v[2] for v in losses]))
  assert np.isfinite(ls).all()
  row=dict(tag=tag,model=self.kind,framework=framework,mode=mode,repeat=rep,steps=len(losses),samples=len(losses)*4,
     wall_s=(end_outer-start_outer)/1e9,training_window_s=(end_train-start_train)/1e9,
     samples_per_s=len(losses)*4/((end_outer-start_outer)/1e9),training_window_samples_per_s=len(losses)*4/((end_train-start_train)/1e9),
     monitor_setup_finalize_export_s=((end_outer-start_outer)-(end_train-start_train))/1e9,loss=float(ls[-1]),
     sample_count=len(monitor.gpu_samples),interval_s=.2,latency_diagnostic=latency,allocator_peak_diagnostic=peaks,
     train_start_ns=start_train,train_end_ns=end_train,run_dir=str(monitor.run_dir) if monitor.run_dir else '')
  if tag not in ('warm','trace'):
   all_runs.append(row)
   for rows,target in [(monitor.gpu_samples,gpu_samples),(monitor.cpu_samples,cpu_samples),(monitor.steps,step_rows)]:
    for v in rows:
     tt=v.get('monotonic_ns',v.get('start_monotonic_ns'))
     target.append(dict(v,tag=tag,model=self.kind,framework=framework,monitor_components=mode,monitor_mode=monitor.mode,repeat=rep,run_dir=row['run_dir'],train_start_ns=start_train,train_end_ns=end_train,inside_train=start_train<=tt<=end_train))
   if monitor.run_dir:(monitor.run_dir/'EXPERIMENT.json').write_text(json.dumps(dict(row=row,monitor=monitor.summary(),weights_sha256=hashes(state),gradients_sha256=hashes(grads),loss_sha256=hashlib.sha256(ls.tobytes()).hexdigest()),indent=2))
   print(json.dumps(row),flush=True)
  return row,state,grads,ls
 def warm(self,framework='hytorch'):
  self.run('disabled',tag='warm',epochs=1,steps=10,framework=framework)

all_runs=[];gpu_samples=[];cpu_samples=[];step_rows=[];errors=[]
def compare(result,reference,label):
 for index,kind,atol,rtol in [(1,'weight',1e-4,1e-3),(2,'gradient',2e-5,2e-4),(3,'loss',1e-4,1e-3)]:
  aa=result[index] if index!=3 else {'all_steps':result[index]};bb=reference[index] if index!=3 else {'all_steps':reference[index]}
  for k,a in aa.items():
   b=bb[k];d=np.abs(a.astype('d')-b.astype('d'));same=a.tobytes()==b.tobytes();fail=int(np.count_nonzero(d>atol+rtol*np.abs(b)))
   row=dict(label=label,kind=kind,parameter=k,bitwise_equal=same,max_abs=float(d.max()),relative_l2=float(np.linalg.norm(d.ravel())/max(np.linalg.norm(b.ravel()),1e-30)),failed_elements=fail,atol=atol,rtol=rtol)
   errors.append(row)
   assert fail==0,row
   if result[0]['framework']==reference[0]['framework']:assert same,row

def save():
 write('RUNS.csv',all_runs);write('GPU_SAMPLES.csv',gpu_samples);write('CPU_SAMPLES.csv',cpu_samples);write('STEP_TIMINGS.csv',step_rows);write('raw/NUMERICAL_ALIGNMENT.csv',errors)

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--trace',action='store_true');a=p.parse_args()
 e=Experiment();e.warm()
 if a.trace:
  e.run('disabled',tag='trace',epochs=1,steps=20,trace=True);sys.exit(0)
 (O/'raw/experiment_environment.json').write_text(json.dumps(dict(python=sys.version,numpy=np.__version__,cupy=cp.__version__,torch=torch.__version__,cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),gpu=str(cp.cuda.runtime.getDeviceProperties(0)),nvml_binding=__import__('pynvml').__file__,cpu_threads=4,tf32_override=os.environ['NVIDIA_TF32_OVERRIDE'],cudnn_deterministic=True,seed=42,batch=4,samples=256,epochs=3,steps=192,config=rt.current_config(),final_config='shared/im2col/native pool/native col2im/packed dweight/F1=True/O1=True/legacy SGD .001'),indent=2))
 modes=['disabled','timing','resources','all']
 # Initialize optional imports and sampler once outside all formal measurements.
 for mode in modes:e.run(mode,tag='warm',epochs=1,steps=3)
 for rep in range(12):
  base=modes if (rep//4)%2==0 else modes[::-1];order=base[rep%4:]+base[:rep%4];results={}
  for pos,mode in enumerate(order):
   results[mode]=e.run(mode,rep=rep);results[mode][0]['order']=pos
  for mode in modes:compare(results[mode],results['disabled'],f'formal/{rep}/{mode}')
  save()
 # Complete standard training with allocator-hook diagnostics; separate from formal overhead.
 functional=e.run('all',tag='functional',peaks=True);baseline=e.run('disabled',tag='functional_control');compare(functional,baseline,'functional_allocator')
 latency=e.run('all',tag='latency',latency=True);compare(latency,baseline,'latency')
 # Longer synthetic window gives ~40 hardware samples at 0.2s rather than interpreting 4 samples per short run.
 e.run('all',tag='trend',epochs=30);save()
 e.warm('torch')
 for rep in range(3):
  rr={}
  for mode in (['disabled','all'] if rep%2==0 else ['all','disabled']):rr[mode]=e.run(mode,rep=rep,tag='torch_reference',framework='torch')
  compare(rr['all'],rr['disabled'],f'torch_monitor/{rep}');compare(rr['disabled'],baseline,f'torch_vs_hytorch/{rep}')
  save()
 tiny=Experiment('tiny');tiny.warm()
 rr={mode:tiny.run(mode,tag='tiny_smoke') for mode in modes}
 for mode in modes:compare(rr[mode],rr['disabled'],'tiny/'+mode)
 save()
 print('ALL EXPERIMENTS COMPLETED',flush=True)
