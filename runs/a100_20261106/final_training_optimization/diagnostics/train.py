from common import *
import argparse,gc
import hytorch.functions as F
from hytorch.optimizers import SGD
from hytorch.dataloaders import DataLoader
from benchmarks.bench_cuda_training import SyntheticDataset,initialize,parameters,snapshot,torch_model,TinyNet
from donkeycar_hy.model import DonkeyNet
p=argparse.ArgumentParser();p.add_argument('--task',choices=['main','references','validation','supplement','memory','trace'],default='main');p.add_argument('--mode',default='B0');p.add_argument('--tag',default='0');a=p.parse_args()
@contextlib.contextmanager
def phase(name):
 cp.cuda.nvtx.RangePush(name)
 try:yield
 finally:cp.cuda.nvtx.RangePop()
def model_setup(kind='donkey',batch=4):
 shape=(3,66,200) if kind=='donkey' else (3,16,16);model=DonkeyNet(dropout_rate=0.) if kind=='donkey' else TinyNet();initial=initialize(model,shape,42);model.to_gpu();return model,initial,SyntheticDataset(256,shape,42)
model,initial,ds=model_setup();sync=cp.cuda.get_current_stream().synchronize

def restore():
 for k,v in parameters(model).items():v.data=cp.asarray(initial[k].copy())
 model.cleargrads();np.random.seed(42);cp.random.seed(42);sync()
def step(batch,opt):
 with phase('transfer'):x,t=map(cp.asarray,batch)
 with phase('forward'):loss=F.mean_square_error(model(x),t)
 with phase('backward'):model.cleargrads();loss.backward()
 with phase('update'):opt.update()
 return loss

def run(mode,batch=4,steps=None,warmup=False,hook=None):
 restore();opt=SGD(.001,backend='native' if mode in ['S','DS'] else 'legacy').setup(model);losses=[];count=0
 with config(mode):
  if warmup:
   b=next(iter(DataLoader(ds,batch,shuffle=False)))
   for _ in range(10):step(b,opt)
   sync();restore()
  if a.task=='trace':cp.cuda.runtime.profilerStart()
  start=time.perf_counter()
  for epoch in range(3):
   for b in DataLoader(ds,batch,shuffle=False):
    with phase('step'):
     loss=step(b,opt);losses.append(loss.data)
    count+=1
    if hook:hook(count)
    if steps and count>=steps:break
   if steps and count>=steps:break
  sync();elapsed=time.perf_counter()-start
  if a.task=='trace':cp.cuda.runtime.profilerStop()
 state=snapshot(model);grads={k:cp.asnumpy(v.grad.data) for k,v in parameters(model).items()};lossarr=cp.asnumpy(cp.stack(losses));del losses
 assert np.all(np.isfinite(lossarr))
 return dict(mode=mode,steps=count,wall_s=elapsed,samples_per_s=count*batch/elapsed,loss=float(lossarr[-1])),state,grads,lossarr

rows=[];errors=[];lossrows=[]
def compare(result,ref,scope):
 row,state,grads,losses=result;_,base,bg,bl=ref
 for name,x,y,atol,rtol in [('weight',state,base,1e-4,1e-3),('gradient',grads,bg,2e-5,2e-4)]:
  for k in x:
   m=metrics(x[k],y[k],atol,rtol);errors.append(dict(scope=scope,mode=row['mode'],kind=name,parameter=k,**m));assert not m['failed'],(scope,row['mode'],name,k,m)
 m=metrics(losses,bl,1e-4,1e-3);assert not m['failed'];errors.append(dict(scope=scope,mode=row['mode'],kind='loss',parameter='trajectory',**m))
 for i,v in enumerate(losses):lossrows.append(dict(scope=scope,mode=row['mode'],step=i,loss=float(v)))

if a.task=='main':
 for mode in MODES:run(mode,steps=1,warmup=True)
 for repeat,order in enumerate(json.loads((O/'raw/order.json').read_text())):
  results={}
  for mode in order:
   result=run(mode);row=result[0];row['repeat']=repeat;rows.append(row);results[mode]=result;print(row,flush=True)
  for mode in MODES:compare(results[mode],results['B0'],str(repeat))
 write(O/'TRAINING.csv',rows);write(O/'TRAINING_ERRORS.csv',errors);write(O/'LOSS_TRAJECTORIES.csv',lossrows)
elif a.task=='validation':
 results={mode:run(mode,steps=1) for mode in MODES}
 for mode in MODES:compare(results[mode],results['B0'],'single_step')
 write(O/'SINGLE_STEP.csv',errors)
 captured=[];original_forward=FC.Conv2DGradW.forward;stepindex=[0];layer=[0]
 def cap(self,x,gy):
  if stepindex[0] in [0,95,191]:captured.append((stepindex[0],layer[0],cp.asnumpy(x),cp.asnumpy(gy),self.kernel_size,self.stride,self.pad))
  layer[0]=(layer[0]+1)%5
  return original_forward(self,x,gy)
 FC.Conv2DGradW.forward=cap
 try:run('B0',hook=lambda count:stepindex.__setitem__(0,count))
 finally:FC.Conv2DGradW.forward=original_forward
 numeric=[];ok=True
 for st,ly,x,g,k,s,pad in captured:ok &= validate(cp.asarray(x),cp.asarray(g),k,s,pad,f'train/{st}/{ly}',numeric)
 write(O/'TRAINING_DWEIGHT_NUMERICS.csv',numeric);assert ok
elif a.task=='supplement':
 for kind,batch in [('donkey',1),('donkey',16),('tiny',4)]:
  model,initial,ds=model_setup(kind,batch)
  for mode in ['B0','D']:run(mode,batch,steps=1,warmup=True)
  for repeat in range(3):
   result={}
   for mode in (['B0','D'] if repeat%2==0 else ['D','B0']):
    result[mode]=run(mode,batch);row=result[mode][0];row.update(model=kind,batch=batch,repeat=repeat);rows.append(row);print(row,flush=True)
   compare(result['D'],result['B0'],f'{kind}/{batch}/{repeat}')
 write(O/'SUPPLEMENT.csv',rows);write(O/'SUPPLEMENT_ERRORS.csv',errors)
elif a.task=='references':
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 tm=torch_model('donkey',initial).cuda();torch_initial={k:v.clone() for k,v in tm.state_dict().items()};torch.set_num_threads(4)
 def trun(warm=False):
  tm.load_state_dict(torch_initial);opt=torch.optim.SGD(tm.parameters(),lr=.001);ls=[];torch.cuda.synchronize();t=time.perf_counter()
  for epoch in range(1 if warm else 3):
   for i,b in enumerate(DataLoader(ds,4,shuffle=False)):
    x,y=[torch.from_numpy(z).cuda() for z in b];out=tm(x);loss=torch.nn.functional.mse_loss(out,y);opt.zero_grad(set_to_none=True);loss.backward();opt.step();ls.append(loss.detach())
    if warm and i==9:break
  torch.cuda.synchronize();elapsed=time.perf_counter()-t
  vals=torch.stack(ls).cpu().numpy();state={};grads={}
  for k in initial:
   layer,param=k.split('/');v=getattr(getattr(tm,layer),'weight' if param=='W' else 'bias');z=v.detach().cpu().numpy();g=v.grad.cpu().numpy();state[k]=z.T.copy() if z.ndim==2 else z.copy();grads[k]=g.T.copy() if g.ndim==2 else g.copy()
  return dict(mode='torch',wall_s=elapsed,steps=len(ls),samples_per_s=len(ls)*4/elapsed,loss=float(vals[-1])),state,grads,vals
 run('cupy',steps=1,warmup=True);trun(True);reference=run('B0')
 for repeat in range(5):
  for mode in (['cupy','torch'] if repeat%2==0 else ['torch','cupy']):
   result=run('cupy') if mode=='cupy' else trun();row=result[0];row['repeat']=repeat;rows.append(row);print(row,flush=True);compare(result,reference,f'reference/{repeat}')
 write(O/'REFERENCES.csv',rows);write(O/'REFERENCE_ERRORS.csv',errors)
 (O/'raw/torch_config.json').write_text(json.dumps(dict(version=torch.__version__,tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32,cudnn_enabled=torch.backends.cudnn.enabled,cudnn_benchmark=torch.backends.cudnn.benchmark,optimizer='SGD',foreach=None,fused=None,set_to_none=True),indent=2))
elif a.task=='trace':
 # Category NVTX exists only in independent profiles, not formal wall measurements.
 for cls,label in [(FC.Conv2DGradW,'dweight'),(FC.Deconv2d,'dinput')]:
  original=getattr(cls,'forward')
  def traced(self,*args,_f=original,_label=label):
   with phase(_label):return _f(self,*args)
  cls.forward=traced
 # Warmup outside capture.
 restore();opt=SGD(.001,backend='native' if a.mode in ['S','DS'] else 'legacy').setup(model)
 with config(a.mode):
  b=next(iter(DataLoader(ds,4,shuffle=False)))
  for _ in range(10):step(b,opt)
 sync();result=run(a.mode,steps=10);write(O/'raw'/f'trace_{a.mode}_{a.tag}.csv',[result[0]])
elif a.task=='memory':
 pool=cp.get_default_memory_pool()
 class Peak(cp.cuda.MemoryHook):
  def __init__(self):self.peak=0;self.reserved=0;self.calls=0
  def malloc_postprocess(self,**kwargs):self.calls+=1;self.peak=max(self.peak,pool.used_bytes());self.reserved=max(self.reserved,pool.total_bytes())
 for batch in [1,4,16]:
  for mode in ['B0','D']:
   model,initial,ds=model_setup();model.cleargrads();gc.collect();sync();pool.free_all_blocks()
   for repeat in range(3):
    pk=Peak()
    with pk:run(mode,batch)
    model.cleargrads();gc.collect();sync();rows.append(dict(mode=mode,batch=batch,repeat=repeat,peak_active_bytes=pk.peak,peak_reserved_bytes=pk.reserved,after_clear_active_bytes=pool.used_bytes(),after_clear_reserved_bytes=pool.total_bytes(),malloc_calls=pk.calls,method='CuPy MemoryHook malloc_postprocess; pool only, excludes torch/CUDA-library private allocations'))
 write(O/'MEMORY.csv',rows)
