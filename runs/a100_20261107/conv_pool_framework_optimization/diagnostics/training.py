from common import *
import argparse,gc,cProfile,pstats
from benchmarks.bench_cuda_training import SyntheticDataset,initialize,parameters,snapshot,torch_model,TinyNet
from hytorch.dataloaders import DataLoader
from hytorch.optimizers import SGD
from donkeycar_hy.model import DonkeyNet
p=argparse.ArgumentParser();p.add_argument('--task',choices=['baseline','cpu','trace','main','supplement','memory','workload','balanced','memory_torch'],required=True);p.add_argument('--mode',default='B0');p.add_argument('--tag',default='a');a=p.parse_args()
class Experiment:
 def __init__(self,kind='donkey',batch=4):
  self.batch=batch;self.kind=kind;shape=(3,66,200) if kind=='donkey' else (3,16,16)
  self.hy=DonkeyNet(dropout_rate=0) if kind=='donkey' else TinyNet();self.initial=initialize(self.hy,shape,42);self.hy.to_gpu();self.ds=SyntheticDataset(256,shape,42)
  self.tm=torch_model(kind,self.initial).cuda();self.saved={k:v.clone() for k,v in self.tm.state_dict().items()}
 def run(self,mode,steps=192,trace=False,warm=False):
  ist=mode=='torch';model=self.tm if ist else self.hy
  if ist:model.load_state_dict(self.saved);opt=torch.optim.SGD(model.parameters(),lr=.001)
  else:
   for k,v in parameters(model).items():v.data=cp.asarray(self.initial[k])
   model.cleargrads();opt=SGD(.001,backend='legacy').setup(model)
  np.random.seed(42);cp.random.seed(42);torch.manual_seed(42);losses=[];count=0
  ctx=contextlib.nullcontext() if ist else config(mode)
  with ctx:
   sync()
   if trace:cp.cuda.runtime.profilerStart()
   start=time.perf_counter()
   while count<steps:
    for batch in DataLoader(self.ds,self.batch,shuffle=False):
     with phase('step',trace):
      with phase('transfer',trace):
       x,t=[torch.from_numpy(v).cuda() for v in batch] if ist else [cp.asarray(v) for v in batch]
      with phase('forward',trace):
       loss=torch.nn.functional.mse_loss(model(x),t) if ist else F.mean_square_error(model(x),t)
      with phase('backward',trace):
       if ist:opt.zero_grad(set_to_none=True)
       else:model.cleargrads()
       loss.backward()
      with phase('update',trace):
       if ist:opt.step()
       else:opt.update()
      losses.append(loss.detach() if ist else loss.data)
     count+=1
     if count>=steps:break
   sync();elapsed=time.perf_counter()-start
   if trace:cp.cuda.runtime.profilerStop()
  if ist:
   state={};grads={}
   for k in self.initial:
    layer,param=k.split('/');v=getattr(getattr(model,layer),'weight' if param=='W' else 'bias');z=host(v);g=host(v.grad)
    state[k]=z.T.copy() if z.ndim==2 else z.copy();grads[k]=g.T.copy() if g.ndim==2 else g.copy()
   ls=host(torch.stack(losses))
  else:state=snapshot(model);grads={k:host(v.grad).copy() for k,v in parameters(model).items()};ls=host(cp.stack(losses))
  assert np.isfinite(ls).all()
  return dict(mode=mode,model=self.kind,batch=self.batch,steps=steps,wall_s=elapsed,samples_per_s=steps*self.batch/elapsed,loss=float(ls[-1])),state,grads,ls
 def warm(self,modes):
  for m in modes:self.run(m,10)

rows=[];errors=[];lossrows=[]
def compare(results,scope):
 ref=results['B0']
 for mode,result in results.items():
  for label,index,atol,rtol in [('weight',1,1e-4,1e-3),('gradient',2,2e-5,2e-4)]:
   for k,v in result[index].items():
    e=error(v,ref[index][k],atol,rtol);errors.append(dict(scope=scope,mode=mode,kind=label,parameter=k,**e));assert not e['failed'],errors[-1];assert mode!='O1' or e['bitwise_equal'],errors[-1]
  e=error(result[3],ref[3],1e-4,1e-3);errors.append(dict(scope=scope,mode=mode,kind='loss',parameter='all',**e));assert not e['failed'],errors[-1];assert mode!='O1' or e['bitwise_equal'],errors[-1]
  lossrows.extend(dict(scope=scope,mode=mode,step=i,loss=float(v)) for i,v in enumerate(result[3]))

e=Experiment()
if a.task=='cpu':
 e.warm([a.mode]);pr=cProfile.Profile();pr.enable();e.run(a.mode,30);pr.disable();pr.dump_stats(str(O/'raw'/f'cpu_{a.mode}.prof'))
 with (O/'raw'/f'cpu_{a.mode}.txt').open('w') as f:pstats.Stats(pr,stream=f).strip_dirs().sort_stats('cumulative').print_stats(90)
 st=pstats.Stats(pr);write(O/'raw'/f'cpu_{a.mode}.csv',[dict(file=k[0],line=k[1],function=k[2],primitive_calls=v[0],calls=v[1],self_s=v[2],cumulative_s=v[3]) for k,v in st.stats.items()])
elif a.task=='trace':e.warm([a.mode]);e.run(a.mode,10,trace=True)
elif a.task in ['baseline','main','balanced']:
 modes=['B0','torch'] if a.task=='baseline' else ['B0','O1','torch'];e.warm(modes)
 for rep in range(5 if a.task=='baseline' else (12 if a.task=='balanced' else 10)):
  order=(list(__import__('itertools').permutations(modes))[rep%6] if a.task=='balanced' else (modes if rep%2==0 else modes[::-1]));results={}
  for mode in order:
   r=e.run(mode);r[0].update(repeat=rep,order=order.index(mode));rows.append(r[0]);results[mode]=r;print(r[0],flush=True)
  compare(results,str(rep))
 prefix='stage_a/' if a.task=='baseline' else ''
 write(O/(prefix+'TRAINING_ABLATION.csv'),rows);write(O/(prefix+'TRAINING_ERRORS.csv'),errors);write(O/(prefix+'LOSS.csv'),lossrows)
elif a.task=='supplement':
 for kind,batch in [('donkey',1),('donkey',16),('tiny',4)]:
  e=Experiment(kind,batch);e.warm(['B0','O1'])
  for rep in range(3):
   results={}
   for m in (['B0','O1'] if rep%2==0 else ['O1','B0']):
    r=e.run(m,768//batch);r[0]['repeat']=rep;rows.append(r[0]);results[m]=r;print(r[0],flush=True)
   compare(results,f'{kind}/{batch}/{rep}')
 write(O/'SUPPLEMENT.csv',rows);write(O/'SUPPLEMENT_ERRORS.csv',errors)
elif a.task=='memory':
 pool=cp.get_default_memory_pool()
 class Peak(cp.cuda.MemoryHook):
  def __init__(self):self.calls=0;self.peak=0;self.reserved=0
  def malloc_postprocess(self,**kw):self.calls+=1;self.peak=max(self.peak,pool.used_bytes());self.reserved=max(self.reserved,pool.total_bytes())
 for batch in [1,4,16]:
  e=Experiment(batch=batch)
  for mode in ['B0','O1']:
   e.warm([mode]);e.hy.cleargrads();gc.collect();sync();pool.free_all_blocks()
   for rep in range(3):
    hook=Peak()
    with hook:r=e.run(mode,768//batch)
    e.hy.cleargrads();gc.collect();sync();rows.append(dict(mode=mode,batch=batch,repeat=rep,malloc_calls=hook.calls,peak_active_bytes=hook.peak,peak_reserved_bytes=hook.reserved,after_clear_bytes=pool.used_bytes()))
 write(O/'MEMORY_COMPARISON.csv',rows)
elif a.task=='memory_torch':
 for batch in [1,4,16]:
  e=Experiment(batch=batch);e.warm(['torch'])
  for rep in range(3):
   gc.collect();sync();torch.cuda.reset_peak_memory_stats();start_bytes=torch.cuda.memory_allocated()
   torch.cuda.memory._record_memory_history(enabled='all',context=None,max_entries=200000,clear_history=True)
   r=e.run('torch',768//batch);sync();peak=torch.cuda.max_memory_allocated()
   events=torch.cuda.memory._snapshot()['device_traces'][0];torch.cuda.memory._record_memory_history(enabled=None)
   rows.append(dict(mode='torch',batch=batch,repeat=rep,malloc_calls=sum(v['action']=='alloc' for v in events),peak_active_bytes=peak,initial_active_bytes=start_bytes,after_clear_bytes=torch.cuda.memory_allocated(),note='includes parameter reset and result capture; diagnostic only'))
 write(O/'raw/MEMORY_TORCH.csv',rows)
elif a.task=='workload':
 records=[];gemms=[];orig=hytorch.core.Function.__call__;og=rt.gemm;odw=rt.packed_dweight;otd=cp.tensordot
 def meta(x):
  x=x.data if isinstance(x,hytorch.Variable) else x
  return dict(shape=list(x.shape),dtype=str(x.dtype),strides=list(x.strides),c_contiguous=bool(x.flags.c_contiguous)) if isinstance(x,(np.ndarray,cp.ndarray)) else None
 def call(self,*inputs):
  out=orig(self,*inputs);records.append(dict(function=type(self).__name__,inputs=[meta(x) for x in inputs],outputs=[meta(x) for x in out] if isinstance(out,list) else [meta(out)],stride=getattr(self,'stride',None),pad=getattr(self,'pad',None)));return out
 def gem(a,b,*args,**kw):gemms.append(dict(M=a.shape[0],N=b.shape[1],K=a.shape[1],a=meta(a),b=meta(b)));return og(a,b,*args,**kw)
 def dw(x,g,k,s,p):
  gemms.append(dict(path='packed_dweight_cublas',M=g.shape[1],N=x.shape[1]*k[0]*k[1],K=g.shape[0]*g.shape[2]*g.shape[3],x=meta(x),gy=meta(g)));return odw(x,g,k,s,p)
 def td(x,y,axes=2):
  gemms.append(dict(path='cupy_tensordot',a=meta(x),b=meta(y),axes=axes));return otd(x,y,axes)
 hytorch.core.Function.__call__=call;rt.gemm=gem;rt.packed_dweight=dw;cp.tensordot=td
 e.run('B0',1)
 (O/'WORKLOAD_COMPLETE.json').write_text(json.dumps(dict(function_calls=records,native_gemm_calls=gemms),indent=2))
 env=dict(torch=torch.__version__,cupy=cp.__version__,cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),cudnn=torch.backends.cudnn.version(),tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32,cudnn_enabled=torch.backends.cudnn.enabled,cudnn_benchmark=False,optimizer='torch SGD foreach=None fused=None; HyTorch legacy',gpu=str(cp.cuda.runtime.getDeviceProperties(0)['name']))
 (O/'ENVIRONMENT.json').write_text(json.dumps(env,indent=2))
