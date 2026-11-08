from common import *
import argparse,statistics,ctypes as ct
p=argparse.ArgumentParser();p.add_argument('--trace',action='store_true');p.add_argument('--trace-repeats',type=int,default=3);p.add_argument('--allocation-only',action='store_true');p.add_argument('--mode',default='B0');p.add_argument('--tag',default='a');args=p.parse_args()
rng=np.random.default_rng(42);cases=[]
def tensor(shape):
 a=rng.normal(0,.1,shape).astype(np.float32);return cp.asarray(a),torch.tensor(a,device='cuda',requires_grad=True)
def add(name,category,layer,funcs,geometry=None):cases.append((name,category,layer,funcs,geometry or {}))
specs=[('conv1',(4,3,66,200),24,(5,5),(2,2),(0,0)),('conv2',(4,24,31,98),36,(5,5),(2,2),(0,0)),('conv3',(4,36,14,47),48,(5,5),(2,2),(0,0)),('conv4',(4,48,5,22),64,(3,3),(1,1),(0,0)),('conv5',(4,64,3,20),64,(3,3),(1,1),(0,0)),('irregular',(2,3,13,17),7,(3,5),(2,3),(1,2)),('asymmetric',(1,5,19,11),9,(5,1),(1,2),(2,0))]
def convcase(spec):
 name,shape,oc,k,s,p=spec;x,tx=tensor(shape);w,tw=tensor((oc,shape[1],*k));b,tb=tensor((oc,));vx,vw,vb=map(hytorch.Variable,(x,w,b));y=FC.conv2d(vx,vw,vb,s,p);ty=torch.nn.functional.conv2d(tx,tw,tb,s,p);g,tg=tensor(y.shape);tg=tg.detach();vg=hytorch.Variable(g);dw=FC.Conv2DGradW(y.creator)
 def hb():
  for v in [vx,vw,vb]:v.cleargrad()
  y.grad=vg;y.backward();return (vx.grad,vw.grad,vb.grad)
 tyx=torch.nn.functional.conv2d(tx,tw.detach(),tb.detach(),s,p)
 tyw=torch.nn.functional.conv2d(tx.detach(),tw,tb.detach(),s,p)
 tyb=torch.nn.functional.conv2d(tx.detach(),tw.detach(),tb,s,p)
 geo=dict(shape=shape,kernel=k,stride=s,pad=p,output=y.shape)
 add(name+'/forward','ConvForward',name,{'HyTorch':lambda:FC.conv2d(vx,vw,vb,s,p),'PyTorch':lambda:torch.nn.functional.conv2d(tx,tw,tb,s,p)},geo)
 add(name+'/dweight','dWeight',name,{'HyTorch':lambda:dw(vx,vg),'PyTorch':lambda:torch.autograd.grad(tyw,tw,tg,retain_graph=True)[0]},geo)
 add(name+'/dinput','dInput',name,{'HyTorch':lambda:FC.deconv2d(vg,vw,None,s,p,shape[2:]),'PyTorch':lambda:torch.autograd.grad(tyx,tx,tg,retain_graph=True)[0]},geo)
 add(name+'/dbias','dBias',name,{'HyTorch':lambda:vg.sum(axis=(0,2,3)),'PyTorch':lambda:torch.autograd.grad(tyb,tb,tg,retain_graph=True)[0]},geo)
 add(name+'/backward','FullBackward',name,{'HyTorch':hb,'PyTorch':lambda:torch.autograd.grad(ty,(tx,tw,tb),tg,retain_graph=True)},geo)

with config(args.mode):
 for sp in specs:convcase(sp)
 # Old library provides always-register dispatch for eligible shapes, never loaded as production runtime.
 old=ct.CDLL(str(O/'baseline/lib_register_pre_dispatch.so'));old.hy_gemm.argtypes=[ct.c_void_p]*3+[ct.c_int]*5+[ct.c_void_p];old.hy_gemm.restype=ct.c_int
 shapes=[(12152,24,75),(2632,36,600),(440,48,900),(240,64,432),(72,64,576),(4,100,1152),(4,50,100),(4,1,50),(1024,1024,1024),(127,257,65),(65,4097,129)]
 def gemcase(sh):
  m,n,k=sh;x,tx=tensor((m,k));w,tw=tensor((k,n));out=cp.empty((m,n),cp.float32);tout=torch.empty((m,n),device='cuda');tx=tx.detach();tw=tw.detach()
  def reg():
   status=old.hy_gemm(x.data.ptr,w.data.ptr,out.data.ptr,m,n,k,2,16,cp.cuda.get_current_stream().ptr)
   if status:raise RuntimeError(status)
   return out
  fs={v:lambda v=v:rt.gemm(x,w,v,16,out) for v in ['basic','blocked','shared']};fs['register_old_dispatch']=reg;fs['PyTorch']=lambda:torch.matmul(tx,tw,out=tout)
  add(f'gemm/{m}/{n}/{k}','GEMM',str(sh),fs,dict(M=m,N=n,K=k,preallocated=True))
 for sh in shapes:gemcase(sh)
 def other():
  for shape in [(4,24,31,98),(4,64,1,18),(3,7,11,13)]:
   def act(shape):
    x,tx=tensor(shape);vx=hytorch.Variable(x)
    add('relu/'+str(shape),'Activation',str(shape),{'HyTorch':lambda:F.relu(vx),'PyTorch':lambda:torch.relu(tx)})
    add('tanh/'+str(shape),'Activation',str(shape),{'HyTorch':lambda:F.tanh(vx),'PyTorch':lambda:torch.tanh(tx)})
   act(shape)
  for shape in [(4,3,16,16),(2,7,13,17)]:
   def pool(shape):
    x,tx=tensor(shape);vx=hytorch.Variable(x);y=F.pooling(vx,2,2);ty=torch.nn.functional.max_pool2d(tx,2,2);g,tg=tensor(y.shape);tg=tg.detach()
    def back():vx.cleargrad();y.grad=hytorch.Variable(g);y.backward();return vx.grad
    add('pool/'+str(shape),'Pooling',str(shape),{'HyTorch':lambda:F.pooling(vx,2,2),'PyTorch':lambda:torch.nn.functional.max_pool2d(tx,2,2)})
    add('poolback/'+str(shape),'PoolBackward',str(shape),{'HyTorch':back,'PyTorch':lambda:torch.autograd.grad(ty,tx,tg,retain_graph=True)[0]})
   pool(shape)
  for m,k,n in [(4,1152,100),(4,100,50),(4,50,1),(3,137,71)]:
   def linear(m,k,n):
    x,tx=tensor((m,k));w,tw=tensor((k,n));b,tb=tensor((n,));vx,vw,vb=map(hytorch.Variable,(x,w,b))
    add(f'linear/{m}/{k}/{n}','Linear',str((m,k,n)),{'HyTorch':lambda:F.linear(vx,vw,vb),'PyTorch':lambda:torch.nn.functional.linear(tx,tw.T,tb)})
   linear(m,k,n)
  from hytorch.optimizers import SGD
  from benchmarks.bench_cuda_training import initialize,parameters
  from donkeycar_hy.model import DonkeyNet
  model=DonkeyNet(dropout_rate=0);initial=initialize(model,(3,66,200),42);model.to_gpu();hp=list(parameters(model).values());tp=[]
  for v in hp:
   g=rng.normal(0,.01,v.shape).astype(np.float32);v.grad=hytorch.Variable(cp.asarray(g));t=torch.nn.Parameter(torch.tensor(host(v),device='cuda'));t.grad=torch.tensor(g,device='cuda');tp.append(t)
  ho=SGD(.001).setup(model);to=torch.optim.SGD(tp,lr=.001);fo=torch.optim.SGD(tp,lr=.001,foreach=True)
  # Three paths have independent state, so numerical comparison is performed separately below.
  init=[host(v).copy() for v in hp]
  def reset_sgd():
   with torch.no_grad():
    for v,t,z in zip(hp,tp,init):v.data[...] =cp.asarray(z);t.copy_(torch.tensor(z,device='cuda'))
  def hu():ho.update();return [v.data for v in hp]
  def tu():to.step();return tp
  def fu():fo.step();return tp
  add('sgd/all16','SGD','all16',{'HyTorch':hu,'PyTorch':tu,'PyTorch_foreach':fu},{'reset':reset_sgd,'tensors':len(hp)})
 other()
 raw=[];errs=[];mem=[]
 class Alloc(cp.cuda.MemoryHook):
  def __init__(self):self.calls=0;self.bytes=0
  def malloc_postprocess(self,**kw):self.calls+=1;self.bytes+=kw.get('size',0)
 def outputs(x):return x if isinstance(x,(tuple,list)) else [x]
 # Supplemental allocator diagnostics, never used as formal timing.
 if args.allocation_only:
  alloc=[]
  for name,cat,layer,fs,geo in cases:
   for label,fn in fs.items():
    for _ in range(10):fn()
    sync()
    if label.startswith('PyTorch'):
     torch.cuda.memory._record_memory_history(enabled='all',context=None,max_entries=10000,clear_history=True)
     try:
      out=fn();sync()
      events=torch.cuda.memory._snapshot()['device_traces'][0]
      alloc.append(dict(case=name,implementation=label,allocator='torch allocator history',allocation_requests=sum(e['action']=='alloc' for e in events),allocator_reported_bytes=sum(e.get('size',0) for e in events if e['action']=='alloc'),free_events=sum(e['action']=='free_requested' for e in events),copy_count_source='OPERATOR_KERNELS and raw SQLite memcpy'))
     finally:torch.cuda.memory._record_memory_history(enabled=None)
    else:
     hook=Alloc()
     with hook:out=fn();sync()
     alloc.append(dict(case=name,implementation=label,allocator='CuPy MemoryHook',allocation_requests=hook.calls,allocator_reported_bytes=hook.bytes,free_events='',copy_count_source='OPERATOR_KERNELS and raw SQLite memcpy'))
  write(O/'OPERATOR_ALLOCATIONS.csv',alloc)
  sys.exit(0)
 # All checks and warmups outside trace capture.
 for name,cat,layer,fs,geo in cases:
  vals={}
  for label,fn in fs.items():
   if 'reset'in geo:geo['reset']()
   vals[label]=[host(v).copy() for v in outputs(fn())]
  ref=vals['PyTorch']
  for label,vs in vals.items():
   for i,(v,r) in enumerate(zip(vs,ref)):
    er=error(v,r);errs.append(dict(case=name,implementation=label,output=i,**er))
    # Preserve failures, do not relax the original gate to hide different cuBLAS orders.
  for fn in fs.values():
   for _ in range(10):fn()
  sync()
  for label,fn in fs.items():
   hook=Alloc()
   if label.startswith('PyTorch'):
    torch.cuda.reset_peak_memory_stats();base=torch.cuda.memory_allocated();out=fn();sync();peak=torch.cuda.max_memory_allocated()-base;calls='';byte=''
   else:
    with hook:out=fn();sync()
    peak='';calls=hook.calls;byte=hook.bytes
   mem.append(dict(case=name,implementation=label,cupy_malloc_calls=calls,cupy_requested_bytes=byte,torch_extra_peak_bytes=peak))
 if args.trace:cp.cuda.runtime.profilerStart()
 for name,cat,layer,fs,geo in cases:
  labels=list(fs)
  for rep in range(args.trace_repeats if args.trace else 15):
   for label in (labels if rep%2==0 else labels[::-1]):
    fn=fs[label]
    if args.trace:
     with phase('op|'+name+'|'+label,True):out=fn()
     sync();continue
    # Each metric measured independently, no events in CPU/fullcall measurements.
    sync();t=time.perf_counter_ns();out=fn();submit=(time.perf_counter_ns()-t)/1000;sync()
    sync();t=time.perf_counter_ns();out=fn();sync();full=(time.perf_counter_ns()-t)/1000
    start,end=cp.cuda.Event(),cp.cuda.Event();start.record();out=fn();end.record();end.synchronize();ev=cp.cuda.get_elapsed_time(start,end)*1000
    raw.append(dict(case=name,category=cat,layer=layer,implementation=label,repeat=rep,submit_us=submit,sync_us=full,event_envelope_us=ev,geometry=json.dumps({k:v for k,v in geo.items() if k!='reset'})))
  print(name,flush=True)
 if args.trace:cp.cuda.runtime.profilerStop()
 else:
  write(O/f'stage_a/OPERATOR_RAW_{args.tag}.csv',raw);write(O/f'stage_a/OPERATOR_ERRORS_{args.tag}.csv',errs);write(O/f'stage_a/OPERATOR_ALLOCATIONS_{args.tag}.csv',mem)
