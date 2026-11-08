"""Five-level diagnostics. No prevalidated entry point is exposed in production."""
from common import *
import argparse,ctypes as ct,itertools,cProfile,pstats
p=argparse.ArgumentParser();p.add_argument('--trace',action='store_true');p.add_argument('--mode',default='B0');p.add_argument('--cpu',action='store_true');p.add_argument('--repeats',type=int,default=15);p.add_argument('--tag',default='');p.add_argument('--case-prefix',default='');a=p.parse_args()
rng=np.random.default_rng(42);cases=[];components=[]
def tensor(shape,positive=False):
 x=rng.normal(0,.1,shape).astype(np.float32)
 if positive:x=np.abs(x)+.01
 return cp.asarray(x),torch.tensor(x,device='cuda',requires_grad=True)
def add(name,cat,level,hy,pt,geo):cases.append(dict(name=name,category=cat,level=level,hy=hy,torch=pt,geometry=geo))
def checked(status):
 if status:raise RuntimeError('native status '+str(status))
lib=rt.load_library()
specs=[('conv1',(4,3,66,200),24,(5,5),(2,2),(0,0)),('conv2',(4,24,31,98),36,(5,5),(2,2),(0,0)),('conv3',(4,36,14,47),48,(5,5),(2,2),(0,0)),('conv4',(4,48,5,22),64,(3,3),(1,1),(0,0)),('conv5',(4,64,3,20),64,(3,3),(1,1),(0,0)),('irregular',(2,3,13,17),7,(3,5),(2,3),(1,2)),('asymmetric',(1,5,19,11),9,(5,1),(1,2),(2,0))]
def convcase(spec):
 name,shape,oc,k,s,p=spec;x,tx=tensor(shape);w,tw=tensor((oc,shape[1],*k));b,tb=tensor((oc,));vx,vw,vb=map(hytorch.Variable,(x,w,b))
 y=FC.conv2d(vx,vw,vb,s,p);ty=torch.nn.functional.conv2d(tx,tw,tb,s,p);g,tg=tensor(y.shape);tg=tg.detach();vg=hytorch.Variable(g)
 geo=dict(shape=shape,kernel=k,stride=s,pad=p,output=y.shape,gradients='x,w,b',frequency_per_donkey_step=1 if name.startswith('conv') else 0)
 dw=FC.Conv2DGradW(y.creator);di=FC.Deconv2d(s,p,shape[2:]);geom=rt._geometry(x,k,s,p);n,c,h,wi,kh,kw,sh,sw,ph,pw,oh,ow=geom
 # Prevalidate representative gcol layout. No numerical workspace reused between calls.
 testcol=cp.rollaxis(cp.tensordot(w,g,(0,1)),3)
 _,packed=rt._col2im_metadata(shape,testcol.shape,testcol.strides,s,p)
 assert not cp.may_share_memory(testcol,x)
 rt.col2im(testcol,shape,s,p);rt.packed_dweight(x,g,k,s,p);del testcol
 def direct():
  col=cp.rollaxis(cp.tensordot(w,g,(0,1)),3);gx=cp.empty(shape,cp.float32)
  checked(lib.hy_col2im_fast(col.data.ptr,gx.data.ptr,packed,cp.cuda.get_current_stream().ptr))
  xc=cp.empty((n*oh*ow,c*kh*kw),cp.float32)
  checked(lib.hy_im2col(x.data.ptr,xc.data.ptr,*geom,cp.cuda.get_current_stream().ptr))
  gm=g.transpose(1,0,2,3).reshape(oc,-1);gw=(gm@xc).reshape(w.shape)
  return gx,gw,g.sum(axis=(0,2,3))
 def raw():return di.forward(g,w,None),dw.forward(x,g),g.sum(axis=(0,2,3))
 def wrapped():
  with hytorch.no_grad():return y.creator.backward(vg)
 def back():
  for v in [vx,vw,vb]:v.cleargrad()
  y.grad=vg;y.backward();return vx.grad,vw.grad,vb.grad
 def tback():return torch.autograd.grad(ty,(tx,tw,tb),tg,retain_graph=True)
 def tdirect():return torch.ops.aten.convolution_backward(tg,tx,tw,[oc],list(s),list(p),[1,1],False,[0,0],1,[True,True,True])
 add(name+'/direct','ConvBackward','L2_prevalidated_diagnostic',direct,tdirect,geo)
 add(name+'/raw','ConvBackward','L2_checked_raw',raw,None,geo)
 add(name+'/function','ConvBackward','L3_backward_Functions',wrapped,None,geo)
 add(name+'/backward','ConvBackward','L4_autograd',back,tback,geo)
 add(name+'/dinput','dInput','L3_Function',lambda:FC.deconv2d(vg,vw,None,s,p,shape[2:]),None,geo)
 add(name+'/dweight','dWeight','L3_Function',lambda:FC.Conv2DGradW(y.creator)(vx,vg),None,geo)
 add(name+'/dbias','dBias','L3_Function',lambda:vg.sum(axis=(0,2,3)),None,geo)

def poolcase(name,shape,k,s,p):
 x,tx=tensor(shape,positive=True);vx=hytorch.Variable(x)
 # Positive inputs ensure native zero padding and cuDNN -inf padding select same winners.
 y=F.pooling(vx,k,s,p);ty=torch.nn.functional.max_pool2d(tx,k,s,p);g,tg=tensor(y.shape);tg=tg.detach();vg=hytorch.Variable(g)
 geo=dict(shape=shape,kernel=k,stride=s,pad=p,output=y.shape,gradients='x',positive_input=True,frequency_per_donkey_step=0)
 geom=rt._geometry(x,k,s,p);out,idx=rt.max_pool2d(x,k,s,p)
 def direct():checked(lib.hy_pool(x.data.ptr,out.data.ptr,idx.data.ptr,*geom,cp.cuda.get_current_stream().ptr));return out
 def raw():return rt.max_pool2d(x,k,s,p)[0]
 def back():vx.cleargrad();y.grad=vg;y.backward();return vx.grad
 add(name+'/direct','PoolForward','L2_prevalidated_preallocated_diagnostic',direct,None,geo)
 add(name+'/raw','PoolForward','L2_checked_raw',raw,None,geo)
 add(name+'/forward','PoolForward','L3_Function',lambda:F.pooling(vx,k,s,p),lambda:torch.nn.functional.max_pool2d(tx,k,s,p),geo)
 add(name+'/backward','PoolBackward','L4_autograd',back,lambda:torch.autograd.grad(ty,tx,tg,retain_graph=True)[0],geo)
 components.extend([(name,'backend_selection',lambda:rt.selected('pool',x)),(name,'arrays_validation',lambda:rt._arrays(x)),(name,'geometry_validation',lambda:rt._geometry(x,k,s,p)),(name,'current_stream',lambda:cp.cuda.get_current_stream().ptr),(name,'pointer_construction',lambda:rt._ptr(x)),(name,'output_allocation_pair',lambda:(cp.empty(y.shape,cp.float32),cp.empty(y.shape,cp.int64)))])
with config('B0'):
 for sp in specs:convcase(sp)
 for sp in [('tiny_actual_pool',(4,4,16,16),2,2,0),('tiny_pool',(4,3,16,16),2,2,0),('irregular_pool',(2,7,13,17),2,2,0),('overlap_pool',(2,5,11,19),3,1,1),('wide_pool',(1,3,9,31),(2,3),(2,1),(0,1))]:poolcase(*sp)
if a.case_prefix:cases=[c for c in cases if c['name'].startswith(a.case_prefix)]
raw=[];errs=[];alloc=[];componentrows=[]
def outputs(v):return list(v) if isinstance(v,(tuple,list)) else [v]
class Alloc(cp.cuda.MemoryHook):
 def __init__(self):self.calls=0;self.bytes=0
 def malloc_postprocess(self,**kw):self.calls+=1;self.bytes+=kw.get('size',0)
# Profiling and component diagnostics are explicitly separated from formal timing.
if a.cpu:
 with config('B0'):
  for case in cases:
   if case['level'] not in ['L4_autograd','L3_Function']:continue
   for _ in range(10):case['hy']()
  sync();pr=cProfile.Profile();pr.enable()
  for _ in range(30):
   for case in cases:
    if case['level'] in ['L4_autograd','L3_Function']:case['hy']()
  sync();pr.disable()
  pr.dump_stats(str(O/'raw/layer_cpu.prof'))
  with (O/'raw/layer_cpu.txt').open('w') as f:pstats.Stats(pr,stream=f).strip_dirs().sort_stats('cumulative').print_stats(100)
  for name,component,fn in components:
   for _ in range(50):fn()
   for rep in range(15):
    t=time.perf_counter_ns()
    for _ in range(1000):fn()
    componentrows.append(dict(case=name,component=component,repeat=rep,ns_per_call=(time.perf_counter_ns()-t)/1000))
 write(O/'raw/CPU_COMPONENTS.csv',componentrows);sys.exit(0)
for case in cases:
 modes=['B0','O1']+(['torch'] if case['torch'] else [])
 vals={}
 for mode in modes:
  fn=case['torch'] if mode=='torch' else case['hy']
  with config(mode):
   vals[mode]=[host(v).copy() for v in outputs(fn())]
   for _ in range(15):fn()
   sync()
   hook=Alloc()
   if mode=='torch':
    torch.cuda.reset_peak_memory_stats();base=torch.cuda.memory_allocated()
    torch.cuda.memory._record_memory_history(enabled='all',context=None,max_entries=10000,clear_history=True)
    out=fn();sync();events=torch.cuda.memory._snapshot()['device_traces'][0];torch.cuda.memory._record_memory_history(enabled=None)
    calls=sum(e['action']=='alloc' for e in events);byte=sum(e.get('size',0) for e in events if e['action']=='alloc');peak=torch.cuda.max_memory_allocated()-base
   else:
    with hook:out=fn();sync()
    calls,byte,peak=hook.calls,hook.bytes,''
   alloc.append(dict(case=case['name'],implementation=mode,allocation_requests=calls,requested_bytes=byte,torch_incremental_peak_bytes=peak))
 for mode,values in vals.items():
  for i,(v,b) in enumerate(zip(values,vals['B0'])):
   er=error(v,b);errs.append(dict(case=case['name'],implementation=mode,output=i,**er));assert mode!='O1' or er['bitwise_equal'],errs[-1]
# Compare raw and direct diagnostic numerical outputs to full backward in addition to O1.
for prefix in [sp[0] for sp in specs]:
 selected=[c for c in cases if c['name'].startswith(prefix+'/') and c['level'].startswith(('L2','L4'))]
 with config('B0'):
  if not selected:continue
  ref=[host(v).copy() for v in outputs(selected[-1]['hy']())]
  for case in selected:
   for i,(v,b) in enumerate(zip(outputs(case['hy']()),ref)):
    er=error(v,b);errs.append(dict(case=case['name'],implementation='direct_vs_full',output=i,**er));assert not er['failed'],errs[-1]
if a.trace:cp.cuda.runtime.profilerStart()
for case in cases:
 modes=[a.mode] if a.trace else ['B0','O1']+(['torch'] if case['torch'] else [])
 if a.trace and a.mode=='torch' and not case['torch']:continue
 orders=list(itertools.permutations(modes))
 for rep in range(a.repeats):
  for mode in orders[rep%len(orders)]:
   fn=case['torch'] if mode=='torch' else case['hy']
   with config(mode):
    if a.trace:
     with phase('op|'+case['name']+'|'+mode,True):out=fn()
     sync();continue
    sync();t=time.perf_counter_ns();out=fn();submit=(time.perf_counter_ns()-t)/1000;sync()
    t=time.perf_counter_ns();out=fn();sync();full=(time.perf_counter_ns()-t)/1000
    raw.append(dict(case=case['name'],category=case['category'],level=case['level'],implementation=mode,repeat=rep,submit_us=submit,sync_us=full,geometry=json.dumps(case['geometry'])))
 print(case['name'],flush=True)
if a.trace:cp.cuda.runtime.profilerStop()
else:
 suffix=('_'+a.tag) if a.tag else ''
 write(O/('raw/LAYER_TIMINGS'+suffix+'.csv'),raw);write(O/('raw/LAYER_ERRORS'+suffix+'.csv'),errs);write(O/('raw/LAYER_ALLOCATIONS'+suffix+'.csv'),alloc)
