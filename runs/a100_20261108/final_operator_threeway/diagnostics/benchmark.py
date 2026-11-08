from common import *
import argparse,ctypes,statistics,subprocess
p=argparse.ArgumentParser();p.add_argument('--trace',action='store_true');p.add_argument('--smoke',action='store_true');p.add_argument('--ops',default='');p.add_argument('--prefix',default='');a=p.parse_args()
stream=cp.cuda.get_current_stream();sync=stream.synchronize
libs={'native':rt.load_library()}
for mode,path in [('shared_old','runs/a100_20261106/libhytorch_cuda_baseline.so'),('register_old','runs/a100_20261106/gemm_shape_dispatch/lib_before.so')]:
 lib=ctypes.CDLL(str(R/path));lib.hy_gemm.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*5+[ctypes.c_void_p];lib.hy_gemm.restype=ctypes.c_int;lib.hy_cuda_error.argtypes=[ctypes.c_int];lib.hy_cuda_error.restype=ctypes.c_char_p;libs[mode]=lib
raw=[];errors=[];summaries=[];allocs=[];unsupported=[]
class Allocation(cp.cuda.MemoryHook):
 def __init__(self):self.count=0;self.bytes=0
 def malloc_preprocess(self,**kw):self.count+=1;self.bytes+=kw.get('size',0)

def measure(fn,cpu,loops):
 if cpu:
  t=time.perf_counter_ns()
  for _ in range(loops):res=fn()
  return {'sync_us':(time.perf_counter_ns()-t)/loops/1000}
 sync();t=time.perf_counter_ns()
 for _ in range(loops):res=fn()
 submit=(time.perf_counter_ns()-t)/loops/1000;sync();wall=(time.perf_counter_ns()-t)/loops/1000
 # Separate run, event instrumentation does NOT pollute sync wall timing.
 ev1,ev2=cp.cuda.Event(),cp.cuda.Event();ev1.record()
 for _ in range(loops):res=fn()
 ev2.record();ev2.synchronize()
 return dict(sync_us=wall,submit_us=submit,event_us=cp.cuda.get_elapsed_time(ev1,ev2)*1000/loops)

cs=cases()
if a.ops:cs=[c for c in cs if c["op"] in a.ops.split(",")]
if a.smoke:cs=[cs[0]]+[c for c in cs if c['id'].startswith('tiny') or c['id'].startswith('negative_pool')]
if a.trace:cp.cuda.runtime.profilerStart()
for ci,c in enumerate(cs):
 print(f'{ci+1}/{len(cs)} {c["op"]} {c["id"]}',flush=True)
 d,g,ref=prepare(c);op=c['op'];base=['numpy','cupy','native'];scopes=['array','function']
 for scope in scopes:
  modes=base.copy()
  if op=='gemm' and scope=='array':modes+=['shared_old','register_old']
  if a.trace:modes=[m for m in modes if m!='numpy']
  funcs={};notes={};timeloops={};nreps={}
  for mode in modes:
   rt._library=libs.get(mode,libs['native'])
   fn,note=make_fn(c,mode,scope,d if mode=='numpy' else g);funcs[mode]=fn;notes[mode]=note
   with config(mode):
    for _ in range(4):result=fn()
    sync()
    if not a.trace:
     errors.append(dict(op=op,case=c['id'],scope=scope,mode=mode,**accuracy(result,ref)))
     t=time.perf_counter_ns();result=fn();sync();probe=(time.perf_counter_ns()-t)/1e9
     # Predeclared budget: 30 CPU seconds per mode/case/scope; no selective exclusion.
     target_reps=30 if len(modes)==5 else 18
     nreps[mode]=target_reps if mode!='numpy' or probe*target_reps<=30 else max(3,int(30/probe));timeloops[mode]=1 if probe>.001 else 5
     if mode!='numpy':
      hook=Allocation()
      with hook:result=fn()
      sync();allocs.append(dict(op=op,case=c['id'],scope=scope,mode=mode,requests=hook.count,requested_bytes=hook.bytes,note='warm pool allocation requests, not driver mallocs'))
  reps=15 if a.trace else max(nreps.values())
  # Rotations with direction alternation: each three-way mode appears six times at each position over 18 reps.
  for rep in range(reps):
   baseorder=modes[:] if (rep//len(modes))%2==0 else modes[::-1];order=baseorder[rep%len(modes):]+baseorder[:rep%len(modes)]
   for pos,mode in enumerate(order):
    if not a.trace and rep>=nreps[mode]:continue
    rt._library=libs.get(mode,libs['native'])
    with config(mode):
     if a.trace:
      sync();cp.cuda.nvtx.RangePush('|'.join([op,c['id'],scope,mode,str(rep)]));result=funcs[mode]();cp.cuda.nvtx.RangePop();sync()
     else:
      vals=measure(funcs[mode],mode=='numpy',timeloops[mode]);raw.append(dict(op=op,case=c['id'],scope=scope,mode=mode,repeat=rep,order=pos,loops=timeloops[mode],stream_ptr=stream.ptr,preallocated_output=(op=='gemm' and scope=='array'),temporary_allocations_included=True,interface=notes[mode],**vals))
  if not a.trace:
   for mode in modes:
    rr=[r for r in raw if r['op']==op and r['case']==c['id'] and r['scope']==scope and r['mode']==mode]
    for metric in ('sync_us','submit_us','event_us'):
     v=[r[metric] for r in rr if metric in r]
     if v:summaries.append(dict(op=op,case=c['id'],scope=scope,mode=mode,metric=metric,mean=statistics.mean(v),median=statistics.median(v),std=statistics.stdev(v) if len(v)>1 else 0,min=min(v),max=max(v),repeats=len(v),loops=timeloops[mode],interface=notes[mode]))
  rt._library=libs['native']
 if not a.trace:
  prefix='raw/smoke_' if a.smoke else a.prefix
  write(prefix+'RAW_TIMINGS.csv',raw);write(prefix+'ACCURACY.csv',errors);write(prefix+'RESULTS_ALL.csv',summaries);write(prefix+'ALLOCATIONS.csv',allocs)
 del d,g,ref,funcs,result;gc.collect();sync()
if a.trace:cp.cuda.runtime.profilerStop()
else:
 for op,name in [('gemm','GEMM'),('im2col','IM2COL'),('col2im','COL2IM'),('conv','CONV_FORWARD'),('dweight','DWEIGHT'),('pool_forward','POOL'),('pool_backward','POOL')]:
  if name=='POOL':sel=[r for r in summaries if r['op'].startswith('pool')]
  else:sel=[r for r in summaries if r['op']==op]
  write(('raw/smoke_' if a.smoke else a.prefix)+name+'_RESULTS.csv',sel)
print('FINISHED',flush=True)
