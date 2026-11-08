from common import *
import statistics
rng=np.random.default_rng(63);cases=[]
def setup(name,shape,oc,k,s,p):
 x=hytorch.Variable(cp.asarray(rng.normal(0,.1,shape).astype(np.float32)));w=hytorch.Variable(cp.asarray(rng.normal(0,.1,(oc,shape[1],*k)).astype(np.float32)));b=hytorch.Variable(cp.zeros(oc,cp.float32))
 with config():y=FC.conv2d(x,w,b,s,p)
 g=hytorch.Variable(cp.asarray(rng.normal(0,.1,y.shape).astype(np.float32)));dw=FC.Conv2DGradW(y.creator)
 cases.append((name+'/forward',lambda:FC.conv2d(x,w,b,s,p)))
 cases.append((name+'/dweight',lambda:dw(x,g)))
specs=[('conv1',(4,3,66,200),24,(5,5),(2,2),(0,0)),('conv2',(4,24,31,98),36,(5,5),(2,2),(0,0)),('conv3',(4,36,14,47),48,(5,5),(2,2),(0,0)),('conv4',(4,48,5,22),64,(3,3),(1,1),(0,0)),('conv5',(4,64,3,20),64,(3,3),(1,1),(0,0)),('irregular',(2,3,13,17),7,(3,5),(2,3),(1,2))]
for sp in specs:setup(*sp)
def gemcase(m,n,k):
 a=cp.asarray(rng.normal(size=(m,k)).astype(np.float32));b=cp.asarray(rng.normal(size=(k,n)).astype(np.float32));out=cp.empty((m,n),cp.float32)
 cases.append((f'gemm/{m}/{n}/{k}',lambda:rt.gemm(a,b,out=out)))
for dims in [(4,100,1152),(127,257,65),(1024,1024,1024)]:gemcase(*dims)
rows=[];errs=[]
for name,fn in cases:
 for mode in ['B0','F1']:
  with config(mode):
   for _ in range(20):fn()
   arr=host(fn()).copy()
  if mode=='B0':ref=arr
  else:
   np.testing.assert_array_equal(arr,ref);errs.append(dict(case=name,exact_value_equal=True))
 for rep in range(15):
  for mode in (['B0','F1'] if rep%2==0 else ['F1','B0']):
   with config(mode):
    sync();t=time.perf_counter_ns();z=fn();submit=(time.perf_counter_ns()-t)/1000;sync()
    t=time.perf_counter_ns();z=fn();sync();full=(time.perf_counter_ns()-t)/1000
    s,e=cp.cuda.Event(),cp.cuda.Event();s.record();z=fn();e.record();e.synchronize();ev=cp.cuda.get_elapsed_time(s,e)*1000
   rows.append(dict(case=name,mode=mode,repeat=rep,submit_us=submit,sync_us=full,event_envelope_us=ev))
write(O/'FRAMEWORK_RAW.csv',rows);write(O/'FRAMEWORK_ERRORS.csv',errs)
out=[]
for case in dict.fromkeys(r['case'] for r in rows):
 for mode in ['B0','F1']:
  rr=[r for r in rows if r['case']==case and r['mode']==mode];d=dict(case=case,mode=mode,n=len(rr))
  for c in ['submit_us','sync_us','event_envelope_us']:d[c+'_median']=statistics.median(r[c] for r in rr);d[c+'_sd']=statistics.stdev(r[c] for r in rr)
  out.append(d)
write(O/'FRAMEWORK_COMPARISON.csv',out)
# Pure host validation timing is diagnostic, not the claimed full-call improvement.
meta=[]
for rep in range(15):
 for mode in (['B0','F1'] if rep%2==0 else ['F1','B0']):
  with config(mode):
   t=time.perf_counter_ns()
   for _ in range(1000):rt._shape_ok((12152,75))
   elapsed=(time.perf_counter_ns()-t)/1000/1000
  meta.append(dict(repeat=rep,mode=mode,us_per_check=elapsed))
write(O/'raw/METADATA_CPU.csv',meta)
print(json.dumps(out,indent=2))
