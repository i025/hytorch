import torch
import time,csv,json
from bindings import *
prepare();rows=[]
for spec in cases()[::3][:5]:
 col=make(spec,42);out=cp.empty(spec['shape'],cp.float32);sync=cp.cuda.get_current_stream().synchronize
 funcs={m:lambda m=m:launch(m,col,spec['shape'],spec['stride'],spec['pad'],out) for m in ['old','cache','fast']}
 rt.col2im(col,spec['shape'],spec['stride'],spec['pad'],out)
 meta,packed=rt._col2im_metadata(spec['shape'],col.shape,col.strides,spec['stride'],spec['pad']);lib=rt.load_library()
 funcs.update(direct18=lambda:lib.hy_col2im(col.data.ptr,out.data.ptr,*meta,cp.cuda.get_current_stream().ptr),direct4=lambda:lib.hy_col2im_fast(col.data.ptr,out.data.ptr,packed,cp.cuda.get_current_stream().ptr))
 for f in funcs.values():
  for _ in range(20):f()
 vals={m:[] for m in funcs}
 for rep in range(31):
  for m in (list(funcs) if rep%2==0 else list(funcs)[::-1]):
   sync();t=time.perf_counter_ns()
   for _ in range(20):funcs[m]()
   vals[m].append((time.perf_counter_ns()-t)/20000)
 for m,v in vals.items():rows.append(dict(case=spec['name'],mode=m,median_us=float(np.median(v)),p10_us=float(np.percentile(v,10)),p90_us=float(np.percentile(v,90)),raw_us=json.dumps(v)))
 print(spec['name'],{m:np.median(v) for m,v in vals.items()},flush=True)
with (O/'raw/pilot.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
