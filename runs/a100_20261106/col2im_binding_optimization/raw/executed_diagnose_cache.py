import sys,time,csv,cProfile,pstats,json,ctypes as ct
from pathlib import Path
import torch
from common import *
O=Path('runs/a100_20261106/col2im_binding_optimization/raw/cache_only');O.mkdir(exist_ok=True); rows=[]
def measure(label,fn,case,count=20):
 for _ in range(20):fn()
 cp.cuda.get_current_stream().synchronize()
 vals=[]
 for rep in range(31):
  cp.cuda.get_current_stream().synchronize();t=time.perf_counter_ns()
  for _ in range(count):fn()
  vals.append((time.perf_counter_ns()-t)/count/1000)
 cp.cuda.get_current_stream().synchronize()
 rows.append(dict(case=case,component=label,median_us=float(np.median(vals)),p10_us=float(np.percentile(vals,10)),p90_us=float(np.percentile(vals,90)),raw_us=json.dumps(vals)))
for spec in cases()[::3][:5]:
 col=make(spec,42);shape=spec['shape'];stride=spec['stride'];pad=spec['pad'];out=cp.empty(shape,cp.float32)
 rt.col2im(col,shape,stride,pad,out);n,c,h,w=shape;_,_,kh,kw,oh,ow=col.shape
 metadata=(n,c,h,w,kh,kw,oh,ow,*stride,*pad,*(s//4 for s in col.strides));lib=rt.load_library();fn=lib.hy_col2im
 def direct():
  status=fn(col.data.ptr,out.data.ptr,*metadata,cp.cuda.get_current_stream().ptr)
  assert status==0
 def full():return rt.col2im(col,shape,stride,pad,out)
 def cupy():return run('cupy',col,shape,stride,pad,out)
 for label,fun in [('A_cupy',cupy),('B_full_native',full),('C_prevalidated_direct',direct),('current_stream',lambda:cp.cuda.get_current_stream().ptr),('ctypes_18_int64_construction',lambda:tuple(ct.c_int64(v) for v in metadata)),('backend_selection',lambda:rt.selected('col2im',col)),('dtype_device_alias',lambda:(col.dtype==cp.float32,col.device.id==cp.cuda.runtime.getDevice(),cp.may_share_memory(col,out))),('shape_strides_read',lambda:(col.shape,col.strides))]:
  with rt.backend(col2im='native'):measure(label,fun,spec['name'])
 prof=cProfile.Profile();prof.enable()
 for _ in range(300):full()
 prof.disable();cp.cuda.get_current_stream().synchronize();prof.dump_stats(str(O/(spec['name']+'.pstats')))
 with (O/(spec['name']+'_cpu_profile.txt')).open('w') as f:pstats.Stats(prof,stream=f).sort_stats('tottime').print_stats(35)
 print(spec['name'],[(r['component'],r['median_us']) for r in rows if r['case']==spec['name']],flush=True)
with (O/'diagnosis.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
