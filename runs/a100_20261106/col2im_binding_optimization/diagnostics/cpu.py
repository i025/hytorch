import torch
import time,cProfile,pstats,csv,json,operator
from bindings import *
prepare();rows=[];raw=[];sync=cp.cuda.get_current_stream().synchronize
for spec in cases()[::3][:5]:
 col=make(spec,42);shape=spec['shape'];stride=spec['stride'];pad=spec['pad'];out=cp.empty(shape,cp.float32)
 rt.col2im(col,shape,stride,pad,out);meta,packed=rt._col2im_metadata(shape,col.shape,col.strides,stride,pad);lib=rt.load_library()
 funcs={m:lambda m=m:launch(m,col,shape,stride,pad,out) for m in ['cupy','old','fast']}
 funcs.update(direct18=lambda:lib.hy_col2im(col.data.ptr,out.data.ptr,*meta,cp.cuda.get_current_stream().ptr),direct4=lambda:lib.hy_col2im_fast(col.data.ptr,out.data.ptr,packed,cp.cuda.get_current_stream().ptr),stream=lambda:cp.cuda.get_current_stream().ptr,selection=lambda:rt.selected('col2im',col),dtype_device_alias=lambda:(col.dtype==cp.float32,col.device.id==cp.cuda.runtime.getDevice(),cp.may_share_memory(col,out)),normalize_metadata=lambda:(tuple(operator.index(v) for v in shape),tuple(operator.index(v) for v in stride),tuple(operator.index(v) for v in pad)),metadata_hit=lambda:rt._col2im_metadata(shape,col.shape,col.strides,stride,pad),metadata_validate_uncached=lambda:rt._col2im_metadata.__wrapped__(shape,col.shape,col.strides,stride,pad))
 vals={m:[] for m in funcs}
 with rt.backend(col2im='native'):
  for f in funcs.values():
   for _ in range(30):f()
  for rep in range(31):
   modes=list(funcs);modes=modes[rep%len(modes):]+modes[:rep%len(modes)]
   for m in modes:
    sync();t=time.perf_counter_ns()
    for _ in range(20):funcs[m]()
    v=(time.perf_counter_ns()-t)/20000;vals[m].append(v);raw.append(dict(case=spec['name'],mode=m,repeat=rep,us=v))
  for m in ['cupy','old','fast']:
   prof=cProfile.Profile();prof.enable()
   for _ in range(300):funcs[m]()
   prof.disable();sync()
   prof.dump_stats(str(O/'raw'/f'{spec["name"]}_{m}.pstats'))
   with (O/'raw'/f'{spec["name"]}_{m}_cpu_profile.txt').open('w') as f:pstats.Stats(prof,stream=f).sort_stats('tottime').print_stats(45)
 # Top-level selection/reshape/allocation path, outside cProfile.
 for mode in ['cupy','old','fast']:
  with mode_context(mode):
   for _ in range(20):FC.col2im_array(col,shape,spec['kernel'],stride,pad,False)
   vv=[]
   for rep in range(31):
    sync();t=time.perf_counter_ns()
    for _ in range(20):FC.col2im_array(col,shape,spec['kernel'],stride,pad,False)
    vv.append((time.perf_counter_ns()-t)/20000)
   vals['col2im_array_'+mode]=vv
 for m,v in vals.items():rows.append(dict(case=spec['name'],component=m,scope='isolated_cpu_submit_elapsed_not_additive',median_us=float(np.median(v)),p10_us=float(np.percentile(v,10)),p90_us=float(np.percentile(v,90)),raw_us=json.dumps(v)))
 print(spec['name'],{m:round(np.median(v),3) for m,v in vals.items()},flush=True)
for path,data in [(O/'CPU_OVERHEAD.csv',rows),(O/'raw/cpu_repeats.csv',raw)]:
 with path.open('w') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
