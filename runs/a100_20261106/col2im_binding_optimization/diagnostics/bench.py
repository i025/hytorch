import torch
import time,csv,json,argparse
from bindings import *
p=argparse.ArgumentParser();p.add_argument('--profile',action='store_true');a=p.parse_args();prepare();rows=[];raw=[]
stream=cp.cuda.Stream(non_blocking=True);stream.use();sync=stream.synchronize
for ci,spec in enumerate(cases()):
 col=make(spec,501+ci);out={m:cp.empty(spec['shape'],cp.float32) for m in ['cupy','old','fast']}
 funcs={m:lambda m=m:launch(m,col,spec['shape'],spec['stride'],spec['pad'],out[m]) for m in out}
 for f in funcs.values():
  for _ in range(20):f()
 sync()
 for m in out:np.testing.assert_array_equal(cp.asnumpy(out[m]),cp.asnumpy(out['cupy']))
 if a.profile:
  if ci==0:cp.cuda.runtime.profilerStart()
  for rep in range(3):
   for m,f in funcs.items():
    cp.cuda.nvtx.RangePush(f'colbench:{ci}:{m}:{rep}');f();cp.cuda.nvtx.RangePop()
  sync();continue
 graphs={}
 for m,f in funcs.items():
  stream.begin_capture()
  for _ in range(32):f()
  graphs[m]=stream.end_capture()
 values={(m,metric):[] for m in funcs for metric in ['gpu_us','single_submit_us','five_submit_us','sync_single_us']}
 for rep in range(15):
  modes=list(funcs);modes=modes[rep%3:]+modes[:rep%3]
  if (rep//3)%2:modes.reverse()
  for m in modes:
   f=funcs[m];start=cp.cuda.Event();end=cp.cuda.Event();sync()
   start.record();graphs[m].launch();end.record();end.synchronize();gpu=cp.cuda.get_elapsed_time(start,end)*1000/32
   sync();t=time.perf_counter_ns();f();single=(time.perf_counter_ns()-t)/1000;sync()
   t=time.perf_counter_ns()
   for _ in range(5):f()
   five=(time.perf_counter_ns()-t)/1000;sync()
   t=time.perf_counter_ns();f();sync();full=(time.perf_counter_ns()-t)/1000
   for metric,v in zip(['gpu_us','single_submit_us','five_submit_us','sync_single_us'],[gpu,single,five,full]):values[(m,metric)].append(v)
   raw.append(dict(case_id=ci,repeat=rep,mode=m,gpu_us=gpu,single_submit_us=single,five_submit_us=five,sync_single_us=full))
 for m in funcs:
  row=dict(case_id=ci,name=spec['name'],layout=spec['layout'],mode=m,shape=spec['shape'],col_shape=col.shape,byte_strides=col.strides,stride=spec['stride'],pad=spec['pad'],max_abs_vs_cupy=0)
  for metric in ['gpu_us','single_submit_us','five_submit_us','sync_single_us']:
   v=values[(m,metric)];row.update({metric+'_median':float(np.median(v)),metric+'_p10':float(np.percentile(v,10)),metric+'_p90':float(np.percentile(v,90))})
  rows.append(row)
 print(ci,spec['name'],spec['layout'],[(m,round(np.median(values[(m,'single_submit_us')]),2)) for m in funcs],flush=True)
if a.profile:cp.cuda.runtime.profilerStop()
else:
 for path,data in [(O/'COL2IM_COMPARISON.csv',rows),(O/'raw/bench_repeats.csv',raw)]:
  with path.open('w') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
