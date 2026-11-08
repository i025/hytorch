import argparse,json,csv,statistics
from pathlib import Path
from common import cp,np,run,cases,make,FC
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--profile',action='store_true');a=p.parse_args();O=Path(a.output);O.mkdir(parents=True,exist_ok=False);rows=[]
if a.profile:cp.cuda.runtime.profilerStart()
for i,case in enumerate(cases()):
 col=make(case,501+i);before=cp.asnumpy(col);outs={m:cp.empty(case['shape'],cp.float32) for m in ['cupy','native']}
 for mode in outs:
  for _ in range(10):run(mode,col,case['shape'],case['stride'],case['pad'],outs[mode])
 cp.cuda.get_current_stream().synchronize()
 graphs={};stream=cp.cuda.Stream(non_blocking=True);batch=32
 for mode in outs:
  with stream:
   stream.begin_capture()
   for _ in range(batch):run(mode,col,case['shape'],case['stride'],case['pad'],outs[mode])
   graphs[mode]=stream.end_capture()
  dot=graphs[mode].debug_dot_str(flags=1)
  (O/f'{i}_{mode}.dot').write_text(dot)
  if mode=='native':assert 'col2im_strided' in dot
  graphs[mode].upload();graphs[mode].launch()
 cp.cuda.get_current_stream().synchronize();times={m:[] for m in outs};start,end=cp.cuda.Event(),cp.cuda.Event()
 for rep in range(3 if a.profile else 15):
  for mode in (list(outs) if rep%2==0 else list(outs)[::-1]):
   cp.cuda.nvtx.RangePush(f'colbench:{i}:{mode}:{rep}')
   if a.profile:run(mode,col,case['shape'],case['stride'],case['pad'],outs[mode]);cp.cuda.get_current_stream().synchronize()
   else:
    start.record();graphs[mode].launch();end.record();end.synchronize();times[mode].append(float(cp.cuda.get_elapsed_time(start,end))*1000/batch)
   cp.cuda.nvtx.RangePop()
 actual={m:cp.asnumpy(v) for m,v in outs.items()};np.testing.assert_array_equal(actual['native'],actual['cupy']);np.testing.assert_array_equal(cp.asnumpy(col),before)
 # Verify the AST-extracted preallocated path equals the actual unchanged function.
 sh,sw=case['stride'];ph,pw=case['pad'];h,w=case['shape'][2:]
 np.testing.assert_array_equal(actual['cupy'],cp.asnumpy(FC._col2im_gpu(col,sh,sw,ph,pw,h,w)))
 for mode in outs:
  rows.append(dict(case_id=i,**case,input_shape=json.dumps(col.shape),strides_elements=json.dumps([x//4 for x in col.strides]),dtype='float32',implementation=mode,median_us=statistics.median(times[mode]) if times[mode] else '',mean_us=statistics.mean(times[mode]) if times[mode] else '',stdev_us=statistics.stdev(times[mode]) if times[mode] else '',speedup_vs_cupy=statistics.median(times['cupy'])/statistics.median(times[mode]) if times[mode] else '',repeats=json.dumps(times[mode]),batch=32,bitwise_equal=True,max_abs_error=0))
 print(i,case['name'],case['layout'],{m:round(statistics.median(v),3) for m,v in times.items() if v},flush=True)
if a.profile:cp.cuda.runtime.profilerStop()
with (O/'benchmark.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(O/'method.json').write_text(json.dumps(dict(seed='501+case_id',warmup=10,repeats=15,graph_batch=32,profile_repeats=3,original_source='ElementwiseKernel AST expression from unchanged functions_conv._col2im_gpu; output preallocated; equality vs full production function checked every case',timing='unprofiled event around graph replay /32, excluding H2D/D2H; profile uses independent direct calls',cases=len(cases())),indent=2))
