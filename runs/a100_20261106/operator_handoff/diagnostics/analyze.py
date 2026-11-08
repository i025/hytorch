"""Join actual CUDA launch correlations to nested NVTX ranges; never time Python as kernels."""
from pathlib import Path
import sqlite3,json,csv,collections,statistics,hashlib,argparse
ROOT=Path.cwd()
p=argparse.ArgumentParser();p.add_argument('--handoff',default='runs/a100_20261106/operator_handoff');p.add_argument('--traces',default='runs/a100_20261106/operator_handoff_traces');args=p.parse_args()
OUT=ROOT/args.handoff;TRACES=ROOT/args.traces
def writecsv(path,rows):
 if not rows:return
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def clean(x):
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items() if k!='ptr'}
 if isinstance(x,list):return [clean(v) for v in x]
 return x

def has_array(x):
 return (isinstance(x,dict) and 'shape' in x) or (isinstance(x,list) and any(has_array(v) for v in x))

def ancestors(r,records):
 result=[]
 while r is not None:
  result.append(r);r=records.get(r['parent_id'])
 return result

def category(phase,chain):
 names=[r['operator'] for r in chain]
 if phase=='forward':return 'conv_forward' if 'Conv2d.forward' in names else 'other_forward'
 if phase=='update':return 'SGD_update'
 if phase=='backward':
  for op,cat in [('Deconv2d.forward','conv_dInput'),('Conv2DGradW.forward','conv_dWeight'),('Conv2d.backward','conv_dBias'),('Pooling2DGrad.forward','pool_backward'),('ReLU.backward','activation_backward'),('Tanh.backward','activation_backward'),('MatMul.backward','linear_backward'),('Add.backward','linear_bias_backward')]:
   if op in names:return cat
  return 'other_backward'
 return phase

def subtotal_name(name,chain):
 names=[r['operator'] for r in chain]
 if name.startswith('cupy_copy__float32_float64'):return 'cast_FP32_to_FP64'
 if name.startswith('cupy_copy'):return 'copy_layout_or_dtype'
 if 'col2im'==name:return 'col2im'
 if 'im2col' in name:return 'im2col'
 if 'gemm_' in name and 'const float *' in name:return 'native_GEMM'
 if 'cupy.tensordot' in names or 'MatMul.forward' in names:return 'CuPy_BLAS_compute'
 if 'cupy_sum' in name or 'reduce' in name:return 'reduction'
 return 'elementwise_or_other'

work=json.loads((OUT/'raw/workload/operators.json').read_text());wr={r['id']:r for r in work}
mem=json.loads((OUT/'raw/workload/memory.json').read_text());workrows=[]
for r in work:
 chain=ancestors(r,wr);phase=r['phase'];cat=category(phase,chain)
 allocations=[m for m in mem if m['kind']=='pool_request' and m['parent_id']==r['id']]
 g=r.get('gemm',{}).copy()
 if r['operator']=='MatMul.forward' and r.get('attrs',{}).get('native'):
  g['kernel']='native dispatch (nested native.gemm is the actual GEMM; do not double-count FLOPs)'
 backend='CuPy generated kernels/cuBLAS' if phase!='forward' else 'CuPy elementwise/shape operations'
 if r['operator'].startswith('native.'):backend='native CUDA ctypes; CuPy owns allocations/packing'
 if r['operator']=='Conv2d.forward':backend='native CUDA im2col+shared GEMM+pack'
 if r['operator']=='MatMul.forward':backend='native CUDA' if r.get('attrs',{}).get('native') else 'CuPy ndarray.dot/cuBLAS'
 needs='No requires_grad pruning; backward computes all input gradients'
 if cat=='conv_dInput':needs='Computed; conv1 leaf input gradient not used by SGD, higher-layer dInput required for earlier weights'
 if cat=='conv_dWeight':needs='Weight gradient required by SGD'
 if cat=='conv_dBias':needs='Bias gradient required by SGD'
 workrows.append(dict(invocation_id=r['id'],operator=r['operator'],phase=phase,category=cat,layer=r['layer'],calls_per_step=1,
  inputs=json.dumps(clean(r['input'])),outputs=json.dumps(clean(r['output'])),original_forward_inputs=json.dumps(clean(r.get('original_inputs',[]))),input_has_creator=json.dumps(r.get('input_has_creator',[])),parameters=json.dumps(r.get('attrs',{})),kwargs=json.dumps(clean(r['kwargs'])),
  gemm_M=g.get('M',''),gemm_N=g.get('N',''),gemm_K=g.get('K',''),gemm_details=json.dumps(g),actual_backend=backend,
  direct_pool_requests=len(allocations),direct_requested_bytes=sum(m['size'] for m in allocations),
  allocation='new logical arrays/requests; caching pool reuse, no device allocations observed after warmup',stream=r['stream_ptr'],gradient_requirement=needs,
  source=str(r['source']).replace(str(ROOT)+'/', ''),line=r['line'],parent_id=r['parent_id']))
writecsv(OUT/'WORKLOAD.csv',workrows)
# Preserve one-step metadata and aggregate repeated shape/signature frequencies.
grouped=collections.Counter()
for r in workrows:
 key=tuple((k,str(v)) for k,v in r.items() if k not in ['invocation_id','calls_per_step','parent_id','direct_pool_requests','direct_requested_bytes'])
 grouped[key]+=1
writecsv(OUT/'raw/workload_grouped.csv',[dict(key,calls_per_step=count) for key,count in grouped.items()])
summary={};profile=[];reference_kernels=None;reference_ops=None
for trace in ['native_trace','native_plain','cupy_trace']:
 c=sqlite3.connect(TRACES/(trace+'.sqlite'));c.row_factory=sqlite3.Row
 strings={r['id']:r['value'] for r in c.execute('select * from StringIds')}
 ranges=[]
 for row in c.execute('select * from NVTX_EVENTS where end is not null'):
  r=dict(row);r['label']=r['text'] or strings.get(r['textId'],'');ranges.append(r)
 ops={r['id']:r for r in json.loads((OUT/'raw'/trace/'operators.json').read_text())}
 api=[dict(r) for r in c.execute('select * from CUPTI_ACTIVITY_KIND_RUNTIME')]
 apis={r['correlationId']:r for r in api}
 phases=[r for r in ranges if r['label'].startswith('phase:')]
 opranges=[r for r in ranges if r['label'].startswith('op_')]
 steps=[r for r in ranges if r['label'].startswith('step:')];nsteps=len(steps)
 def locate(ar):
  containing=[r for r in phases if r['globalTid']==ar['globalTid'] and r['start']<=ar['start'] and r['end']>=ar['end']]
  phase=min(containing,key=lambda r:r['end']-r['start'])['label'][6:] if containing else 'outside'
  containing=[r for r in opranges if r['globalTid']==ar['globalTid'] and r['start']<=ar['start'] and r['end']>=ar['end']]
  rec=ops.get(int(min(containing,key=lambda r:r['end']-r['start'])['label'][3:])) if containing else None
  return phase,rec
 kernels=[]
 for raw in c.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL order by start'):
  k=dict(raw);ar=apis.get(k['correlationId']);phase,rec=locate(ar) if ar else ('uncorrelated',None)
  name=strings[k['demangledName']]
  if trace=='native_plain':
   ref=reference_kernels[len(kernels)]
   assert name==ref['name'] and phase==ref['phase']
   assert all(k[z]==ref[z] for z in ['gridX','gridY','gridZ','blockX','blockY','blockZ','registersPerThread','staticSharedMemory','dynamicSharedMemory'])
   rec=reference_ops.get(ref['operator_id'])
  chain=ancestors(rec,reference_ops if trace=='native_plain' else ops) if rec else []
  shape_rec=next((r for r in chain if has_array(r['input'])),rec)
  k.update(name=name,phase=phase,category=category(phase,chain),subtype=subtotal_name(name,chain),
           operator=rec['operator'] if rec else '',operator_id=rec['id'] if rec else None,layer=rec['layer'] if rec else '',
           input=json.dumps(clean(shape_rec['input'])) if shape_rec else '',duration_us=(k['end']-k['start'])/1000,
           api_us=(ar['end']-ar['start'])/1000 if ar else None,
           launch_to_start_us=(k['start']-ar['start'])/1000 if ar else None)
  kernels.append(k)
 total=sum(k['duration_us'] for k in kernels)
 if trace=='native_trace':reference_kernels=kernels;reference_ops=ops
 if trace=='native_plain':assert len(kernels)==len(reference_kernels)
 # Plain attribution is sequence-validated against the detailed trace; it is
 # not a directly observed per-operator NVTX range in the plain trace.
 writecsv(OUT/'raw'/(trace+'_kernels.csv'),kernels)
 groups=collections.defaultdict(list)
 for k in kernels:groups[(k['phase'],k['category'],k['layer'],k['subtype'],k['name'],k['gridX'],k['gridY'],k['gridZ'],k['blockX'],k['blockY'],k['blockZ'],k['registersPerThread'],k['staticSharedMemory'],k['dynamicSharedMemory'],k['input'])].append(k['duration_us'])
 kernelrows=[]
 for key,times in groups.items():
  ph,cat,layer,sub,name,gx,gy,gz,bx,by,bz,regs,ss,ds,shape=key
  row=dict(trace=trace,scope='kernel_by_shape',phase=ph,category=cat,layer=layer,name=name,subtype=sub,
    count=len(times),calls_per_step=len(times)/nsteps,mean_us=statistics.mean(times),median_us=statistics.median(times),total_us=sum(times),us_per_step=sum(times)/nsteps,
    percent_kernel_total=100*sum(times)/total,denominator='sum of kernel durations in this trace; not wall time',timing='Nsight Systems CUPTI kernel activity',
    grid=f'{gx},{gy},{gz}',block=f'{bx},{by},{bz}',registers_per_thread=regs,static_shared_bytes=ss,dynamic_shared_bytes=ds,input_shape_layout=shape)
  kernelrows.append(row);profile.append(row)
 writecsv(OUT/'raw'/(trace+'_kernel_summary.csv'),sorted(kernelrows,key=lambda r:-r['total_us']))
 coarse={};cats={}
 for field,target in [('phase',coarse),('category',cats)]:
  for name in sorted({k[field] for k in kernels}):
   vals=[k['duration_us'] for k in kernels if k[field]==name];target[name]=dict(count=len(vals),total_us=sum(vals),us_per_step=sum(vals)/nsteps,pct=100*sum(vals)/total)
   profile.append(dict(trace=trace,scope=field+'_kernel_sum',name=name,count=len(vals),calls_per_step=len(vals)/nsteps,total_us=sum(vals),us_per_step=sum(vals)/nsteps,mean_us=statistics.mean(vals),percent_kernel_total=100*sum(vals)/total,denominator='kernel sum; overlaps other aggregation levels, do not add levels',timing='Nsight Systems kernel durations grouped by launch NVTX range'))
 # Inclusive operator sums capture dInput/dWeight/dBias and suboperations without attributing CPU gaps to kernels.
 inclusive=collections.defaultdict(list)
 for k in kernels:
  if k['operator_id'] is not None and trace!='native_plain':
   for r in ancestors(ops[k['operator_id']],ops):inclusive[r['id']].append(k['duration_us'])
 opgroups=collections.defaultdict(list)
 for rr in opranges:
  rec=ops[int(rr['label'][3:])];opgroups[(rec['phase'],rec['operator'],rec['layer'],json.dumps(clean(rec['input'])))].append((rr,rec))
 op_rows=[]
 for (ph,operator,layer,shape),v in opgroups.items():
  gpu=[sum(inclusive.get(rec['id'],[])) for _,rec in v];cpu=[(rr['end']-rr['start'])/1000 for rr,_ in v]
  row=dict(trace=trace,scope='operator_inclusive',phase=ph,name=operator,layer=layer,count=len(v),calls_per_step=len(v)/nsteps,
           mean_us=statistics.mean(gpu),median_us=statistics.median(gpu),total_us=sum(gpu),us_per_step=sum(gpu)/nsteps,
           percent_kernel_total=sum(gpu)/total*100,cpu_inclusive_mean_us=statistics.mean(cpu),cpu_inclusive_total_us=sum(cpu),
           timing='sum of descendant CUPTI kernel durations; CPU column is NVTX elapsed including children/instrumentation',denominator='inclusive ancestors overlap; never sum rows',input_shape_layout=shape)
  op_rows.append(row);profile.append(row)
 writecsv(OUT/'raw'/(trace+'_operators.csv'),op_rows)
 phase_cpu={}
 for ph in sorted({r['label'][6:] for r in phases}):
  vals=[(r['end']-r['start'])/1000 for r in phases if r['label']=='phase:'+ph];phase_cpu[ph]=sum(vals)/nsteps
  profile.append(dict(trace=trace,scope='phase_cpu_wall',name=ph,count=len(vals),mean_us=statistics.mean(vals),total_us=sum(vals),us_per_step=sum(vals)/nsteps,timing='NVTX CPU elapsed, asynchronous enqueue, not CPU active time or GPU time',denominator='elapsed phase span'))
 apirows=[]
 for name in sorted({strings[r['nameId']] for r in api}):
  selected=[r for r in api if strings[r['nameId']]==name];vals=[(r['end']-r['start'])/1000 for r in selected]
  row=dict(trace=trace,scope='cuda_api_cpu_wall',name=name,count=len(vals),calls_per_step=len(vals)/nsteps,mean_us=statistics.mean(vals),total_us=sum(vals),us_per_step=sum(vals)/nsteps,timing='CUPTI CPU API duration; may overlap kernel execution',denominator='not additive with GPU durations')
  apirows.append(row);profile.append(row)
 writecsv(OUT/'raw'/(trace+'_api.csv'),apirows)
 copies=[dict(r) for r in c.execute('select * from CUPTI_ACTIVITY_KIND_MEMCPY')];copyrows=[]
 copytypes={r['id']:r['label'] for r in c.execute('select * from ENUM_CUDA_MEMCPY_OPER')}
 for kind in sorted({r['copyKind'] for r in copies}):
  v=[r for r in copies if r['copyKind']==kind];tot=sum(r['end']-r['start'] for r in v)/1000
  row=dict(trace=trace,scope='GPU_memcpy',name=copytypes.get(kind,str(kind)),count=len(v),calls_per_step=len(v)/nsteps,total_us=tot,us_per_step=tot/nsteps,mean_us=tot/len(v),bytes=sum(r['bytes'] for r in v),timing='CUPTI memcpy activity',denominator='separate from kernel sum');copyrows.append(row);profile.append(row)
 # Union accounts for overlap; idle span is neither occupancy nor hardware utilization.
 spans=sorted([(k['start'],k['end']) for k in kernels]+[(r['start'],r['end']) for r in copies]);merged=[]
 for s,e in spans:
  if merged and s<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],e)
  else:merged.append([s,e])
 busy=sum(e-s for s,e in merged)/1000;step_wall=sum(r['end']-r['start'] for r in steps)/1000
 device=[dict(r) for r in c.execute('select * from TARGET_INFO_CUDA_DEVICE')]
 summary[trace]=dict(steps=nsteps,kernel_count=len(kernels),kernel_total_us=total,kernel_us_per_step=total/nsteps,
  step_wall_us_per_step=step_wall/nsteps,activity_union_us_per_step=busy/nsteps,activity_fraction_of_step_wall=busy/step_wall,
  kernel_under_10us=sum(k['duration_us']<10 for k in kernels),by_phase=coarse,by_category=cats,phase_cpu_us_per_step=phase_cpu,memcpy=copyrows,device=device,
  missing_correlations=sum(k['phase']=='uncorrelated' for k in kernels),attribution=('1610/1610 kernel sequence+phase+grid/block/register/shared signatures matched detailed trace; shapes inherited' if trace=='native_plain' else 'direct launch correlation to containing NVTX ranges'))
writecsv(OUT/'PROFILE.csv',profile)
(OUT/'raw/profile_summary.json').write_text(json.dumps(summary,indent=2))
# Actual GEMM FLOPs, once per compute op (do not double count wrappers).
gemms=[r for r in work if r['operator']=='native.gemm' or r['operator']=='cupy.tensordot' or (r['operator']=='MatMul.forward' and not r.get('attrs',{}).get('native'))]
grows=[]
for r in gemms:grows.append(dict(id=r['id'],phase=r['phase'],layer=r['layer'],operator=r['operator'],**r['gemm'],inputs=json.dumps(clean(r['input']))))
writecsv(OUT/'raw/gemm_workload.csv',grows)
flops=sum(r['gemm']['flops'] for r in gemms);native=[r for r in gemms if r['operator']=='native.gemm'];registered=[r for r in native if r['gemm']['kernel']=='gemm_register']
fstats=dict(total_gemm_flops_per_step=flops,native_gemm_calls=len(native),register_calls=len(registered),register_flops=sum(r['gemm']['flops'] for r in registered),native_flops=sum(r['gemm']['flops'] for r in native),pool_calls=sum('Pooling' in r['operator'] for r in work),pool_requests=sum(x['kind']=='pool_request' for x in mem),pool_requested_bytes=sum(x['size'] for x in mem if x['kind']=='pool_request'),device_allocations=sum(x['kind']=='device_alloc' for x in mem))
(OUT/'raw/workload_summary.json').write_text(json.dumps(fstats,indent=2))
print(json.dumps({k:{z:v[z] for z in ['steps','kernel_count','kernel_us_per_step','step_wall_us_per_step','activity_fraction_of_step_wall','by_category']} for k,v in summary.items()},indent=2));print(fstats)
