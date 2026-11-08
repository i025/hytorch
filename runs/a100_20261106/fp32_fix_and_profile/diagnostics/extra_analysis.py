"""CPU-only trace export analysis and dtype audit."""
from pathlib import Path
import sqlite3,json,csv,collections,statistics,argparse
ROOT=Path.cwd();p=argparse.ArgumentParser();p.add_argument('--output',default='runs/a100_20261106/fp32_fix_and_profile');p.add_argument('--traces',default='runs/a100_20261106/fp32_fix_and_profile_traces');args=p.parse_args();OUT=ROOT/args.output;TR=ROOT/args.traces
def write(path,rows):
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with path.open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def clean(v):
 if isinstance(v,dict):return {k:clean(z) for k,z in v.items() if k!='ptr'}
 if isinstance(v,list):return [clean(z) for z in v]
 return v
audit=[]
for mode,folder in [('before_native','before_workload'),('after_native','workload'),('after_cupy','cupy_trace')]:
 records=[r for r in json.loads((OUT/'raw'/folder/'operators.json').read_text()) if r['step']==0]
 for layer in ['conv1','conv2','conv3','conv4','conv5','fc1','fc2','fc3']:
  conv=layer.startswith('conv');fwd=next(r for r in records if r['layer']==layer and r['phase']=='forward' and r['operator']==('Conv2d.forward' if conv else 'MatMul.forward'))
  back=next(r for r in records if r['layer']==layer and r['operator']==('Conv2d.backward' if conv else 'MatMul.backward'))
  values={'forward_activation':fwd['output'],'dInput':back['output'][0],'dWeight':back['output'][1]}
  if conv:values['dBias']=back['output'][2]
  else:
   ab=next(r for r in records if r['layer']==layer and r['operator']=='Add.backward')
   values['dBias']=next(v for v in ab['output'] if len(v['shape'])==1)
  for role,v in values.items():audit.append(dict(mode=mode,layer=layer,role=role,dtype=v['dtype'],shape=json.dumps(v['shape']),strides=json.dumps(v['strides_bytes']),source=f'raw/{folder}/operators.json',observation='actual tensor'))
 # Keep all observed operator input/output tensors, including constants.
 flat=[]
 def walk(v,where):
  if isinstance(v,dict) and 'dtype' in v:flat.append(dict(mode=mode,operator=r['operator'],layer=r['layer'],phase=r['phase'],where=where,**clean(v)))
  elif isinstance(v,list):
   for i,z in enumerate(v):walk(z,f'{where}[{i}]')
 for r in records:
  walk(r['input'],'input');walk(r.get('output'),'output')
 write(OUT/'raw'/f'{mode}_all_tensor_dtypes.csv',flat)
 if mode!='before_native':assert not any(x['dtype']=='float64' for x in flat)
for x in json.loads((OUT/'raw/training/parameter_dtype.json').read_text()):
 if x['repeat']!=0:continue
 for field,role in [('param_dtype','parameter_after_update'),('grad_dtype','parameter_grad'),('sgd_delta_dtype','SGD_lr_times_grad')]:
  audit.append(dict(mode=x['mode'],layer=x['parameter'],role=role,dtype=x[field],source='raw/training/parameter_dtype.json',observation='actual expression dtype after training, outside timing'))
write(OUT/'DTYPE_AUDIT.csv',audit)
profiles=list(csv.DictReader((OUT/'PROFILE.csv').open()));summaries={}
for mode in ['before_native','after_native','after_cupy','torch']:
 trace='train_'+mode;c=sqlite3.connect(TR/(trace+'.sqlite'));c.row_factory=sqlite3.Row
 strings={r['id']:r['value'] for r in c.execute('select * from StringIds')}
 ranges=[]
 for r in c.execute('select * from NVTX_EVENTS where end is not null'):
  d=dict(r);d['label']=d['text'] or strings.get(d['textId'],'');ranges.append(d)
 phases=[r for r in ranges if r['label'].startswith('phase:')];steps=[r for r in ranges if r['label'].startswith('step:')];ns=len(steps)
 api=[dict(r) for r in c.execute('select * from CUPTI_ACTIVITY_KIND_RUNTIME')];ar={r['correlationId']:r for r in api}
 kernels=[]
 for r in c.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL order by start'):
  d=dict(r);call=ar.get(d['correlationId']);match=[v for v in phases if call and v['globalTid']==call['globalTid'] and v['start']<=call['start'] and v['end']>=call['end']]
  # PyTorch autograd launches on a worker thread. Use the unique enclosing
  # main-thread backward API interval only when same-thread attribution is absent.
  cross_thread=False
  if not match and mode=='torch' and call:
   cross=[v for v in phases if v['label']=='phase:backward' and v['start']<=call['start'] and v['end']>=call['end']]
   if len(cross)==1:match=cross;cross_thread=True
  d.update(cross_thread_backward=cross_thread,name=strings[d['demangledName']],phase=min(match,key=lambda x:x['end']-x['start'])['label'][6:] if match else 'outside',us=(d['end']-d['start'])/1000)
  kernels.append(d)
 write(OUT/'raw'/f'{trace}_kernels.csv',kernels)
 total=sum(r['us'] for r in kernels);phasegpu={p:sum(r['us'] for r in kernels if r['phase']==p)/ns for p in {r['phase'] for r in kernels}}
 cpuphase={p:sum(r['end']-r['start'] for r in phases if r['label']=='phase:'+p)/1000/ns for p in {r['label'][6:] for r in phases}}
 apigroups=collections.defaultdict(list)
 for r in api:apigroups[strings[r['nameId']]].append((r['end']-r['start'])/1000)
 apisummary={k:dict(count=len(v),us_per_step=sum(v)/ns) for k,v in apigroups.items()}
 copies=[dict(r) for r in c.execute('select * from CUPTI_ACTIVITY_KIND_MEMCPY')]
 spans=sorted([(r['start'],r['end']) for r in kernels]+[(r['start'],r['end']) for r in copies]);merged=[]
 for s,e in spans:
  if merged and s<=merged[-1][1]:merged[-1][1]=max(e,merged[-1][1])
  else:merged.append([s,e])
 lo=min(r['start'] for r in steps);hi=max([r['end'] for r in steps]+[r['end'] for r in phases if r['label']=='phase:final_sync'])
 busy=sum(max(0,min(e,hi)-max(s,lo)) for s,e in merged)
 summary=dict(cross_thread_backward_kernels=sum(r['cross_thread_backward'] for r in kernels),steps=ns,kernel_count=len(kernels),kernel_us_per_step=total/ns,phase_gpu_us=phasegpu,phase_cpu_elapsed_us=cpuphase,cuda_api=apisummary,span_us_per_step=(hi-lo)/1000/ns,activity_union_fraction=busy/(hi-lo),float32_to_float64_kernel_count=sum(r['name'].startswith('cupy_copy__float32_float64') for r in kernels),fp64_named_kernel_count=sum(any(v in r['name'].lower() for v in ['dgemm','f64gemm','d884gemm']) for r in kernels),memcpy_count=len(copies),memcpy_us_per_step=sum(r['end']-r['start'] for r in copies)/1000/ns,unattributed_kernels=sum(r['phase']=='outside' for r in kernels),kernel_under_10us=sum(r['us']<10 for r in kernels))
 summaries[mode]=summary
 groups=collections.defaultdict(list)
 for r in kernels:groups[(r['phase'],r['name'],r['gridX'],r['gridY'],r['gridZ'],r['blockX'],r['blockY'],r['blockZ'])].append(r['us'])
 for (phase,name,gx,gy,gz,bx,by,bz),v in groups.items():profiles.append(dict(trace=trace,scope='kernel_by_phase',phase=phase,name=name,count=len(v),calls_per_step=len(v)/ns,total_us=sum(v),us_per_step=sum(v)/ns,mean_us=statistics.mean(v),percent_kernel_total=100*sum(v)/total,grid=f'{gx},{gy},{gz}',block=f'{bx},{by},{bz}',timing='Nsight CUPTI GPU kernel duration',denominator='kernel sum in own trace; separate from detailed trace'))
 for phase,us in phasegpu.items():profiles.append(dict(trace=trace,scope='phase_kernel_sum',name=phase,us_per_step=us,timing='CUPTI GPU kernel durations',denominator='overlaps kernel rows'))
 for phase,us in cpuphase.items():profiles.append(dict(trace=trace,scope='phase_cpu_wall',name=phase,us_per_step=us,timing='NVTX asynchronous CPU elapsed; not active CPU scheduling time'))
 for name,d in apisummary.items():profiles.append(dict(trace=trace,scope='cuda_api_cpu_wall',name=name,**d,timing='CUPTI API CPU elapsed; overlaps GPU'))
write(OUT/'PROFILE.csv',profiles)
(OUT/'raw/training_profile_summary.json').write_text(json.dumps(summaries,indent=2))
training=list(csv.DictReader((OUT/'raw/training/training.csv').open()))
for r in training:
 s=summaries[r['mode']]
 r.update(profile_gpu_kernel_us_per_step=s['kernel_us_per_step'],profile_forward_us=s['phase_gpu_us'].get('forward',0),profile_backward_us=s['phase_gpu_us'].get('backward',0),profile_sgd_us=s['phase_gpu_us'].get('update',0),profile_cuda_api_cpu_us=sum(x['us_per_step'] for x in s['cuda_api'].values()),profile_sync_cpu_us=sum(x['us_per_step'] for k,x in s['cuda_api'].items() if 'Synchronize' in k),profile_cpu_elapsed_forward_us=s['phase_cpu_elapsed_us']['forward'],profile_cpu_elapsed_backward_us=s['phase_cpu_elapsed_us']['backward'],profile_cpu_elapsed_update_us=s['phase_cpu_elapsed_us']['update'],profile_note='separate 10-step trace in same GPU window; repeated columns are not five profiler repeats; CPU and GPU overlap')
write(OUT/'TRAINING_COMPARISON.csv',training)
# Actual GEMM launch geometry from cuBLAS/native profile, matched by launch correlation and NVTX.
c=sqlite3.connect(TR/'gemm.sqlite');c.row_factory=sqlite3.Row
strings={r['id']:r['value'] for r in c.execute('select * from StringIds')};ranges=[]
for r in c.execute('select * from NVTX_EVENTS where end is not null'):
 d=dict(r);d['label']=d['text'] or strings.get(d['textId'],'')
 if d['label'].startswith('gemm:'):ranges.append(d)
api={r['correlationId']:dict(r) for r in c.execute('select * from CUPTI_ACTIVITY_KIND_RUNTIME')};bindings=collections.defaultdict(list)
for r in c.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL'):
 d=dict(r);a=api.get(d['correlationId']);match=[v for v in ranges if a and v['globalTid']==a['globalTid'] and v['start']<=a['start'] and v['end']>=a['end']]
 if not match:continue
 _,idx,mode,repeat=min(match,key=lambda x:x['end']-x['start'])['label'].split(':')
 bindings[(idx,mode)].append(dict(repeat=int(repeat),name=strings[d['demangledName']],grid=[d['gridX'],d['gridY'],d['gridZ']],block=[d['blockX'],d['blockY'],d['blockZ']],registers=d['registersPerThread'],shared=d['staticSharedMemory']+d['dynamicSharedMemory'],kernel_us=(d['end']-d['start'])/1000))
gemm=list(csv.DictReader((OUT/'raw/gemm/gemm.csv').open()))
for r in gemm:
 v=bindings[(r['shape_id'],r['implementation'])];assert len(v)>=20
 sig={json.dumps({k:x for k,x in d.items() if k not in ['repeat','kernel_us']},sort_keys=True) for d in v}
 r['observed_kernel_signatures']=json.dumps([json.loads(x) for x in sorted(sig)])
 r['profile_kernel_us_per_call']=sum(d['kernel_us'] for d in v)/20
 r['profile_kernel_launches_per_call']=len(v)/20
 if r['implementation']=='cublas':
  r['block_count']=';'.join(str(d['grid'][0]*d['grid'][1]*d['grid'][2]) for d in [json.loads(x) for x in sorted(sig)])
write(OUT/'GEMM_REAL_SHAPES.csv',gemm)
(OUT/'raw/gemm_kernel_bindings.json').write_text(json.dumps({str(k):v for k,v in bindings.items()},indent=2))
print(json.dumps(summaries,indent=2))
