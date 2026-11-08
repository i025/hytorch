"""CPU-only CUPTI/NVTX joins, observed dispatch and paired statistics."""
from pathlib import Path
import sqlite3,json,csv,statistics,collections,math
ROOT=Path.cwd();O=ROOT/'runs/a100_20261106/gemm_shape_dispatch';T=ROOT/'runs/a100_20261106/gemm_shape_dispatch_traces'
def write(path,rows):
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with path.open('w') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def readtrace(name):
 c=sqlite3.connect(T/(name+'.sqlite'));c.row_factory=sqlite3.Row
 strings={r['id']:r['value'] for r in c.execute('select * from StringIds')};ranges=[]
 for r in c.execute('select * from NVTX_EVENTS where end is not null'):
  d=dict(r);d['label']=d['text'] or strings.get(d['textId'],'');ranges.append(d)
 api=[dict(r) for r in c.execute('select * from CUPTI_ACTIVITY_KIND_RUNTIME')];apis={r['correlationId']:r for r in api};kernels=[]
 for r in c.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL order by start'):
  d=dict(r);d['name']=strings[d['demangledName']];d['us']=(d['end']-d['start'])/1000;d['api']=apis.get(d['correlationId']);kernels.append(d)
 return c,strings,ranges,api,kernels
def locate(k,ranges):
 a=k['api'];v=[r for r in ranges if a and a['globalTid']==r['globalTid'] and r['start']<=a['start'] and r['end']>=a['end']]
 return min(v,key=lambda r:r['end']-r['start']) if v else None
c,ss,ranges,api,ks=readtrace('gemm');ranges=[r for r in ranges if r['label'].startswith('gemm:')];groups=collections.defaultdict(list)
for k in ks:
 r=locate(k,ranges)
 if not r:continue
 _,si,tile,mode,rep=r['label'].split(':');k['repeat']=int(rep);groups[(si,tile,mode)].append(k)
comparison=list(csv.DictReader((O/'raw/gemm_paired_retry/comparison.csv').open()));decisions=list(csv.DictReader((O/'raw/gemm_paired_retry/decisions.csv').open()))
observed=[]
for r in comparison:
 key=(r['shape_id'],r['tile'],r['implementation']);v=groups[key];assert v,key
 sigs={json.dumps(dict(name=k['name'],grid=[k['gridX'],k['gridY'],k['gridZ']],block=[k['blockX'],k['blockY'],k['blockZ']],registers=k['registersPerThread'],shared_bytes=k['staticSharedMemory']+k['dynamicSharedMemory']),sort_keys=True) for k in v}
 sigs=[json.loads(s) for s in sorted(sigs)];numreps=len(set(k['repeat'] for k in v));assert numreps==2
 r.update(profile_direct_kernel_us=sum(k['us'] for k in v)/numreps,observed_kernel_signatures=json.dumps(sigs),profile_kernel_launches_per_call=len(v)/numreps,profile_note='separate two-call direct-launch Nsight check; primary performance=12 unprofiled CUDA-event graph repeats')
 d=next(d for d in decisions if (d['shape_id'],d['tile'],d['implementation'])==key)
 if r['implementation']!='cublas':
  assert len(sigs)==1 and d['actual_graph_kernel'] in sigs[0]['name']
  m,n,t=int(r['M']),int(r['N']),int(r['tile']);reg='gemm_register' in sigs[0]['name'];expected=[(n+(64 if reg else t)-1)//(64 if reg else t),(m+(64 if reg else t)-1)//(64 if reg else t),1]
  assert sigs[0]['grid']==expected,(key,sigs,expected)
  assert sigs[0]['block']==([16,16,1] if reg else [t,t,1])
 d.update(actual_kernel_names=json.dumps([s['name'] for s in sigs]),actual_grid=json.dumps([s['grid'] for s in sigs]),actual_block=json.dumps([s['block'] for s in sigs]),actual_ctas=json.dumps([math.prod(s['grid']) for s in sigs]),actual_registers=json.dumps([s['registers'] for s in sigs]),actual_shared_bytes=json.dumps([s['shared_bytes'] for s in sigs]),profile_direct_kernel_us=r['profile_direct_kernel_us'])
write(O/'GEMM_COMPARISON.csv',comparison);write(O/'DISPATCH_DECISIONS.csv',decisions)
summaries={};profile=[]
for mode in ['old','new']:
 c,ss,ranges,api,kernels=readtrace('train_'+mode);phases=[r for r in ranges if r['label'].startswith('phase:')];steps=[r for r in ranges if r['label'].startswith('step:')];ns=len(steps)
 for k in kernels:
  phase=locate(k,phases);k['phase']=phase['label'][6:] if phase else 'outside';k.pop('api')
 assert all(k['phase']!='outside' for k in kernels)
 write(O/'raw'/f'train_{mode}_kernels.csv',kernels)
 pg={ph:sum(k['us'] for k in kernels if k['phase']==ph)/ns for ph in {k['phase'] for k in kernels}}
 cpu={ph:sum(r['end']-r['start'] for r in phases if r['label']=='phase:'+ph)/1000/ns for ph in {r['label'][6:] for r in phases}}
 ag=collections.defaultdict(list)
 for r in api:ag[ss[r['nameId']]].append((r['end']-r['start'])/1000)
 apisummary={k:dict(count=len(v),us_per_step=sum(v)/ns) for k,v in ag.items()}
 total=sum(k['us'] for k in kernels)
 g=collections.defaultdict(list)
 for k in kernels:g[(k['phase'],k['name'],k['gridX'],k['gridY'],k['gridZ'],k['blockX'],k['blockY'],k['blockZ'])].append(k['us'])
 for key,v in g.items():
  phase,name,gx,gy,gz,bx,by,bz=key;profile.append(dict(mode=mode,scope='kernel',phase=phase,name=name,count=len(v),calls_per_step=len(v)/ns,mean_us=statistics.mean(v),total_us=sum(v),us_per_step=sum(v)/ns,percent_kernel_sum=100*sum(v)/total,grid=f'{gx},{gy},{gz}',block=f'{bx},{by},{bz}'))
 copies=[dict(r) for r in c.execute('select * from CUPTI_ACTIVITY_KIND_MEMCPY')]
 spans=sorted([(k['start'],k['end']) for k in kernels]+[(x['start'],x['end']) for x in copies]);merged=[]
 for s,e in spans:
  if merged and s<=merged[-1][1]:merged[-1][1]=max(e,merged[-1][1])
  else:merged.append([s,e])
 start=min(r['start'] for r in steps);end=max(r['end'] for r in phases if r['label']=='phase:final_sync')
 busy=sum(max(0,min(e,end)-max(s,start)) for s,e in merged)
 summaries[mode]=dict(steps=ns,kernel_count=len(kernels),kernel_us_per_step=total/ns,phase_gpu_us=pg,phase_cpu_elapsed_us=cpu,cuda_api=apisummary,native_gemm_us_per_step=sum(k['us'] for k in kernels if 'gemm_shared' in k['name'] or 'gemm_register' in k['name'])/ns,all_gemm_named_us_per_step=sum(k['us'] for k in kernels if 'gemm' in k['name'].lower())/ns,activity_union_fraction=busy/(end-start),trace_wall_us_per_step=(end-start)/1000/ns,fp64_cast_count=sum(k['name'].startswith('cupy_copy__float32_float64') for k in kernels))
 for phase,val in pg.items():profile.append(dict(mode=mode,scope='phase_gpu_sum',name=phase,us_per_step=val))
 for phase,val in cpu.items():profile.append(dict(mode=mode,scope='phase_cpu_elapsed',name=phase,us_per_step=val))
 for name,val in apisummary.items():profile.append(dict(mode=mode,scope='api_cpu_elapsed',name=name,**val))
write(O/'PROFILE.csv',profile)
(O/'raw/profile_summary.json').write_text(json.dumps(summaries,indent=2))
training=list(csv.DictReader((O/'raw/training_paired/training.csv').open()))
for r in training:
 s=summaries[r['mode']];r.update(profile_kernel_us_per_step=s['kernel_us_per_step'],profile_forward_us=s['phase_gpu_us']['forward'],profile_backward_us=s['phase_gpu_us']['backward'],profile_sgd_us=s['phase_gpu_us']['update'],profile_native_gemm_us=s['native_gemm_us_per_step'],profile_all_gemm_named_us=s['all_gemm_named_us_per_step'],profile_cuda_api_cpu_us=sum(v['us_per_step'] for v in s['cuda_api'].values()),profile_sync_cpu_us=sum(v['us_per_step'] for k,v in s['cuda_api'].items() if 'Synchronize' in k),profile_note='one separate 10-step trace per mode, not repeated wall-run profiling; CPU and GPU overlap')
write(O/'TRAINING_COMPARISON.csv',training)
stats={}
for mode in ['old','new']:
 r=[x for x in training if x['mode']==mode];v=[float(x['wall_s']) for x in r];stats[mode]=dict(wall_mean=statistics.mean(v),wall_median=statistics.median(v),wall_sd=statistics.stdev(v),samples_per_s=statistics.mean(float(x['samples_per_s']) for x in r),loss=float(r[0]['loss']))
old={r['repeat']:float(r['wall_s']) for r in training if r['mode']=='old'};new={r['repeat']:float(r['wall_s']) for r in training if r['mode']=='new'};ratios=[old[k]/new[k] for k in old];deltas=[old[k]-new[k] for k in old]
stats.update(paired_speedups=ratios,paired_speedup_mean=statistics.mean(ratios),paired_speedup_sd=statistics.stdev(ratios),paired_wall_deltas_s=deltas,paired_wall_delta_mean_s=statistics.mean(deltas),paired_wall_delta_sd_s=statistics.stdev(deltas),paired_wall_delta_95_t_interval_s=[statistics.mean(deltas)+sgn*2.44691185*statistics.stdev(deltas)/math.sqrt(len(deltas)) for sgn in [-1,1]])
(O/'raw/training_statistics.json').write_text(json.dumps(stats,indent=2));print(json.dumps(stats,indent=2));print(json.dumps(summaries,indent=2))
