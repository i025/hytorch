from pathlib import Path
import csv,json,sqlite3,collections,statistics,math
ROOT=Path.cwd();O=ROOT/'runs/a100_20261106/native_col2im';T=ROOT/'runs/a100_20261106/native_col2im_traces'
def write(path,rows):
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with path.open('w') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def trace(name):
 c=sqlite3.connect(T/(name+'.sqlite'));c.row_factory=sqlite3.Row;strings={r['id']:r['value'] for r in c.execute('select * from StringIds')};ranges=[]
 for r in c.execute('select * from NVTX_EVENTS where end is not null'):
  d=dict(r);d['label']=d['text'] or strings.get(d['textId'],'');ranges.append(d)
 api=[dict(r) for r in c.execute('select * from CUPTI_ACTIVITY_KIND_RUNTIME')];byid={r['correlationId']:r for r in api};kernels=[]
 for r in c.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL order by start'):
  d=dict(r);d.update(name=strings[d['demangledName']],us=(d['end']-d['start'])/1000,api=byid.get(d['correlationId']));kernels.append(d)
 return c,strings,ranges,api,kernels
def locate(k,rs):
 a=k['api'];v=[r for r in rs if a and a['globalTid']==r['globalTid'] and r['start']<=a['start'] and r['end']>=a['end']]
 return min(v,key=lambda r:r['end']-r['start']) if v else None
c,ss,rs,api,ks=trace('col2im');rs=[r for r in rs if r['label'].startswith('colbench:')];groups=collections.defaultdict(list)
for k in ks:
 r=locate(k,rs)
 if r:
  _,case,mode,rep=r['label'].split(':');k['rep']=rep;groups[(case,mode)].append(k)
bench=list(csv.DictReader((O/'raw/col2im_bench/benchmark.csv').open()))
for r in bench:
 v=groups[(r['case_id'],r['implementation'])];assert len(v)==3
 sig={json.dumps(dict(name=k['name'],grid=[k['gridX'],k['gridY'],k['gridZ']],block=[k['blockX'],k['blockY'],k['blockZ']],registers=k['registersPerThread'],shared_bytes=k['staticSharedMemory']+k['dynamicSharedMemory']),sort_keys=True) for k in v}
 assert len(sig)==1
 r.update(profile_kernel_us=statistics.mean(k['us'] for k in v),observed_signature=next(iter(sig)),profile_repeats=3)
write(O/'COL2IM_BENCHMARK.csv',bench)
reference=list(csv.DictReader(open(ROOT/'runs/a100_20261106/fp32_fix_and_profile/raw/native_plain_kernels.csv')))
profile=[];summaries={};layerrows=[]
for mode in ['cupy','native']:
 c,ss,rs,api,ks=trace('train_'+mode);phases=[r for r in rs if r['label'].startswith('phase:')];steps=[r for r in rs if r['label'].startswith('step:')];ns=len(steps);assert len(ks)==len(reference)==1470
 colnumber=0
 for i,k in enumerate(ks):
  r=locate(k,phases);k['phase']=r['label'][6:] if r else 'outside';assert k['phase']!='outside'
  col=k['name']=='col2im' or 'col2im_strided' in k['name']
  if col:
   k['layer']=['conv5','conv4','conv3','conv2','conv1'][colnumber%5];colnumber+=1
  else:k['layer']=reference[i]['layer']
  if k['phase']=='backward':
   ref=reference[i];assert k['phase']==ref['phase']
   if not col:assert k['name']==ref['name'] and all(str(k[a])==ref[a] for a in ['gridX','gridY','gridZ','blockX','blockY','blockZ'])
   else:assert ref['name']=='col2im' and k['layer']==ref['layer']
   k['category']=ref['category']
  else:k['category']=k['phase']
  k.pop('api')
 assert colnumber==50
 write(O/'raw'/f'train_{mode}_kernels.csv',ks)
 total=sum(k['us'] for k in ks);pg={p:sum(k['us'] for k in ks if k['phase']==p)/ns for p in {k['phase'] for k in ks}};cats={p:sum(k['us'] for k in ks if k['category']==p)/ns for p in {k['category'] for k in ks}}
 cpu={p:sum(r['end']-r['start'] for r in phases if r['label']=='phase:'+p)/1000/ns for p in {r['label'][6:] for r in phases}}
 ag=collections.defaultdict(list)
 for a in api:ag[ss[a['nameId']]].append((a['end']-a['start'])/1000)
 apis={k:dict(count=len(v),us_per_step=sum(v)/ns) for k,v in ag.items()}
 colks=[k for k in ks if k['name']=='col2im' or 'col2im_strided' in k['name']];layers={l:sum(k['us'] for k in colks if k['layer']==l)/ns for l in ['conv1','conv2','conv3','conv4','conv5']}
 copies=[dict(r) for r in c.execute('select * from CUPTI_ACTIVITY_KIND_MEMCPY')];spans=sorted([(k['start'],k['end']) for k in ks]+[(d['start'],d['end']) for d in copies]);merged=[]
 for s,e in spans:
  if merged and s<=merged[-1][1]:merged[-1][1]=max(e,merged[-1][1])
  else:merged.append([s,e])
 lo=min(r['start'] for r in steps);hi=max(r['end'] for r in phases if r['label']=='phase:final_sync');busy=sum(max(0,min(e,hi)-max(s,lo)) for s,e in merged)
 summaries[mode]=dict(steps=ns,kernel_count=len(ks),kernel_us_per_step=total/ns,phase_gpu_us=pg,category_gpu_us=cats,phase_cpu_elapsed_us=cpu,cuda_api=apis,col2im_us_per_step=sum(layers.values()),col2im_layers_us=layers,activity_union_fraction=busy/(hi-lo),trace_wall_us_per_step=(hi-lo)/1000/ns,memcpy_count=len(copies),memcpy_bytes=sum(d['bytes'] for d in copies),fp64_cast_count=sum(k['name'].startswith('cupy_copy__float32_float64') for k in ks))
 groups=collections.defaultdict(list)
 for k in ks:groups[(k['phase'],k['category'],k['layer'],k['name'],k['gridX'],k['gridY'],k['gridZ'],k['blockX'],k['blockY'],k['blockZ'],k['registersPerThread'])].append(k['us'])
 for key,v in groups.items():
  phase,cat,layer,name,gx,gy,gz,bx,by,bz,reg=key
  profile.append(dict(mode=mode,scope='kernel_by_layer',phase=phase,category=cat,layer=layer,name=name,count=len(v),calls_per_step=len(v)/ns,mean_us=statistics.mean(v),us_per_step=sum(v)/ns,total_us=sum(v),pct_kernel_sum=100*sum(v)/total,grid=f'{gx},{gy},{gz}',block=f'{bx},{by},{bz}',registers=reg))
 for p,v in pg.items():profile.append(dict(mode=mode,scope='phase_gpu_sum',name=p,us_per_step=v))
 for p,v in cpu.items():profile.append(dict(mode=mode,scope='phase_cpu_elapsed',name=p,us_per_step=v))
 for p,v in cats.items():profile.append(dict(mode=mode,scope='category_gpu_sum',name=p,us_per_step=v))
 for p,v in apis.items():profile.append(dict(mode=mode,scope='api_cpu_elapsed',name=p,**v))
write(O/'PROFILE_BEFORE_AFTER.csv',profile);(O/'raw/profile_summary.json').write_text(json.dumps(summaries,indent=2))
t=list(csv.DictReader((O/'raw/training/training.csv').open()));old={r['repeat']:float(r['wall_s']) for r in t if r['mode']=='cupy'};new={r['repeat']:float(r['wall_s']) for r in t if r['mode']=='native'}
for r in t:
 s=summaries[r['mode']];r.update(paired_cupy_minus_native_s=old[r['repeat']]-new[r['repeat']],paired_speedup=old[r['repeat']]/new[r['repeat']],profile_kernel_us=s['kernel_us_per_step'],profile_col2im_us=s['col2im_us_per_step'],profile_backward_us=s['phase_gpu_us']['backward'],profile_forward_us=s['phase_gpu_us']['forward'],profile_sgd_us=s['phase_gpu_us']['update'],profile_api_cpu_us=sum(v['us_per_step'] for v in s['cuda_api'].values()),profile_launch_cpu_us=sum(v['us_per_step'] for k,v in s['cuda_api'].items() if 'Launch' in k),profile_sync_cpu_us=sum(v['us_per_step'] for k,v in s['cuda_api'].items() if 'Synchronize' in k),profile_backward_cpu_elapsed_us=s['phase_cpu_elapsed_us']['backward'],profile_note='one independent 10-step trace per mode; repeated columns are not 7 independent profiles; CPU elapsed and GPU overlap')
write(O/'TRAINING_COMPARISON.csv',t)
stats={}
for mode in ['cupy','native']:
 rows=[r for r in t if r['mode']==mode];vals=[float(r['wall_s']) for r in rows];stats[mode]=dict(wall_mean=statistics.mean(vals),wall_sd=statistics.stdev(vals),samples_per_s=statistics.mean(float(r['samples_per_s']) for r in rows),loss=float(rows[0]['loss']))
d=[old[k]-new[k] for k in old];stats.update(paired_cupy_minus_native_s=d,paired_delta_mean_s=statistics.mean(d),paired_delta_sd_s=statistics.stdev(d),paired_delta_95_t_interval_s=[statistics.mean(d)+a*2.44691185*statistics.stdev(d)/math.sqrt(7) for a in [-1,1]])
(O/'raw/training_statistics.json').write_text(json.dumps(stats,indent=2));print(json.dumps(stats,indent=2));print(json.dumps(summaries,indent=2))
