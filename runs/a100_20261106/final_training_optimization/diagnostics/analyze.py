from pathlib import Path
import csv,json,statistics,math,sqlite3,collections
O=Path.cwd()/'runs/a100_20261106/final_training_optimization'
def read(name):return list(csv.DictReader((O/name).open()))
def write(name,rows):
 with (O/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
train=read('TRAINING.csv');by={(r['mode'],int(r['repeat'])):float(r['wall_s']) for r in train};stats={};paired=[]
for mode in ['B0','D','S','DS']:
 vals=[float(r['wall_s']) for r in train if r['mode']==mode];stats[mode]=dict(mean=statistics.mean(vals),sd=statistics.stdev(vals),median=statistics.median(vals),samples_per_s=statistics.mean(float(r['samples_per_s']) for r in train if r['mode']==mode))
 if mode!='B0':
  d=[by['B0',i]-by[mode,i] for i in range(12)];mu=statistics.mean(d);half=2.20098516*statistics.stdev(d)/math.sqrt(12)
  stats[mode].update(delta_mean_s=mu,ci95_s=[mu-half,mu+half],wins=sum(v>0 for v in d),wall_reduction_pct=100*mu/stats['B0']['mean'])
  paired.extend(dict(mode=mode,repeat=i,B0_minus_mode_s=v) for i,v in enumerate(d))
write('PAIRED.csv',paired);(O/'raw/training_summary.json').write_text(json.dumps(stats,indent=2));print(json.dumps(stats,indent=2))
# CUPTI GPU times attributed by host-launch correlation within NVTX ranges.
def trace(path):
 c=sqlite3.connect(path);c.row_factory=sqlite3.Row;strings={r['id']:r['value'] for r in c.execute('select * from StringIds')};ranges=[]
 for r in c.execute('select * from NVTX_EVENTS where end is not null'):
  d=dict(r);d['label']=d['text'] or strings.get(d['textId'],'');ranges.append(d)
 api=[dict(r) for r in c.execute('select * from CUPTI_ACTIVITY_KIND_RUNTIME')];aid={r['correlationId']:r for r in api};ks=[]
 for r in c.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL'):
  d=dict(r);a=aid.get(d['correlationId']);d.update(name=strings[d['demangledName']],us=(d['end']-d['start'])/1000,api=a)
  d['ranges']=[v['label'] for v in ranges if a and v['globalTid']==a['globalTid'] and v['start']<=a['start'] and v['end']>=a['end']];ks.append(d)
 return c,strings,ranges,api,ks
profiles=[];summary=[]
for rep in range(2):
 for mode in ['B0','D','S','DS']:
  c,strings,rs,api,ks=trace(O/'traces'/f'train_{mode}_{rep}.sqlite');groups=collections.defaultdict(list)
  for k in ks:
   phase=next((p for p in ['forward','backward','update','transfer'] if p in k['ranges']),'outside');cat=next((p for p in ['dweight','dinput'] if p in k['ranges']),phase)
   groups[(phase,cat,k['name'],k['gridX'],k['blockX'],k['registersPerThread'])].append(k['us'])
  for (phase,cat,name,grid,block,regs),v in groups.items():profiles.append(dict(mode=mode,repeat=rep,scope='GPU_kernel',phase=phase,category=cat,name=name,calls=len(v),calls_per_step=len(v)/10,total_us=sum(v),us_per_step=sum(v)/10,gridX=grid,blockX=block,registers=regs))
  for label in ['forward','backward','update','dweight','dinput']:
   rr=[r for r in rs if r['label']==label];profiles.append(dict(mode=mode,repeat=rep,scope='CPU_NVTX_elapsed',name=label,us_per_step=sum(r['end']-r['start'] for r in rr)/10000))
  ag=collections.defaultdict(list)
  for r in api:ag[strings[r['nameId']]].append((r['end']-r['start'])/1000)
  for name,v in ag.items():profiles.append(dict(mode=mode,repeat=rep,scope='CUDA_API_CPU_elapsed',name=name,calls=len(v),us_per_step=sum(v)/10))
  summary.append(dict(mode=mode,repeat=rep,kernels_per_step=len(ks)/10,im2col_per_step=sum('im2col' in k['name'] for k in ks)/10,update_kernels_per_step=sum('update' in k['ranges'] for k in ks)/10,dweight_kernels_per_step=sum('dweight' in k['ranges'] for k in ks)/10,GPU_us_per_step=sum(k['us'] for k in ks)/10,dweight_us_per_step=sum(k['us'] for k in ks if 'dweight' in k['ranges'])/10,dinput_us_per_step=sum(k['us'] for k in ks if 'dinput' in k['ranges'])/10,forward_us_per_step=sum(k['us'] for k in ks if 'forward' in k['ranges'])/10,update_us_per_step=sum(k['us'] for k in ks if 'update' in k['ranges'])/10,memcpy_calls=len(list(c.execute('select * from CUPTI_ACTIVITY_KIND_MEMCPY')))))
write('PROFILE.csv',profiles);write('PROFILE_SUMMARY.csv',summary)
c,strings,rs,api,ks=trace(O/'traces/operators.sqlite');groups=collections.defaultdict(list)
for k in ks:
 label=next((r for r in k['ranges'] if r.startswith(('dwbench:','sgdbench:'))),None)
 if label:groups[(label,k['name'],k['gridX'],k['blockX'],k['registersPerThread'],k['localMemoryTotal'])].append(k)
ops=[]
for (label,name,grid,block,reg,local),v in groups.items():ops.append(dict(range=label,kernel=name,count=len(v),total_gpu_us=sum(k['us'] for k in v),gridX=grid,blockX=block,registers=reg,local_bytes=local,launch_api_cpu_us=sum((k['api']['end']-k['api']['start'])/1000 for k in v)))
write('OPERATOR_PROFILE.csv',ops)
