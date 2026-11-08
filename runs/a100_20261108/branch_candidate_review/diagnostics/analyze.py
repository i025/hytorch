from pathlib import Path
import csv,json,statistics as st,math,sqlite3,collections,bisect
O=Path(__file__).resolve().parents[1]
def read(name):return list(csv.DictReader((O/name).open()))
def write(name,rows):
 if not rows:return
 with (O/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def interval(d):
 n=len(d);t={6:2.570582,10:2.262157,12:2.200985,20:2.093024}[n];mean=st.mean(d);h=t*st.stdev(d)/math.sqrt(n)
 return mean,mean-h,mean+h
r=read('TRAINING_COMPARISON.csv');summary=[];pairs=[]
for model in ['donkey','tiny']:
 q=[x for x in r if x['model']==model];b={x['repeat']:float(x['wall_s']) for x in q if x['mode']=='baseline'}
 for mode in dict.fromkeys(x['mode'] for x in q):
  v=[x for x in q if x['mode']==mode];a=[float(x['wall_s']) for x in v];d=[float(x['wall_s'])-b[x['repeat']] for x in v];mu,lo,hi=interval(d)
  for x,delta in zip(v,d):pairs.append(dict(model=model,mode=mode,repeat=x['repeat'],baseline_s=b[x['repeat']],candidate_s=x['wall_s'],delta_s=delta))
  bm=st.mean(b.values());am=st.mean(a)
  summary.append(dict(experiment='same_process',model=model,mode=mode,repeats=len(a),baseline_mean_s=bm,candidate_mean_s=am,candidate_std_s=st.stdev(a),speedup=bm/am,wall_reduction_pct=100*(1-am/bm),paired_delta_s=mu,ci95_low_s=lo,ci95_high_s=hi))
r=read('ORIGINAL_COMPARISON.csv');old={x['pair']:float(x['wall_s']) for x in r if x['source']=='old'};new={x['pair']:float(x['wall_s']) for x in r if x['source']=='new'}
d=[new[k]-old[k] for k in old];mu,lo,hi=interval(d);a,b=st.mean(old.values()),st.mean(new.values())
summary.append(dict(experiment='exact_4317bf5_paired_processes',model='donkey',mode='relu_prune',repeats=len(d),baseline_mean_s=a,candidate_mean_s=b,candidate_std_s=st.stdev(new.values()),speedup=a/b,wall_reduction_pct=100*(1-b/a),paired_delta_s=mu,ci95_low_s=lo,ci95_high_s=hi))
write('SUMMARY.csv',summary);write('TRAINING_PAIRS.csv',pairs)
rows=[]
for key,v in __import__('itertools').groupby(sorted(read('RELU_BENCHMARK.csv'),key=lambda r:(r['shape'],r['layout'],r['mode'])),key=lambda r:(r['shape'],r['layout'],r['mode'])):
 v=list(v)
 for metric in ['event_interval_us','cpu_submit_us','synchronized_wall_us']:
  vals=[float(x[metric]) for x in v];rows.append(dict(shape=key[0],layout=key[1],mode=key[2],metric=metric,repeats=len(v),median_us=st.median(vals),min_us=min(vals),max_us=max(vals),std_us=st.stdev(vals)))
write('RELU_SUMMARY.csv',rows)
profiles=[];kernels=[];timelines=[]
for label in ['original','candidate']:
 c=sqlite3.connect(O/f'traces/{label}.sqlite');c.row_factory=sqlite3.Row
 strings={r['id']:r['value'] for r in c.execute('select * from StringIds')};phases=[];steps=[]
 phase_names=['data_load','data_prepare','h2d','forward','loss','backward','optimizer']
 for row in c.execute('select * from NVTX_EVENTS where end is not null'):
  r=dict(row);name=r['text'] or strings.get(r['textId'],'')
  if name in phase_names:phases.append(dict(r,name=name))
  if name=='step':steps.append(r)
 n=len(steps);assert n==20
 # Attribution requires same host thread and containment, then CUDA correlation ID.
 apis={};apirows=[]
 for row in c.execute('select * from CUPTI_ACTIVITY_KIND_RUNTIME'):
  r=dict(row);matching=[p for p in phases if p['globalTid']==r['globalTid'] and p['start']<=r['start'] and p['end']>=r['end']]
  phase=matching[-1]['name'] if matching else 'outside';inside=any(s['globalTid']==r['globalTid'] and s['start']<=r['start'] and s['end']>=r['end'] for s in steps)
  r.update(name=strings[r['nameId']],phase=phase,inside=inside,us=(r['end']-r['start'])/1000);apis[r['correlationId']]=r;apirows.append(r)
 ks=[]
 for row in c.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL'):
  r=dict(row);a=apis.get(r['correlationId'],{});r.update(name=strings[r['demangledName']],phase=a.get('phase','outside'),inside=a.get('inside',False),us=(r['end']-r['start'])/1000);ks.append(r)
 for phase in phase_names:
  kk=[k for k in ks if k['phase']==phase];aa=[a for a in apirows if a['phase']==phase];pp=[p for p in phases if p['name']==phase]
  profiles.append(dict(source=label,phase=phase,steps=n,kernels_per_step=len(kk)/n,kernel_us_per_step=sum(k['us'] for k in kk)/n,cpu_nvtx_us_per_step=sum(p['end']-p['start'] for p in pp)/n/1000,cuda_api_us_per_step=sum(a['us'] for a in aa)/n,launch_api_us_per_step=sum(a['us'] for a in aa if 'LaunchKernel' in a['name'])/n))
 groups=collections.defaultdict(list)
 for k in ks:groups[(k['phase'],k['name'])].append(k)
 for (phase,name),v in groups.items():kernels.append(dict(source=label,phase=phase,kernel=name,calls=len(v),calls_per_step=len(v)/n,total_us=sum(k['us'] for k in v),us_per_step=sum(k['us'] for k in v)/n))
 timelines.append(dict(source=label,steps=n,kernels_per_step=len([k for k in ks if k['inside']])/n,kernel_us_per_step=sum(k['us'] for k in ks if k['inside'])/n))
write('PROFILE_PHASES.csv',profiles);write('PROFILE_KERNELS.csv',kernels);write('PROFILE_TOTALS.csv',timelines)
print(json.dumps(summary,indent=2));print(timelines)
