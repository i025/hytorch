from pathlib import Path
import csv,sqlite3,collections,json,sys,statistics
O=Path(__file__).resolve().parents[1]
def write(path,rows):
 if not rows:return
 with path.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def trace(path):
 c=sqlite3.connect(path);c.row_factory=sqlite3.Row;strings={r['id']:r['value'] for r in c.execute('select * from StringIds')};rs=[]
 for r in c.execute('select * from NVTX_EVENTS where end is not null'):
  d=dict(r);d['label']=d['text'] or strings.get(d['textId'],'');rs.append(d)
 ap=[dict(r) for r in c.execute('select * from CUPTI_ACTIVITY_KIND_RUNTIME')];aid={r['correlationId']:r for r in ap};ks=[]
 for r in c.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL'):
  d=dict(r);a=aid.get(d['correlationId']);d.update(name=strings[d['demangledName']],us=(d['end']-d['start'])/1000,api=a)
  d['ranges']=[v['label'] for v in rs if a and (v['globalTid']==a['globalTid'] or v['label'] in ['forward','backward','update','transfer','step'] or v['label'].startswith('op|')) and v['start']<=a['start'] and v['end']>=a['end']];ks.append(d)
 tables={r[0] for r in c.execute("select name from sqlite_master where type='table'")}
 copies=[dict(r) for r in c.execute('select * from CUPTI_ACTIVITY_KIND_MEMCPY')] if 'CUPTI_ACTIVITY_KIND_MEMCPY' in tables else []
 return c,strings,rs,ap,ks,copies

def union_ns(ks):
 end=0;total=0
 for k in sorted(ks,key=lambda x:x['start']):total+=max(0,k['end']-max(end,k['start']));end=max(end,k['end'])
 return total
profiles=[];hot=[];ops=[];opcalls=[]
for path in sorted((O/'traces').glob('*.sqlite')):
 c,ss,rs,api,ks,copies=trace(path)
 if path.stem.startswith(('operators','layers')):
  repeats=15 if path.stem=='operators_15' or path.stem.startswith('layers') else 3
  for rr in rs:
   if not rr['label'].startswith('op|'):continue
   vv=[k for k in ks if k['api'] and rr['start']<=k['api']['start'] and k['api']['end']<=rr['end']]
   pa=[z for z in api if rr['start']<=z['start'] and z['end']<=rr['end']]
   opcalls.append(dict(trace=path.stem,case=rr['label'].split('|')[1],implementation=rr['label'].split('|')[2],gpu_kernel_us=sum(k['us'] for k in vv),kernel_count=len(vv),cpu_elapsed_us=(rr['end']-rr['start'])/1000,cuda_api_elapsed_us=sum(z['end']-z['start'] for z in pa)/1000,cuda_api_calls=len(pa),launch_api_elapsed_us=sum(z['end']-z['start'] for z in pa if 'LaunchKernel' in ss[z['nameId']])/1000,sync_api_elapsed_us=sum(z['end']-z['start'] for z in pa if 'Synchronize' in ss[z['nameId']])/1000))
  groups=collections.defaultdict(list)
  for k in ks:
   label=next((r for r in k['ranges'] if r.startswith('op|')),None)
   if label:groups[(label,k['name'],k['gridX'],k['gridY'],k['blockX'],k['blockY'],k['registersPerThread'])].append(k)
  for (label,name,gx,gy,bx,by,reg),v in groups.items():ops.append(dict(trace=path.stem,case=label.split('|')[1],implementation=label.split('|')[2],kernel=name,calls=len(v),total_gpu_us=sum(k['us'] for k in v),gpu_us_per_call=sum(k['us'] for k in v)/repeats,calls_per_operator=len(v)/repeats,gridX=gx,gridY=gy,blockX=bx,blockY=by,registers=reg))
  continue
 mode=path.stem.split('_')[1];phasegroups={p:[k for k in ks if p in k['ranges']] for p in ['forward','backward','update','transfer']};groups=collections.defaultdict(list)
 for k in ks:
  phase=next((p for p in phasegroups if p in k['ranges']),'outside');groups[(phase,k['name'])].append(k)
 for (phase,name),v in groups.items():profiles.append(dict(trace=path.stem,mode=mode,scope='GPU_kernel',phase=phase,name=name,calls=len(v),calls_per_step=len(v)/10,us_per_step=sum(k['us'] for k in v)/10))
 for p,v in phasegroups.items():
  rr=[r for r in rs if r['label']==p];cpu=sum(r['end']-r['start'] for r in rr)/10000
  pa=[a for a in api if any(r['start']<=a['start'] and a['end']<=r['end'] for r in rr)]
  hot.append(dict(trace=path.stem,mode=mode,phase=p,kernels_per_step=len(v)/10,gpu_kernel_us_per_step=sum(k['us'] for k in v)/10,cpu_range_elapsed_us_per_step=cpu,cuda_api_elapsed_us_per_step=sum(a['end']-a['start'] for a in pa)/10000,api_calls_per_step=len(pa)/10))
 ag=collections.defaultdict(list)
 for r in api:ag[ss[r['nameId']]].append((r['end']-r['start'])/1000)
 for name,v in ag.items():profiles.append(dict(trace=path.stem,mode=mode,scope='CUDA_API_CPU_elapsed',phase='',name=name,calls=len(v),calls_per_step=len(v)/10,us_per_step=sum(v)/10))
 capture_span=max(k['end'] for k in ks)-min(k['start'] for k in ks);busy=union_ns(ks)
 hot.append(dict(trace=path.stem,mode=mode,phase='ALL',kernels_per_step=len(ks)/10,gpu_kernel_us_per_step=sum(k['us'] for k in ks)/10,gpu_active_union_us_per_step=busy/10000,kernel_span_gaps_us_per_step=(capture_span-busy)/10000,cuda_api_elapsed_us_per_step=sum(a['end']-a['start'] for a in api)/10000,memcpy_calls_per_step=len(copies)/10,memcpy_bytes_per_step=sum(v['bytes'] for v in copies)/10))
 c.close()
write(O/'PROFILE_KERNELS.csv',profiles);write(O/'TRAINING_HOTSPOTS.csv',hot);write(O/'OPERATOR_KERNELS.csv',ops);write(O/'OPERATOR_KERNEL_SAMPLES.csv',opcalls)

print('profiles',len(profiles),'hotspots',len(hot),'layer kernels',len(ops))
