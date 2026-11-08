from pathlib import Path
import csv,json,math,statistics,collections,sqlite3,bisect
O=Path(__file__).resolve().parents[1]
def read(n):return list(csv.DictReader((O/n).open()))
def write(n,rows):
 if not rows:return
 with (O/n).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
runs=read('RUNS.csv');samples=read('GPU_SAMPLES.csv');cpu=read('CPU_SAMPLES.csv');steps=read('STEP_TIMINGS.csv')
formal=[r for r in runs if r['tag']=='formal'];baseline={r['repeat']:r for r in formal if r['mode']=='disabled'}
pairs=[];summary=[]
for mode in ['disabled','timing','resources','all']:
 rows=[r for r in formal if r['mode']==mode]
 for metric in ['wall_s','training_window_s']:
  vv=[float(r[metric]) for r in rows];dd=[float(r[metric])-float(baseline[r['repeat']][metric]) for r in rows];ratios=[(float(r[metric])/float(baseline[r['repeat']][metric])-1)*100 for r in rows]
  n=len(dd);se=statistics.stdev(dd)/math.sqrt(n);ci=2.200985*se
  summary.append(dict(mode=mode,metric=metric,repeats=n,mean_s=statistics.mean(vv),median_s=statistics.median(vv),std_s=statistics.stdev(vv),min_s=min(vv),max_s=max(vv),samples_per_s=768/statistics.mean(vv),paired_delta_mean_s=statistics.mean(dd),paired_delta_ci95_low_s=statistics.mean(dd)-ci,paired_delta_ci95_high_s=statistics.mean(dd)+ci,paired_overhead_mean_pct=statistics.mean(ratios),ratio_of_means_overhead_pct=(statistics.mean(vv)/statistics.mean(float(baseline[r['repeat']][metric]) for r in rows)-1)*100,ci_method='paired t df=11; sequential shared-server samples'))
  for r,delta,pct in zip(rows,dd,ratios):pairs.append(dict(mode=mode,metric=metric,repeat=r['repeat'],order=r['order'],wall_s=r[metric],baseline_s=baseline[r['repeat']][metric],delta_s=delta,overhead_pct=pct))
write('MONITOR_OVERHEAD.csv',summary);write('raw/OVERHEAD_PAIRS.csv',pairs)

stats=[]
for tag in ['functional','latency','trend','formal','formal_all','formal_timing','torch_reference']:
 for frame in ['hytorch','torch']:
  rr=[r for r in steps if r['tag']==('formal' if tag in ('formal_all','formal_timing') else tag) and r['framework']==frame and (tag not in ('formal_all','formal_timing') or r['monitor_components']==tag.split('_')[1])]
  if not rr:continue
  for key in ['host_iteration_s','step_wall_s','step_gpu_interval_s']+[p+s for p in ['data_load','data_prepare','h2d','forward','loss','backward','optimizer'] for s in ['_cpu_s','_gpu_event_s']]:
   vals=[float(r[key]) for r in rr if r.get(key)]
   if vals:stats.append(dict(tag=tag,framework=frame,metric=key,count=len(vals),mean_s=statistics.mean(vals),median_s=statistics.median(vals),std_s=statistics.stdev(vals),min_s=min(vals),max_s=max(vals)))
write('TIMING_SUMMARY.csv',stats)
res=[]
for tag in ['functional','trend','formal','torch_reference']:
 for source,rows,keys in [('NVML/CuPy/Torch',samples,['gpu_utilization_pct','memory_utilization_pct','gpu_used_bytes','gpu_total_bytes','gpu_temperature_c','gpu_power_w','sm_clock_mhz','memory_clock_mhz','process_gpu_used_bytes','cupy_active_bytes','cupy_reserved_bytes','torch_allocated_bytes','torch_reserved_bytes','schedule_delay_s','sample_collection_s']),('psutil',cpu,['system_cpu_pct','process_cpu_pct','process_rss_bytes','process_threads','schedule_delay_s','sample_collection_s'])]:
  rr=[r for r in rows if r['tag']==tag and r['inside_train']=='True']
  for key in keys:
   v=[float(r[key]) for r in rr if r.get(key)]
   res.append(dict(tag=tag,source=source,metric=key,samples=len(v),mean=statistics.mean(v) if v else None,min=min(v) if v else None,max=max(v) if v else None,scope='samples whose collection START is inside training; hardware itself has coarser device-defined window'))
write('RESOURCE_SUMMARY.csv',res)

# Current training Nsight Systems trace, independently collected without monitor events.
c=sqlite3.connect(O/'traces/training.sqlite');c.row_factory=sqlite3.Row
strings={r['id']:r['value'] for r in c.execute('select * from StringIds')};phase_names=['data_load','data_prepare','h2d','forward','loss','backward','optimizer'];phases=[];sranges=[]
for r in c.execute('select * from NVTX_EVENTS where end is not null'):
 name=r['text'] or strings.get(r['textId'],'')
 if name in phase_names:phases.append(dict(r,name=name))
 if name=='step':sranges.append(dict(r))
phases.sort(key=lambda r:r['start']);ps=[r['start'] for r in phases];sranges.sort(key=lambda r:r['start']);ss=[r['start'] for r in sranges]
apis={};apirows=[]
for row in c.execute('select * from CUPTI_ACTIVITY_KIND_RUNTIME'):
 r=dict(row);i=bisect.bisect_right(ps,r['start'])-1;j=bisect.bisect_right(ss,r['start'])-1
 phase=phases[i]['name'] if i>=0 and phases[i]['end']>=r['end'] else 'outside'
 step=j if j>=0 and sranges[j]['end']>=r['end'] else None
 r.update(name=strings[r['nameId']],phase=phase,step=step,us=(r['end']-r['start'])/1000);apis[r['correlationId']]=r;apirows.append(r)
ks=[];activities=[]
for row in c.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL'):
 r=dict(row);a=apis.get(r['correlationId'],{});r.update(name=strings[r['demangledName']],phase=a.get('phase','outside'),step=a.get('step'),us=(r['end']-r['start'])/1000)
 ks.append(r);activities.append(r)
tables={r[0] for r in c.execute("select name from sqlite_master where type='table'")}
for table in ['CUPTI_ACTIVITY_KIND_MEMCPY','CUPTI_ACTIVITY_KIND_MEMSET']:
 if table in tables:
  for row in c.execute('select * from '+table):
   r=dict(row);a=apis.get(r['correlationId'],{});r.update(name=table,phase=a.get('phase','outside'),step=a.get('step'),us=(r['end']-r['start'])/1000);activities.append(r)
def union_ns(rows):
 end=total=0
 for r in sorted(rows,key=lambda r:r['start']):total+=max(0,r['end']-max(end,r['start']));end=max(end,r['end'])
 return total
n=len(sranges);profiles=[]
for phase in phase_names:
 kk=[k for k in ks if k['phase']==phase];aa=[a for a in apirows if a['phase']==phase];pr=[p for p in phases if p['name']==phase]
 profiles.append(dict(phase=phase,steps=n,kernel_us_per_step=sum(k['us'] for k in kk)/n,kernels_per_step=len(kk)/n,cpu_nvtx_us_per_step=sum(p['end']-p['start'] for p in pr)/n/1000,cuda_api_us_per_step=sum(a['us'] for a in aa)/n,launch_api_us_per_step=sum(a['us'] for a in aa if 'LaunchKernel' in a['name'])/n,sync_api_us_per_step=sum(a['us'] for a in aa if 'Synchronize' in a['name'])/n))
write('PROFILE_PHASES.csv',profiles)
groups=collections.defaultdict(list)
for k in ks:groups[(k['phase'],k['name'])].append(k)
write('PROFILE_KERNELS.csv',[dict(phase=p,kernel=name,calls=len(v),calls_per_step=len(v)/n,total_us=sum(k['us'] for k in v),us_per_step=sum(k['us'] for k in v)/n) for (p,name),v in groups.items()])
inside=[a for a in activities if a['step'] is not None];span=max(a['end'] for a in inside)-min(a['start'] for a in inside);busy=union_ns(inside);kbusy=union_ns([k for k in ks if k['step'] is not None])
timeline=dict(steps=n,activity_span_us=span/1000,observed_kernel_copy_memset_busy_union_us=busy/1000,observed_gap_fraction=1-busy/span,kernel_busy_union_us=kbusy/1000,kernels_per_step=len([k for k in ks if k['step'] is not None])/n,interpretation='gaps in recorded kernel/copy/memset timeline, not SM occupancy; profiler overhead changes host scheduling')
(O/'raw/timeline.json').write_text(json.dumps(timeline,indent=2))
bottlenecks=[]
for p in profiles:
 cpurow=next(r for r in stats if r['tag']=='formal_all' and r['metric']==p['phase']+'_cpu_s')
 ev=next((r for r in stats if r['tag']=='formal_all' and r['metric']==p['phase']+'_gpu_event_s'),None)
 bottlenecks.append(dict(phase=p['phase'],monitored_cpu_mean_ms=cpurow['mean_s']*1000,monitored_gpu_event_mean_ms=ev['mean_s']*1000 if ev else None,nsys_kernel_us_per_step=p['kernel_us_per_step'],nsys_kernel_calls_per_step=p['kernels_per_step'],nsys_cuda_api_cpu_us_per_step=p['cuda_api_us_per_step'],source='formal full-monitor 2304 steps (no peak hook) + independent 20-step Nsight; not additive'))
write('BOTTLENECK_SUMMARY.csv',bottlenecks)
print('overhead',json.dumps(summary,indent=2));print('timeline',timeline)
