from pathlib import Path
import csv,json,statistics as st,math,collections
O=Path(__file__).resolve().parents[1]
def read(n):return list(csv.DictReader((O/n).open()))
def write(n,rows):
 if not rows:return
 with (O/n).open('w') as f:w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def stats(v):return dict(median=st.median(v),mean=st.mean(v),sd=st.stdev(v) if len(v)>1 else 0,min=min(v),max=max(v))
raw=read('raw/LAYER_TIMINGS_balanced18.csv')+read('raw/LAYER_TIMINGS_actual18.csv');ks=read('OPERATOR_KERNEL_SAMPLES.csv');kernelrows=read('OPERATOR_KERNELS.csv')
alloc=read('raw/LAYER_ALLOCATIONS_balanced18.csv')+read('raw/LAYER_ALLOCATIONS_actual18.csv');out=[]
for key in dict.fromkeys((r['case'],r['implementation']) for r in raw):
 rs=[r for r in raw if (r['case'],r['implementation'])==key];r={k:rs[0][k] for k in ['case','category','level','implementation','geometry']};r['n_unprofiled']=len(rs)
 for col in ['submit_us','sync_us']:
  for k,v in stats([float(x[col]) for x in rs]).items():r[col+'_'+k]=v
 pp=[x for x in ks if (x['case'],x['implementation'])==key];r['n_cupti']=len(pp)
 if pp:
  for k,v in stats([float(x['gpu_kernel_us']) for x in pp]).items():r['gpu_kernel_us_'+k]=v
  r['kernels_per_call']=st.mean(float(x['kernel_count']) for x in pp)
  for name in ['cuda_api_elapsed_us','cuda_api_calls','launch_api_elapsed_us','sync_api_elapsed_us']:
   r['profiled_'+name+'_median']=st.median(float(x[name]) for x in pp)
 aa=[x for x in alloc if (x['case'],x['implementation'])==key][0];r.update({k:v for k,v in aa.items() if k not in ['case','implementation']});out.append(r)
write('LAYER_COMPARISON.csv',out)
write('CONV_BACKWARD_COMPARISON.csv',[r for r in out if r['category'] in ['ConvBackward','dInput','dWeight','dBias']])
write('POOLING_COMPARISON.csv',[r for r in out if r['category'] in ['PoolForward','PoolBackward']])
# Both new paths and paired difference CI. Independent synchronized calls are NOT subtracted from submit calls.
paired=[]
for case in dict.fromkeys(r['case'] for r in raw):
 for col in ['submit_us','sync_us']:
  b={int(r['repeat']):float(r[col]) for r in raw if r['case']==case and r['implementation']=='B0'};o={int(r['repeat']):float(r[col]) for r in raw if r['case']==case and r['implementation']=='O1'};d=[b[i]-o[i] for i in b];mu=st.mean(d);half=2.109815578*st.stdev(d)/math.sqrt(len(d))
  paired.append(dict(case=case,metric=col,n=len(d),saved_mean_us=mu,ci95_low_us=mu-half,ci95_high_us=mu+half,wins=sum(x>0 for x in d)))
write('LOCAL_PAIRED.csv',paired)
cpu=[dict(scope='unprofiled_layer',**r) for r in out]
for key in dict.fromkeys((r['case'],r['component']) for r in read('raw/CPU_COMPONENTS.csv')):
 v=[float(r['ns_per_call'])/1000 for r in read('raw/CPU_COMPONENTS.csv') if (r['case'],r['component'])==key]
 cpu.append(dict(scope='unprofiled_metadata_component',case=key[0],component=key[1],n=len(v),submit_us_median=st.median(v),submit_us_sd=st.stdev(v),note='1000 calls per timing batch; metadata components overlap full call, do not sum'))
for mode in ['B0','O1']:
 for r in read('raw/cpu_'+mode+'.csv'):cpu.append(dict(scope='training_cProfile_30steps',implementation=mode,**r))
write('CPU_OVERHEAD.csv',cpu)
# Keep every trace separate, and add aggregate over post-change b/c captures only.
hot=read('TRAINING_HOTSPOTS.csv');summ=[]
for mode in ['B0','O1','torch']:
 for phase in ['forward','backward','update','transfer','ALL']:
  rs=[r for r in hot if r['mode']==mode and r['phase']==phase and r['trace'].endswith(('_b','_c'))]
  row=dict(scope='training_10step_capture_mean',mode=mode,phase=phase,captures=len(rs))
  for key in rs[0]:
   if key in ['trace','mode','phase']:continue
   vs=[float(r[key]) for r in rs if r[key]!='']
   if vs:row[key]=st.mean(vs)
  summ.append(row)
write('PROFILE_SUMMARY.csv',summ)
write('PROFILE_BEFORE_AFTER.csv',[dict(scope='training_NVTX_CUPTI',**r) for r in hot]+[dict(scope='layer_CUPTI',**r) for r in kernelrows]+[dict(measurement='training_API_and_kernels',**r) for r in read('PROFILE_KERNELS.csv')])
# Do not mutate the raw memory observations.
mem=read('MEMORY_COMPARISON.csv')
if not (O/'raw/MEMORY_CUPY.csv').exists():write('raw/MEMORY_CUPY.csv',mem)
mem=read('raw/MEMORY_CUPY.csv')
for r in mem:r['allocator']='CuPy MemoryHook';r['scope']='reset + training + result collection; not formal timing'
for r in read('raw/MEMORY_TORCH.csv'):r['allocator']='PyTorch allocator history';mem.append(r)
write('MEMORY_COMPARISON.csv',mem)
summary=json.loads((O/'SUMMARY.json').read_text())
idx={(r['case'],r['implementation']):r for r in out}
local={}
for label,names in [('conv_real',[f'conv{i}/backward' for i in range(1,6)]),('conv_all',[s+'/backward' for s in ['conv1','conv2','conv3','conv4','conv5','irregular','asymmetric']]),('pool_main_window',[s+'/forward' for s in ['tiny_pool','irregular_pool','overlap_pool','wide_pool']]),('pool_actual_supplement',['tiny_actual_pool/forward'])]:
 local[label]={}
 for metric in ['submit_us_median','sync_us_median','gpu_kernel_us_median']:
  local[label][metric]=dict(reduction_pct_median=st.median(100*(1-idx[n,'O1'][metric]/idx[n,'B0'][metric]) for n in names),B0_over_torch_median=st.median(idx[n,'B0'][metric]/idx[n,'torch'][metric] for n in names),O1_over_torch_median=st.median(idx[n,'O1'][metric]/idx[n,'torch'][metric] for n in names))
summary['local']=local;(O/'SUMMARY.json').write_text(json.dumps(summary,indent=2));print(json.dumps(local,indent=2))
