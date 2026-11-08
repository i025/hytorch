"""Parse separately profiled CUPTI events; never merge into formal wall timings."""
from pathlib import Path
import sqlite3,csv,collections,bisect,statistics,json,argparse
p=argparse.ArgumentParser();p.add_argument("--input",default="operators");p.add_argument("--prefix",default="");args=p.parse_args()
O=Path(__file__).resolve().parents[1]
def write(name,rows):
 if not rows:return
 target=O/(args.prefix+name);target.parent.mkdir(parents=True,exist_ok=True)
 with target.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
c=sqlite3.connect(O/('traces/'+args.input+'.sqlite'));c.row_factory=sqlite3.Row
ss={r['id']:r['value'] for r in c.execute('select * from StringIds')};ranges=[]
for r in c.execute('select * from NVTX_EVENTS where end is not null'):
 label=r['text'] or ss.get(r['textId'],'')
 if label.count('|')==4:
  op,case,scope,mode,rep=label.split('|');ranges.append(dict(r,op=op,case=case,scope=scope,mode=mode,repeat=int(rep),label=label))
ranges.sort(key=lambda v:v['start']);starts=[r['start'] for r in ranges];bylabel={r['label']:r for r in ranges};apis={};calls=collections.defaultdict(list)
for r in c.execute('select * from CUPTI_ACTIVITY_KIND_RUNTIME'):
 i=bisect.bisect_right(starts,r['start'])-1
 if i<0:continue
 rr=ranges[i]
 if rr['globalTid']==r['globalTid'] and r['end']<=rr['end']:
  a=dict(r,label=rr['label'],name=ss[r['nameId']]);apis[r['correlationId']]=a;calls[rr['label']].append(a)
kernels=collections.defaultdict(list);raw=[]
for r in c.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL'):
 a=apis.get(r['correlationId'])
 if a is None:continue
 rr=bylabel[a['label']];row={k:rr[k] for k in ('op','case','scope','mode','repeat')}
 row.update(kernel=ss[r['demangledName']],kernel_us=(r['end']-r['start'])/1000,gridX=r['gridX'],gridY=r['gridY'],gridZ=r['gridZ'],blockX=r['blockX'],blockY=r['blockY'],blockZ=r['blockZ'],registers_per_thread=r['registersPerThread'],stream_id=r['streamId'])
 raw.append(row);kernels[a['label']].append(row)
samples=[]
for label,r in bylabel.items():
 ks=kernels[label];aa=calls[label];row={k:r[k] for k in ('op','case','scope','mode','repeat')};row.update(kernel_us=sum(v['kernel_us'] for v in ks),kernel_count=len(ks),copy_named_kernels=sum('copy' in v['kernel'].lower() for v in ks),cuda_api_us=sum(v['end']-v['start'] for v in aa)/1000,cuda_api_calls=len(aa),launch_api_us=sum(v['end']-v['start'] for v in aa if 'LaunchKernel' in v['name'])/1000,cpu_nvtx_us=(r['end']-r['start'])/1000)
 samples.append(row)
write('raw/PROFILE_KERNEL_SAMPLES.csv',raw);write('raw/PROFILE_OPERATOR_SAMPLES.csv',samples)
groups=collections.defaultdict(list)
for r in samples:groups[tuple(r[k] for k in ('op','case','scope','mode'))].append(r)
summary=[]
for key,rs in groups.items():
 z=dict(zip(('op','case','scope','mode'),key));z['repeats']=len(rs)
 for metric in ('kernel_us','kernel_count','copy_named_kernels','cuda_api_us','cuda_api_calls','launch_api_us','cpu_nvtx_us'):
  vals=[r[metric] for r in rs];z[metric+'_mean']=statistics.mean(vals);z[metric+'_std']=statistics.stdev(vals) if len(vals)>1 else 0
 summary.append(z)
write('PROFILE_SUMMARY.csv',summary)
groups=collections.defaultdict(list)
for r in raw:groups[tuple(r[k] for k in ('op','case','scope','mode','kernel','gridX','gridY','gridZ','blockX','blockY','blockZ','registers_per_thread'))].append(r)
rows=[]
for key,rs in groups.items():
 z=dict(zip(('op','case','scope','mode','kernel','gridX','gridY','gridZ','blockX','blockY','blockZ','registers_per_thread'),key));z.update(calls=len(rs),total_kernel_us=sum(r['kernel_us'] for r in rs),mean_kernel_us=statistics.mean(r['kernel_us'] for r in rs));rows.append(z)
write('KERNEL_DETAILS.csv',rows)
write('DISPATCH_DECISIONS.csv',[r for r in rows if r['op']=='gemm' and r['scope']=='array' and r['mode']!='cupy'])
print('NVTX operators',len(samples),'CUDA kernels',len(raw),'summary groups',len(summary))
