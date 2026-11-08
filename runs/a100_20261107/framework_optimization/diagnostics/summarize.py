from pathlib import Path
import csv,json,statistics,math
O=Path(__file__).resolve().parents[1]
def read(n):return list(csv.DictReader((O/n).open()))
def write(n,rows):
 with (O/n).open('w') as f:w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
r=read('TRAINING_ABLATION.csv');stats={};pairs=[]
for mode in ['B0','F1','torch']:
 rr=[x for x in r if x['mode']==mode];v=[float(x['wall_s']) for x in rr];stats[mode]=dict(n=len(v),mean_s=statistics.mean(v),median_s=statistics.median(v),sd_s=statistics.stdev(v),samples_per_s=statistics.mean(float(x['samples_per_s']) for x in rr))
b={int(x['repeat']):float(x['wall_s']) for x in r if x['mode']=='B0'};f={int(x['repeat']):float(x['wall_s']) for x in r if x['mode']=='F1'}
d=[b[i]-f[i] for i in sorted(b)];mu=statistics.mean(d);half=2.20098516*statistics.stdev(d)/math.sqrt(len(d))
stats['paired']=dict(delta_mean_s=mu,ci95_s=[mu-half,mu+half],wins=sum(x>0 for x in d),wall_reduction_pct=100*mu/stats['B0']['mean_s'],speedup=stats['B0']['mean_s']/stats['F1']['mean_s'],before_vs_torch=stats['B0']['mean_s']/stats['torch']['mean_s'],after_vs_torch=stats['F1']['mean_s']/stats['torch']['mean_s'])
write('PAIRED.csv',[dict(repeat=i,B0_s=b[i],F1_s=f[i],saved_s=b[i]-f[i]) for i in sorted(b)])
sups=[];r=read('SUPPLEMENT.csv')
for kind,batch in dict.fromkeys((x['model'],x['batch']) for x in r):
 rr=[x for x in r if x['model']==kind and x['batch']==batch];bb={x['repeat']:float(x['wall_s']) for x in rr if x['mode']=='B0'};ff={x['repeat']:float(x['wall_s']) for x in rr if x['mode']=='F1'}
 sups.append(dict(model=kind,batch=batch,B0_s=statistics.mean(bb.values()),F1_s=statistics.mean(ff.values()),reduction_pct=100*(1-statistics.mean(ff.values())/statistics.mean(bb.values())),wins=sum(bb[i]>ff[i] for i in bb),n=len(bb)))
stats['supplement']=sups
for name in ['TRAINING_ERRORS.csv','SUPPLEMENT_ERRORS.csv']:
 es=read(name);stats[name]=dict(failed_elements=sum(int(x['failed']) for x in es),max_abs=max(float(x['max_abs']) for x in es),F1_max_abs=max(float(x['max_abs']) for x in es if x['mode']=='F1'))
# Baseline immutable results, distinct window segment, not formal acceptance denominator.
a=read('stage_a/TRAINING_ABLATION.csv');stats['stage_a']={m:statistics.mean(float(x['wall_s']) for x in a if x['mode']==m) for m in ['B0','torch']}
(O/'SUMMARY.json').write_text(json.dumps(stats,indent=2));print(json.dumps(stats,indent=2))
