from pathlib import Path
import csv,json,statistics,collections,math
O=Path(__file__).resolve().parents[1]
def read(n):return list(csv.DictReader((O/n).open()))
def write(n,rows):
 with (O/n).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
rs=read('RESULTS_ALL.csv');raw=read('RAW_TIMINGS.csv');acc=read('ACCURACY.csv');g=collections.defaultdict(dict)
for r in rs:g[r['op'],r['case'],r['scope'],r['metric']][r['mode']]=r
speed=[]
for (op,case,scope,metric),rr in g.items():
 if 'native' not in rr:continue
 for old in ['numpy','cupy','shared_old','register_old']:
  if old not in rr:continue
  # Paired per-repeat ratios/changes, using same metric and loop-normalized time.
  aa={int(r['repeat']):float(r[metric]) for r in raw if r['op']==op and r['case']==case and r['scope']==scope and r['mode']==old and r.get(metric)}
  bb={int(r['repeat']):float(r[metric]) for r in raw if r['op']==op and r['case']==case and r['scope']==scope and r['mode']=='native' and r.get(metric)}
  ds=[aa[i]-bb[i] for i in aa.keys()&bb.keys()];ratios=[aa[i]/bb[i] for i in aa.keys()&bb.keys()];n=len(ds);se=statistics.stdev(ds)/math.sqrt(n) if n>1 else 0
  speed.append(dict(op=op,case=case,scope=scope,metric=metric,baseline=old,new='native',baseline_mean_us=rr[old]['mean'],native_mean_us=rr['native']['mean'],ratio_of_means=float(rr[old]['mean'])/float(rr['native']['mean']),paired_ratio_median=statistics.median(ratios),paired_delta_mean_us=statistics.mean(ds),paired_delta_ci95_low_us=statistics.mean(ds)-2.12*se,paired_delta_ci95_high_us=statistics.mean(ds)+2.12*se,repeats=n,ci_note='approximate t interval; sequential shared-server samples not IID guarantee'))
write('SPEEDUP_SUMMARY.csv',speed)
aggregate=[]
for scope in ('array','function'):
 for op in sorted(set(r['op'] for r in rs)):
  row=dict(op=op,scope=scope,aggregation='equal case weight, arithmetic means of sync-complete means')
  for mode in ('numpy','cupy','native'):
   vs=[float(r['mean']) for r in rs if r['op']==op and r['scope']==scope and r['metric']=='sync_us' and r['mode']==mode];row[mode+'_us']=statistics.mean(vs);row['case_count']=len(vs)
  row['native_vs_numpy']=row['numpy_us']/row['native_us'];row['native_vs_cupy']=row['cupy_us']/row['native_us'];aggregate.append(row)
write('OPERATOR_AGGREGATE.csv',aggregate)
print(json.dumps(aggregate,indent=2));print('accuracy max',max(float(r['max_abs']) for r in acc),'rel',max(float(r['relative_l2']) for r in acc),'fails',sum(int(r['mixed_fail_count']) for r in acc))
