"""Derive reproducible summaries from retained sample values, without retesting."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--gpu',type=Path);args=p.parse_args()
root=args.results
cpu=json.loads((root/'completed-03/cpu-summary.json').read_text())
raw=list(csv.DictReader((root/'samples.csv').open()))
methods=sorted({r['strategy'] for r in raw});paired=[];cancellation=[]
for n in (1024,2048,4096):
    chosen=[r for r in raw if int(r['n'])==n]
    bymethod={method:{(r['row'],r['col']):r for r in chosen if r['strategy']==method} for method in methods}
    for stratum in ('all','historical_worst','near_zero','random_normal'):
        keys=[key for key,row in bymethod['sequential_fma'].items() if stratum=='all' or row['stratum']==stratum]
        seq=[bymethod['sequential_fma'][key] for key in keys]
        cond=np.array([float(r['cancellation_condition']) for r in seq]);rel=np.array([float(r['relative_error_fp64']) for r in seq])
        cancellation.append(dict(n=n,stratum=stratum,samples=len(seq),median_condition=float(np.median(cond)),
            max_condition=float(np.max(cond)),sequential_fail_original=sum(r['pass_original_numpy']=='False' for r in seq),
            log_condition_relative_error_correlation=float(np.corrcoef(np.log10(cond),np.log10(rel+1e-300))[0,1])))
        for method in methods:
            actual=[bymethod[method][key] for key in keys]
            a=np.array([float(r['abs_error_fp64']) for r in actual]);b=np.array([float(r['abs_error_fp64']) for r in seq])
            paired.append(dict(n=n,stratum=stratum,strategy=method,samples=len(keys),
                mean_abs_reduction_vs_sequential=float(np.mean(b)/np.mean(a)),rms_reduction_vs_sequential=float(np.sqrt(np.mean(b*b)/np.mean(a*a))),
                better_than_sequential=int(np.count_nonzero(a<b)),equal_to_sequential=int(np.count_nonzero(a==b)),worse_than_sequential=int(np.count_nonzero(a>b)),
                closer_to_numpy_but_worse_fp64=sum(float(r['abs_error_numpy32'])<float(s['abs_error_numpy32']) and float(r['abs_error_fp64'])>float(s['abs_error_fp64']) for r,s in zip(actual,seq))))
summary=dict(cpu=cpu,paired_comparisons=paired,cancellation=cancellation,gpu_status='NOT_RUN')
if args.gpu:
    summary['gpu']=json.loads((args.gpu/'gpu-summary.json').read_text());summary['gpu_status']='EXECUTED'
    summary['full_matrix_gpu_comparisons']=json.loads((args.gpu/'full-matrix-comparisons.json').read_text())
    # Exact sampled CPU FMA/GPU comparisons are additional to full-matrix GPU-vs-GPU comparisons.
    gpu=list(csv.DictReader((args.gpu/'gpu-samples.csv').open()));sample_cross=[]
    for n in (1024,2048,4096):
        fma={(r['row'],r['col']):float(r['value']) for r in raw if int(r['n'])==n and r['strategy']=='sequential_fma'}
        for method in sorted({r['strategy'] for r in gpu}):
            selected=[r for r in gpu if int(r['n'])==n and r['strategy']==method]
            diff=[abs(float(r['value'])-fma[(r['row'],r['col'])]) for r in selected]
            sample_cross.append(dict(n=n,strategy=method,samples=len(diff),unequal_to_sequential_fma=sum(d!=0 for d in diff),max_abs_difference=max(diff)))
    summary['sampled_gpu_vs_cpu_fma']=sample_cross
out=root/('summary.json' if args.gpu else 'cpu-analysis.json')
with out.open('x') as f:json.dump(summary,f,indent=2,allow_nan=False)
for row in cpu['groups']:
    if row['stratum']=='all':print(json.dumps(row),flush=True)
print('Summary:',out)
