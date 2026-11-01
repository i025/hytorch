"""Join measured scopes for the report; never combine their timing denominators."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[4]
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'docs/experiments/fused-sgd-cuda/scripts'))
from analyze_fused_results import csv_rows,write_csv,statistics,json_file
from cases import cases


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profiles',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    pure=csv_rows(a.profiles/'pure-kernel-comparisons.csv')
    wall=csv_rows(BASE/'results/micro-01/comparisons.csv')
    timings=csv_rows(BASE/'results/micro-01/summary.csv')
    merged=[]
    for c in cases('history')+cases('fresh'):
        name=c['case'];gpu=next(r for r in pure if r['case']==name)
        full=next(r for r in wall if r['case']==name)
        row=dict(c,split=full['split'])
        row.update({k:float(v) for k,v in gpu.items() if k not in ('case','split','separated_5pct')})
        row.update({k:float(v) for k,v in full.items() if k not in ('case','split','shape','weight_shape')})
        for method in ('fused','fused_opt'):
            for metric in ('host_submit_us','end_wait_us'):
                row[method+'_'+metric]=next(float(r['mean']) for r in timings if
                    (r['case'],r['scope'],r['method'],r['metric'])==(name,'full',method,metric))
        merged.append(row)
    write_csv(a.output/'all-shapes.csv',merged)
    distribution=[]
    for split in ('history','fresh','all'):
        rr=[r for r in merged if split=='all' or r['split']==split]
        for key in ('gpu_old_over_opt','wall_old_over_opt','wall_cupy_over_opt'):
            values=[r[key] for r in rr]
            distribution.append(dict(split=split,metric=key,faster=sum(v>1 for v in values),
                slower=sum(v<1 for v in values),**statistics(values)))
    write_csv(a.output/'shape-distribution.csv',distribution)
    phases=[]
    for r in csv_rows(a.profiles/'trace-phases.csv'):
        count=int(r['nvtx_count'])
        if r['phase'] not in ('forward','backward','update'):continue
        row=dict(model=r['model'],method=r['method'],phase=r['phase'],steps=count)
        for key in ('host_nvtx_sum_ms','launch_api_ms','sync_api_ms','allocation_api_ms',
                    'gpu_kernel_sum_ms','gpu_memcpy_sum_ms'):
            row[key.replace('_ms','_us_per_step')]=float(r[key])*1000/count
        for key in ('gpu_kernel_count','launch_api_count','gpu_memcpy_count'):
            row[key+'_per_step']=int(r[key])/count
        phases.append(row)
    write_csv(a.output/'trace-phases-per-step.csv',phases)
    components=csv_rows(a.profiles/'gradient-components.csv');parts=[]
    for model in ('donkey','spatial'):
        for method in 'AFH':
            for layer in sorted({r['layer'] for r in components if r['model']==model}):
                for part in ('dInput','dWeight','dBias'):
                    rr=[r for r in components if (r['model'],r['method'],r['layer'],r['part'])==(model,method,layer,part)]
                    if not rr:continue
                    parts.append(dict(model=model,method=method,layer=layer,part=part,calls=len(rr),
                        kernel_us=np.mean([float(r['kernel_us']) for r in rr]),
                        host_us=np.mean([float(r['host_us']) for r in rr]),
                        kernels=np.mean([int(r['kernels']) for r in rr])))
    write_csv(a.output/'trace-components-per-call.csv',parts)
    intervals=csv_rows(a.profiles/'backward-windows.csv');windows=[]
    for model in ('donkey','spatial'):
        for method in 'AFH':
            rr=[r for r in intervals if (r['model'],r['method'])==(model,method)]
            windows.append(dict(model=model,method=method,steps=len(rr),**{key:float(np.mean([float(r[key]) for r in rr]))
                for key in ('host_us','kernel_span_us','kernel_union_us','between_own_kernels_us')}))
    write_csv(a.output/'trace-backward-intervals.csv',windows)
    result=dict(profiles=json_file(a.profiles/'analysis.json'),
        training_windows={folder:json_file(BASE/'results'/folder/'analysis.json') for folder in ('training-01','training-02')},
        shape_distributions=distribution,
        timing_scope='Each CSV scope is independent; trace gaps are between kernels attributed to this run, not SM idle or utilization',
        raw_profiles=str(a.profiles.resolve()))
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
