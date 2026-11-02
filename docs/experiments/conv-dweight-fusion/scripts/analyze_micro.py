"""Join ordinary measurements with separate correlated CUPTI durations."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'docs/experiments/fused-sgd-cuda/scripts'))
from analyze_fused_results import csv_rows,write_csv,statistics,arrays,identical
from cases import cases


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,required=True);p.add_argument('--analysis',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    timings,errors,memory,processes,trace=[],[],[],[],[]
    for repeat in range(1,4):
        for method in ('original','fused','torch'):
            folder=a.raw/f'micro-{method}-r{repeat}'
            data=csv_rows(folder/'latencies.csv');timings.extend(data)
            ee=csv_rows(folder/'errors.csv');errors.extend(ee);memory.extend(csv_rows(folder/'memory.csv'))
            assert all(int(r['mismatches'])==0 and r['finite']=='True' for r in ee)
            gpu=csv_rows(a.analysis/f'trace-micro-{method}-r{repeat}'/'micro-kernels.csv')
            trace.extend(dict(r,method=method,repeat=repeat) for r in gpu)
            for c in cases():
                key=c['case'];assert identical(arrays(folder/(key+'-inputs.npz')),arrays(a.raw/'micro-original-r1'/(key+'-inputs.npz')))
                for metric in ('event_us','wall_us','host_submit_us','end_wait_us','kernel_us'):
                    source=gpu if metric=='kernel_us' else data
                    values=[float(r[metric]) for r in source if r['case']==key];assert len(values)==30
                    processes.append(dict(case=key,method=method,repeat=repeat,metric=metric,**statistics(values)))
    stats=[]
    for c in cases():
        for method in ('original','fused','torch'):
            for metric in ('event_us','wall_us','host_submit_us','end_wait_us','kernel_us'):
                stats.append(dict(case=c['case'],method=method,metric=metric,**statistics([r['mean'] for r in processes if
                    (r['case'],r['method'],r['metric'])==(c['case'],method,metric)])))
    comparisons=[]
    for c in cases():
        row=dict(c)
        for method in ('original','fused','torch'):
            for metric in ('kernel_us','wall_us'):
                row[method+'_'+metric]=next(r['mean'] for r in stats if (r['case'],r['method'],r['metric'])==(c['case'],method,metric))
        row['gpu_original_over_fused']=row['original_kernel_us']/row['fused_kernel_us']
        row['wall_original_over_fused']=row['original_wall_us']/row['fused_wall_us'];comparisons.append(row)
    for name,rows in [('latencies',timings),('errors',errors),('memory',memory),('process-summary',processes),('summary',stats),
        ('trace-calls',trace),('comparisons',comparisons)]:write_csv(a.output/(name+'.csv'),rows)
    result=dict(status='PASSED',cases=len(cases()),numeric_comparisons=len(errors),max_abs=max(float(r['max_abs']) for r in errors),
        kernel_faster=sum(r['gpu_original_over_fused']>1 for r in comparisons),kernel_slower=sum(r['gpu_original_over_fused']<1 for r in comparisons),
        wall_faster=sum(r['wall_original_over_fused']>1 for r in comparisons),wall_slower=sum(r['wall_original_over_fused']<1 for r in comparisons),
        scopes='Nsight CUPTI vs ordinary synchronized wall, separate processes in one physical GPU window')
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
