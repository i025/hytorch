"""Preserve all per-call values and summarize independent process means."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'docs/experiments/fused-sgd-cuda/scripts'))
from analyze_fused_results import csv_rows,write_csv,statistics,arrays,identical,json_file
from cases import cases


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    assert json_file(a.raw/'completion.json')['status']=='COMPLETED'
    timings,memory,errors,processes=[],[],[],[]
    for scope in ('full','kernel'):
        methods=['original','fused','fused_opt','torch'] if scope=='full' else ['fused','fused_opt']
        for split in ('history','fresh'):
            for repeat in range(1,4):
                for method in methods:
                    path=a.raw/f'{scope}-{split}-{method}-r{repeat}'
                    assert not json_file(path/'config.json')['nvtx']
                    local=csv_rows(path/'latencies.csv');timings.extend(local)
                    memory.extend(dict(r,split=split) for r in csv_rows(path/'memory.csv'))
                    ee=csv_rows(path/'errors.csv');errors.extend(dict(r,split=split) for r in ee)
                    assert all(int(r['mismatches'])==0 and r['finite']=='True' for r in ee)
                    for case in cases(split):
                        name=case['case'];base=a.raw/f'full-{split}-original-r1'
                        assert identical(arrays(path/(name+'-inputs.npz')),arrays(base/(name+'-inputs.npz')))
                        for metric in ('event_us','wall_us','host_submit_us','end_wait_us'):
                            processes.append(dict(case=name,split=split,method=method,scope=scope,repeat=repeat,metric=metric,
                                **statistics([float(r[metric]) for r in local if r['case']==name])))
    aggregate=[]
    for case in cases('history')+cases('fresh'):
        split='fresh' if case['case'].startswith('o') else 'history'
        for scope in ('full','kernel'):
            for method in (['original','fused','fused_opt','torch'] if scope=='full' else ['fused','fused_opt']):
                for metric in ('event_us','wall_us','host_submit_us','end_wait_us'):
                    rr=[r['mean'] for r in processes if (r['case'],r['scope'],r['method'],r['metric'])==(case['case'],scope,method,metric)]
                    aggregate.append(dict(case=case['case'],split=split,scope=scope,method=method,metric=metric,**statistics(rr)))
    comparisons=[]
    def mean(case,method,metric):return next(r['mean'] for r in aggregate if (r['case'],r['scope'],r['method'],r['metric'])==(case,'full',method,metric))
    for c in cases('history')+cases('fresh'):
        name=c['case'];row=dict(case=name,split='fresh' if name.startswith('o') else 'history',shape=str(c['shape']),weight_shape=str(c['weight_shape']))
        for method in ('original','fused','fused_opt','torch'):
            row[method+'_wall_us']=mean(name,method,'wall_us');row[method+'_event_span_us']=mean(name,method,'event_us')
        row['wall_old_over_opt']=row['fused_wall_us']/row['fused_opt_wall_us']
        row['wall_cupy_over_opt']=row['original_wall_us']/row['fused_opt_wall_us']
        comparisons.append(row)
    for name,data in [('latencies',timings),('memory',memory),('errors',errors),('process-summary',processes),('summary',aggregate),('comparisons',comparisons)]:write_csv(a.output/(name+'.csv'),data)
    result=dict(status='PASSED',cases=len(comparisons),numeric_comparisons=len(errors),max_abs=max(float(r['max_abs']) for r in errors),
        full_wall_faster=sum(r['wall_old_over_opt']>1 for r in comparisons),full_wall_slower=sum(r['wall_old_over_opt']<1 for r in comparisons),
        scope='Full dInput host synchronized wall; pure kernel times come from separate Nsight activity records')
    (a.output/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
