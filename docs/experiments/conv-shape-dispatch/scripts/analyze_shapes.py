"""Analyze measured shapes; fit a conservative threshold using calibration only."""
import argparse
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

import numpy as np
from cases import cases


def read(path):
    with path.open() as f:return list(csv.DictReader(f))


def write(path,rows):
    with path.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--split',choices=['calibration','held-out','validation-2','validation-final'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--policy',type=Path,help='Frozen calibration policy; held-out validation only')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    assert json.loads((a.raw/'completion.json').read_text())['status']=='COMPLETED'
    configs=cases(a.split);timings,errors,memory,processes=[],[],[],[]
    methods=['original','fused','torch']+(['auto'] if (a.raw/'auto-r1').exists() else [])
    for repeat in range(1,4):
        for method in methods:
            folder=a.raw/(method+'-r'+str(repeat));config=json.loads((folder/'config.json').read_text())
            assert config['split']==a.split and not config['nvtx']
            local=read(folder/'latencies.csv');timings.extend(local)
            ee=read(folder/'errors.csv');errors.extend(ee);memory.extend(read(folder/'memory.csv'))
            assert all(int(e['mismatches'])==0 and e['finite']=='True' for e in ee)
            for case in configs:
                name=case['case']
                with np.load(folder/(name+'-arrays.npz')) as actual,np.load(a.raw/'original-r1'/(name+'-arrays.npz')) as base:
                    assert np.array_equal(actual['weight'],base['weight']) and np.array_equal(actual['dy'],base['dy'])
                for metric in ('wall_us','event_us','host_submit_us','end_wait_us'):
                    values=np.array([float(r[metric]) for r in local if r['case']==name])
                    processes.append(dict(case=name,method=method,repeat=repeat,metric=metric,
                        mean=float(values.mean()),std=float(values.std(ddof=1)),median=float(np.median(values)),
                        p95=float(np.percentile(values,95)),count=len(values)))
    summaries=[];comparisons=[]
    for case in configs:
        means={}
        for method in methods:
            for metric in ('wall_us','event_us','host_submit_us','end_wait_us'):
                values=[r['mean'] for r in processes if (r['case'],r['method'],r['metric'])==(case['case'],method,metric)]
                row=dict(case=case['case'],method=method,metric=metric,work=case['work'],
                    mean=float(np.mean(values)),std=float(np.std(values,ddof=1)),minimum=min(values),maximum=max(values))
                summaries.append(row)
                if metric=='wall_us':means[method]=row
        original,fused=means['original'],means['fused']
        ratio=original['mean']/fused['mean']
        comparisons.append(dict(case=case['case'],split=a.split,work=case['work'],
            original_wall_us=original['mean'],fused_wall_us=fused['mean'],torch_wall_us=means['torch']['mean'],
            original_std=original['std'],fused_std=fused['std'],original_minimum=original['minimum'],
            fused_maximum=fused['maximum'],wall_speedup=ratio,gain_pct=(ratio-1)*100,
            calibration_safe=bool(ratio>=1.10 and fused['maximum']<original['minimum'])))
        if 'auto' in means:
            variants={r['selected'] for r in timings if r['case']==case['case'] and r['method']=='auto'}
            assert len(variants)==1
            comparisons[-1].update(auto_wall_us=means['auto']['mean'],auto_std=means['auto']['std'],
                actual_auto_selected=variants.pop(),actual_auto_speedup=original['mean']/means['auto']['mean'])
    for name,rows in [('latencies',timings),('errors',errors),('memory',memory),('process-summary',processes),
                      ('summary',summaries),('comparisons',comparisons)]:write(a.output/(name+'.csv'),rows)
    if a.split=='calibration':
        assert a.policy is None
        ordered=sorted(comparisons,key=lambda r:r['work']);threshold=None
        for work in sorted({r['work'] for r in ordered}):
            group=[r for r in ordered if r['work']==work]
            if not all(r['calibration_safe'] for r in group):break
            threshold=work
        dims=np.array([(*c['shape'],c['weight_shape'][0]) for c in configs])
        result=dict(status='CANDIDATE' if threshold is not None else 'INSUFFICIENT_EVIDENCE',
            frozen_at=datetime.now(timezone.utc).isoformat(),calibration_raw=str(a.raw.resolve()),
            feature='N*Cin*H*W*Cout*Kh*Kw/(Sh*Sw)',threshold=threshold,
            minimum_gain_pct=10.,calibration_selected=sum(r['work']<=threshold for r in ordered) if threshold else 0,
            domain=dict(minimum=dims.min(axis=0).tolist(),maximum=dims.max(axis=0).tolist(),
                dimensions=['N','Cin','H','W','Cout'],kernels=[1,3,5],strides=[1,2],padding=[0,1],
                dtype='float32',layout='contiguous NCHW/OIHW',groups=1,dilation=1),
            held_out_used_for_selection=False)
        with (a.output/'frozen-policy.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    else:
        policy=json.loads(a.policy.read_text());threshold=policy['threshold']
        decisions=[]
        shapes={r['case']:r for r in configs};domain=policy['domain']
        for row in comparisons:
            c=shapes[row['case']];dims=(*c['shape'],c['weight_shape'][0])
            covered=all(lo<=v<=hi for lo,v,hi in zip(domain['minimum'],dims,domain['maximum']))
            serial=c['weight_shape'][0]*c['weight_shape'][2]*c['weight_shape'][3]
            serial_ok=serial<=policy['serial_limit'] if policy.get('version')==2 else True
            if policy.get('version')==3:
                covered=covered and c['weight_shape'][2:]==(1,1) and c['stride']==1 and c['pad']==0
            selected='fused' if covered and serial_ok and threshold is not None and row['work']<=threshold else 'original'
            if 'actual_auto_selected' in row:assert row['actual_auto_selected']==selected
            speedup=row.get('actual_auto_speedup',row['wall_speedup'])
            decisions.append(dict(row,selected=selected,covered=covered,serial=serial,
                passes=selected=='original' or speedup>=1.05))
        write(a.output/'decisions.csv',decisions)
        result=dict(status='VALIDATED' if all(r['passes'] for r in decisions) and
            sum(r['selected']=='fused' for r in decisions)>=4 and any(r['selected']=='original' for r in decisions)
            else 'NOT_VALIDATED',policy=str(a.policy.resolve()),threshold=threshold,
            selected_fused=sum(r['selected']=='fused' for r in decisions),cases=len(decisions),
            no_rule_reselection=True,failed_cases=[r['case'] for r in decisions if not r['passes']])
    with (a.output/'analysis.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
