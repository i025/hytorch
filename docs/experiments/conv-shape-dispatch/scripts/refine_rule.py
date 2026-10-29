"""Fit one two-condition candidate on already-observed shapes, not validation-2."""
import argparse
import csv
from datetime import datetime,timezone
import json
from pathlib import Path

import numpy as np
from cases import cases

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--calibration',type=Path,required=True)
p.add_argument('--previous-validation',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
data=[]
for split,folder in [('calibration',a.calibration),('held-out',a.previous_validation)]:
    config={r['case']:r for r in cases(split)}
    with (folder/'comparisons.csv').open() as f:
        for row in csv.DictReader(f):
            case=config[row['case']];co,ci,kh,kw=case['weight_shape']
            data.append(dict(case=case['case'],work=float(row['work']),serial=co*kh*kw,
                safe=row['calibration_safe']=='True',saving=float(row['original_wall_us'])-float(row['fused_wall_us'])))
candidates=[]
for serial_limit in sorted({r['serial'] for r in data}):
    eligible=[r for r in data if r['serial']<=serial_limit];threshold=None
    for work in sorted({r['work'] for r in eligible}):
        if not all(r['safe'] for r in eligible if r['work']==work):break
        threshold=work
    selected=[r for r in eligible if threshold is not None and r['work']<=threshold]
    candidates.append(dict(serial_limit=serial_limit,threshold=threshold,selected=len(selected),
        total_saved_us=sum(r['saving'] for r in selected)))
best=max(candidates,key=lambda r:(r['selected'],r['total_saved_us'],-r['serial_limit']))
dims=np.array([(*r['shape'],r['weight_shape'][0]) for s in ('calibration','held-out') for r in cases(s)])
policy=dict(status='CANDIDATE',version=2,frozen_at=datetime.now(timezone.utc).isoformat(),
    threshold=best['threshold'],serial_limit=best['serial_limit'],selected_training_cases=best['selected'],
    training_cases=52,training_sources=[str(a.calibration),str(a.previous_validation)],
    independent_split='validation-2 (24 unseen cases)',independent_results_used=False,
    feature='N*Cin*H*W*Cout*Kh*Kw/(Sh*Sw)',serial_feature='Cout*Kh*Kw',minimum_gain_pct=10.,
    domain=dict(minimum=dims.min(axis=0).tolist(),maximum=dims.max(axis=0).tolist(),
        dimensions=['N','Cin','H','W','Cout'],kernels=[1,3,5],strides=[1,2],padding=[0,1],
        dtype='float32',layout='contiguous NCHW/OIHW',groups=1,dilation=1))
with (a.output/'candidate-search.json').open('x') as f:json.dump(candidates,f,indent=2);f.write('\n')
with (a.output/'frozen-policy.json').open('x') as f:json.dump(policy,f,indent=2);f.write('\n')
print(json.dumps(policy,indent=2))
