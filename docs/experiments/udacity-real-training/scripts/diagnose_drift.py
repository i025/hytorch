"""Locate actual strict-tolerance trajectory mismatches without another GPU run."""
import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--raw',type=Path,required=True);p.add_argument('--prepared',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
protocol=json.loads((a.prepared/'protocol.json').read_text());rows=[];weights=[]
for actual,reference in itertools.combinations('AFIE',2):
    aa=[json.loads(s) for s in (a.raw/('train-'+actual)/'losses.jsonl').read_text().splitlines()]
    bb=[json.loads(s) for s in (a.raw/('train-'+reference)/'losses.jsonl').read_text().splitlines()]
    x=np.array([r['loss'] for r in aa]);y=np.array([r['loss'] for r in bb]);error=abs(x-y)
    bad=np.flatnonzero(error>1e-6+1e-5*abs(y));index=int(bad[0]) if len(bad) else None
    row=dict(actual=actual,reference=reference,steps=len(x),loss_mismatches=len(bad),max_loss_abs=float(error.max()),
        first_bad_global_step=index+1 if index is not None else None)
    if index is not None:
        record=aa[index];indices=protocol['epoch_order'][record['epoch']-1][(record['step']-1)*4:record['step']*4]
        row.update(first_bad_epoch=record['epoch'],first_bad_epoch_step=record['step'],
            first_bad_actual=float(x[index]),first_bad_reference=float(y[index]),first_bad_delta=float(error[index]),
            original_csv_rows=json.dumps(indices))
    rows.append(row)
    with np.load(a.raw/('train-'+actual)/'final_weights.npz') as ww,np.load(a.raw/('train-'+reference)/'final_weights.npz') as vv:
        for name in ww.files:
            x,y=ww[name].astype(np.float64),vv[name].astype(np.float64);delta=x-y
            index=np.unravel_index(abs(delta).argmax(),delta.shape)
            weights.append(dict(actual=actual,reference=reference,parameter=name,elements=x.size,
                mismatches=int((abs(delta)>1e-6+1e-5*abs(y)).sum()),max_abs=float(abs(delta).max()),
                relative_l2=float(np.linalg.norm(delta)/np.linalg.norm(y)),worst_index=str(index),
                actual_at_worst=float(x[index]),reference_at_worst=float(y[index]),allowed_at_worst=float(1e-6+1e-5*abs(y[index]))))
for name,data in [('loss-drift',rows),('weight-drift',weights)]:
    with (a.output/(name+'.csv')).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in data for k in r)));w.writeheader();w.writerows(data)
result=dict(status='FAIL_UNDER_EXISTING_TOLERANCE',root_cause='UNRESOLVED: trajectory logs and final weights do not identify the first differing backward operation',
    unchanged_tolerance=dict(atol=1e-6,rtol=1e-5),pairs=rows,
    limits='No per-step activation/gradient checkpoints; no claim that this is a CUDA bug or solely FP32 rounding')
(a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
