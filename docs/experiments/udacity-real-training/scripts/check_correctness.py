"""Real-image forward, parameter gradients and SGD updates at existing tolerances."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from engine import Engine


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    data=np.load(a.prepared/'throughput-inputs.npz');x,t=data['x'][:12],data['t'][:12]
    np.savez(a.output/'inputs.npz',x=x,t=t,rows=data['indices'][:12])
    methods='AE' if a.device=='cpu' else 'AFIE';engines={m:Engine(m,a.device) for m in methods}
    records=[]
    for step in range(3):
        results={}
        for m,e in engines.items():
            with e.context():
                batch=x[step*4:(step+1)*4];labels=t[step*4:(step+1)*4]
                prediction=e.predict(batch);loss=e.loss(e.convert(batch),e.convert(labels));e.backward(loss)
                grads=e.gradients();value=e.value(loss);e.update();e.sync()
                results[m]=dict(forward=prediction,loss=np.array(value),**{'grad/'+k:v for k,v in grads.items()},
                    **{'weight/'+k:v for k,v in e.state().items()})
                np.savez(a.output/f'step-{step}-{m}.npz',**results[m])
        for m in methods:
            for ref in ('A','E'):
                for name,value in results[m].items():
                    target=results[ref][name];delta=value.astype(np.float64)-target
                    atol,rtol=(1e-5,1e-4) if name.startswith('grad/') or name=='forward' else (1e-6,1e-5)
                    row=dict(step=step,method=m,reference=ref,value=name,atol=atol,rtol=rtol,
                        max_abs=float(abs(delta).max()),relative_l2=float(np.linalg.norm(delta)/max(np.linalg.norm(target),1e-30)),
                        mismatches=int((abs(delta)>atol+rtol*abs(target)).sum()),finite=bool(np.isfinite(value).all()))
                    records.append(row)
    with (a.output/'errors.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    passed=all(r['finite'] and r['mismatches']==0 for r in records)
    result=dict(status='PASSED' if passed else 'FAILED',device=a.device,comparisons=len(records),
        max_abs=max(r['max_abs'] for r in records),input_kind='UDACITY_REAL_RGB',no_fingerprints=True)
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    assert passed,'Full failing values and producer image row IDs preserved'


if __name__=='__main__':main()
