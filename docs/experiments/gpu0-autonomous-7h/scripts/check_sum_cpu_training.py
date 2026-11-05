"""Replay the existing Sum arithmetic model in CPU CNN training, not CUDA execution."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import sys

import numpy as np
from common import Engine,compare,result,write_json,write_csv
from reference import modeled_sum,reference_sum
from hytorch.backends.cuda import runtime


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    data=np.load(a.prepared/'throughput-inputs.npz')
    original=runtime.sum_array;calls=[];checks=[]
    for model in ('donkey','spatial'):
        outputs={}
        for mode in ('original','cpu_model'):
            engine=Engine('A',model,device='cpu');step=0
            def shadow(x,axis=None,keepdims=False):
                old=original(x,axis,keepdims)
                if x.dtype!=np.float32 or not x.size or x.ndim>8:return old
                actual=modeled_sum(x,axis,keepdims)
                fp64=reference_sum(x,axis,keepdims)
                frame=sys._getframe(1)
                owner=frame.f_locals.get('self')
                caller=type(owner).__name__ if owner is not None else frame.f_code.co_name
                row=dict(model=model,mode=mode,step=step,call=len(calls),caller=caller,
                    shape=str(x.shape),axis=str(axis),keepdims=keepdims,
                    sum_abs=float(np.abs(x.astype(np.float64)).sum()),
                    minimum_output_abs=float(np.abs(fp64).min()))
                for ref,target in (('numpy',old),('fp64',fp64)):
                    stats=compare(actual,np.asarray(target),'forward/sum',reference=ref,**row)
                    calls.append(stats)
                    if stats['mismatches']:
                        np.savez(a.output/f'failure-{model}-{mode}-{step}-{len(calls)}.npz',x=x,actual=actual,reference=target)
                return actual if mode=='cpu_model' else old
            runtime.sum_array=shadow
            try:
                outputs[mode]=[]
                for step in range(1,4):
                    outputs[mode].append(result(engine,data['x'][(step-1)*4:step*4],data['t'][(step-1)*4:step*4]))
            finally:
                runtime.sum_array=original
            np.savez(a.output/(model+'-'+mode+'.npz'),**{
                str(i)+'/'+key:value for i,step_ in enumerate(outputs[mode]) for key,value in step_.items()})
        for step,(actual,old) in enumerate(zip(outputs['cpu_model'],outputs['original'])):
            for key in actual:checks.append(compare(actual[key],old[key],key,model=model,step=step+1))
    write_csv(a.output/'sum-calls.csv',calls);write_csv(a.output/'training-errors.csv',checks)
    summary=dict(status='PASS' if all(r['finite'] and not r['mismatches'] for r in checks) else 'FAIL',
        training_comparisons=len(checks),training_failed=sum(bool(r['mismatches']) or not r['finite'] for r in checks),
        reduction_comparisons=len(calls),reduction_failed=sum(bool(r['mismatches']) or not r['finite'] for r in calls),
        gpu='NOT_RUN',scope='CPU execution with FP32 lane/tree simulation only; no native device call',
        finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(a.output/'summary.json',summary);print(summary,flush=True)
    assert summary['status']=='PASS'


if __name__=='__main__':main()
