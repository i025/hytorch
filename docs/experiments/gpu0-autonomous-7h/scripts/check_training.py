"""Same-state and sequential real-image checks; no tolerance changes or hashes."""
import argparse
import json
from pathlib import Path

import numpy as np
from common import Engine,compare,result,write_json,write_csv
from hytorch.backends.cuda import dispatch_stats


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sum-enabled',action='store_true')
    p.add_argument('--relu-disabled',action='store_true')
    p.add_argument('--run-gpu',action='store_true',required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    data=np.load(a.prepared/'throughput-inputs.npz');rows=[];eligible={};base_pass=True
    methods=['A','I','P','E','E-prefetch']
    if not a.relu_disabled:methods.append('R')
    if a.sum_enabled:methods.append('S')
    methods.append('J')
    for model in ('donkey','spatial'):
        values={}
        for method in methods:
            engine=Engine(method,model,sum_enabled=a.sum_enabled,relu_enabled=not a.relu_disabled)
            dispatch_stats(reset=True)
            with engine.context():
                steps=[]
                for scope in ('same-state','three-step'):
                    engine.restore()
                    for step in range(3):
                        if scope=='same-state':engine.restore()
                        steps.append(result(engine,data['x'][4*step:4*step+4],data['t'][4*step:4*step+4]))
            counts=dispatch_stats()
            if method in ('A','E','E-prefetch'):
                assert not any(counts.get(k,0) for k in ('gemm_shared','conv_im2col','pool_native',
                    'sgd_fused','conv_dinput_fused','conv_dweight_fused','relu_backward_fused','sum_native'))
            write_json(a.output/(model+'-'+method+'-dispatch.json'),counts)
            values[method]=steps
            np.savez(a.output/(model+'-'+method+'.npz'),**{
                str(i)+'/'+k:v for i,step in enumerate(steps) for k,v in step.items()})
            del engine
        pairs=[(m,'I') for m in methods if m not in ('I','E','E-prefetch')]
        pairs += [(m,'E') for m in methods if m!='E']
        local=[]
        for method,reference in pairs:
            for step,(actual,target) in enumerate(zip(values[method],values[reference])):
                for key in actual:
                    local.append(compare(actual[key],target[key],key,model=model,method=method,
                        reference=reference,scope='same-state' if step<3 else 'three-step',step=step%3+1))
        eligible[model]={m:all(r['finite'] and not r['mismatches'] for r in local if r['method']==m) for m in methods}
        eligible[model]['E']=True
        base_pass &= all(eligible[model][m] for m in ('A','I','P','E-prefetch'))
        rows.extend(local);write_csv(a.output/(model+'-errors.csv'),local)
    summary=dict(status='PASS' if all(r['finite'] and not r['mismatches'] for r in rows) else 'NUMERICAL_FAILURES_RETAINED',
        comparisons=len(rows),failed=sum(not r['finite'] or bool(r['mismatches']) for r in rows),
        base_pass=base_pass,eligible=eligible,gpu='EXECUTED',
        sum_general_status='Restricted real-input checks only; retained cancellation failures remain FAIL')
    write_json(a.output/'summary.json',summary);print(json.dumps(summary),flush=True)
    assert base_pass,'Baseline numerical checks failed; no formal training performance'


if __name__=='__main__':main()
