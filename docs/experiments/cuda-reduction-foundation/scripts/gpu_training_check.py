"""Future same-state and three-step real-data numerical acceptance before timing."""
import argparse
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--run-gpu', action='store_true', required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    from training_common import Engine, compare, result, write_json, write_csv
    from hytorch.backends.cuda import dispatch_stats
    data = np.load(a.prepared/'throughput-inputs.npz')
    rows = []
    for model in ('donkey', 'spatial'):
        values = {}
        for method in ('A','I','S','R','SR','E'):
            engine = Engine(method, model)
            dispatch_stats(reset=True)
            with engine.context():
                steps = []
                # Reset before each same-state step, then independently test three SGD steps.
                for step in range(3):
                    engine.restore()
                    steps.append(result(engine,data['x'][4*step:4*step+4],data['t'][4*step:4*step+4]))
                engine.restore()
                for step in range(3):
                    steps.append(result(engine,data['x'][4*step:4*step+4],data['t'][4*step:4*step+4]))
            counts = dispatch_stats()
            if method in ('A','E'):
                assert not any(counts.get(k,0) for k in ('gemm_shared','conv_im2col','pool_native',
                    'sgd_fused','sum_native','relu_backward_fused','conv_dweight_fused','conv_dinput_fused'))
            if method in ('S','SR'):
                assert counts.get('sum_native',0)>0
            write_json(a.output/(model+'-'+method+'-dispatch.json'), counts)
            values[method] = steps
            np.savez(a.output/(model+'-'+method+'.npz'), **{
                str(i)+'/'+key: value for i, step in enumerate(steps) for key,value in step.items()})
            del engine
        for method, reference in (('S','I'),('SR','I'),('R','I'),('A','E'),('I','E'),('S','E'),('SR','E')):
            for step,(actual,target) in enumerate(zip(values[method],values[reference])):
                for key in actual:
                    rows.append(compare(actual[key], target[key], key, model=model,
                        method=method, reference=reference, step=step%3+1,
                        scope='same-state' if step<3 else 'three-step'))
        write_csv(a.output/(model+'-errors.csv'), [r for r in rows if r['model']==model])
    failures = sum(not r['finite'] or r['mismatches']>0 for r in rows)
    write_json(a.output/'summary.json', dict(status='FAIL' if failures else 'PASS',
        comparisons=len(rows), failed=failures, gpu='EXECUTED'))
    assert failures == 0, 'Numerical failures retained; do not proceed to performance'


if __name__ == '__main__':
    main()
