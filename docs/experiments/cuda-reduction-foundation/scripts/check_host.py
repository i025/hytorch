"""CPU-only library ABI loading and matched CNN training semantics."""
import argparse
import ctypes as ct
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from hytorch.backends.cuda import backend, dispatch_stats
from hytorch.backends.cuda.runtime import load_library


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    lib = load_library()  # CDLL and host ABI function only; never call a CUDA entry.
    symbol = lib.hy_reduce_sum
    assert isinstance(symbol, ct._CFuncPtr)
    spec = importlib.util.spec_from_file_location('prior_common', ROOT/'docs/experiments/autonomous-optimization-6h/scripts/common.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = []
    rng = np.random.default_rng(42)
    x = rng.normal(0, .1, (4,3,66,200)).astype(np.float32)
    t = rng.normal(0, .1, (4,1)).astype(np.float32)
    for model in ('donkey', 'spatial'):
        values = {}
        for mode in ('original', 'native', 'torch'):
            engine = module.Engine('E' if mode == 'torch' else 'A', model, device='cpu')
            dispatch_stats(reset=True)
            with backend(reduction='native' if mode == 'native' else 'original'):
                values[mode] = [module.result(engine, x, t) for _ in range(3)]
            if mode == 'native':
                assert dispatch_stats().get('sum_native', 0) == 0
                assert dispatch_stats()['reduction_fallback_cpu_or_dtype'] > 0
        for mode in ('native', 'torch'):
            for step, (actual, target) in enumerate(zip(values[mode], values['original'])):
                for key in actual:
                    row = module.compare(actual[key], target[key], key, model=model, mode=mode, step=step+1)
                    rows.append(row)
        np.savez(args.output/(model+'-values.npz'), **{
            mode+'/'+str(step+1)+'/'+key: value
            for mode, steps in values.items() for step, values_ in enumerate(steps)
            for key, value in values_.items()})
    module.write_csv(args.output/'comparisons.csv', rows)
    result = dict(status='PASS' if all(r['finite'] and not r['mismatches'] for r in rows) else 'FAIL',
                  comparisons=len(rows), failed=sum(not r['finite'] or r['mismatches'] > 0 for r in rows),
                  library_load='PASS', symbol='hy_reduce_sum', gpu='NOT_RUN',
                  scope='CPU original/native-selection fallback vs CPU PyTorch, not native GPU execution')
    module.write_json(args.output/'summary.json', result)
    print(json.dumps(result), flush=True)
    assert result['status'] == 'PASS'


if __name__ == '__main__':
    main()
