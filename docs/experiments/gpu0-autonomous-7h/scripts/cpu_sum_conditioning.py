"""CPU FP32 arithmetic conditioning study; no CUDA execution or performance claim."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
from common import write_csv, write_json
from reference import modeled_sum


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--schedule', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(a.schedule.read_text())
    window = Path(cfg['raw']) / 'gpu-window-01/window.json'
    cutoff = datetime.fromisoformat(cfg['gpu_cutoff_utc']).timestamp()
    def cpu_slot():
        while window.exists() and json.loads(window.read_text())['status'] == 'RUNNING':
            if time.time() >= cutoff:
                raise RuntimeError('CPU diagnostic cutoff')
            time.sleep(1)
        if time.time() >= cutoff:
            raise RuntimeError('CPU diagnostic cutoff')
    sizes = (1024, 4097, 65537, 1048577)
    seeds = list(range(20))
    write_json(a.output / 'config.json', dict(sizes=sizes, seeds=seeds,
               input_dtype='float32', ordinary='default_rng(seed).standard_normal(N).astype(float32)',
               cancellation='ordinary minus float32(mean computed in float64)',
               reference='FP64 sum of the fixed FP32 inputs; no FP32 output rounding of reference',
               atol=1e-5, rtol=1e-4, gpu='NOT_RUN'))
    rows = []
    summary = []
    for size in sizes:
        for seed in seeds:
            cpu_slot()
            ordinary = np.random.default_rng(seed).standard_normal(size).astype(np.float32)
            for family in ('ordinary', 'cancellation'):
                x = ordinary if family == 'ordinary' else ordinary - np.float32(ordinary.mean(dtype=np.float64))
                high = float(x.sum(dtype=np.float64))
                absolute_sum = float(np.abs(x.astype(np.float64)).sum())
                condition = absolute_sum / abs(high) if high != 0 else float('inf')
                model = float(modeled_sum(x))
                numpy = float(x.sum(dtype=np.float32))
                failed = False
                for method, value in (('cpu_model', model), ('numpy_fp32', numpy)):
                    error = abs(value - high)
                    limit = 1e-5 + 1e-4 * abs(high)
                    passed = bool(np.isfinite(value) and error <= limit)
                    rows.append(dict(size=size, seed=seed, family=family, method=method,
                                     value=value, fp64_reference=high, sum_absolute_inputs=absolute_sum,
                                     condition_number=condition, absolute_error=error,
                                     relative_error=error/abs(high) if high else float('inf'),
                                     tolerance=limit, status='PASS' if passed else 'FAIL'))
                    failed |= not passed
                if failed:
                    np.savez(a.output / f'failure-n{size}-seed{seed}-{family}.npz', x=x,
                             cpu_model=np.float32(model), numpy_fp32=np.float32(numpy), fp64=np.float64(high))
        print('Completed size', size, flush=True)
    for size in sizes:
        for family in ('ordinary', 'cancellation'):
            for method in ('cpu_model', 'numpy_fp32'):
                selected = [r for r in rows if r['size'] == size and r['family'] == family and r['method'] == method]
                summary.append(dict(size=size, family=family, method=method, samples=len(selected),
                                    failed=sum(r['status'] == 'FAIL' for r in selected),
                                    median_abs_error=float(np.median([r['absolute_error'] for r in selected])),
                                    max_abs_error=max(r['absolute_error'] for r in selected),
                                    median_condition=float(np.median([r['condition_number'] for r in selected]))))
    write_csv(a.output / 'samples.csv', rows)
    write_csv(a.output / 'summary.csv', summary)
    result = dict(status='DIAGNOSIS_COMPLETED', gpu='NOT_RUN', comparisons=len(rows),
                  failures_retained=sum(r['status'] == 'FAIL' for r in rows),
                  ordinary_failures=sum(r['status'] == 'FAIL' for r in rows if r['family'] == 'ordinary'),
                  cancellation_failures=sum(r['status'] == 'FAIL' for r in rows if r['family'] == 'cancellation'),
                  finished_utc=datetime.now(timezone.utc).isoformat(),
                  scope='CPU full-vector samples; neither GPU correctness nor an FP32 tolerance waiver')
    write_json(a.output / 'summary.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
