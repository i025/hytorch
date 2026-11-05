"""Future GPU benchmark. --plan is CPU-only; --run-gpu explicitly executes CUDA."""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'docs/experiments/cuda-reduction-foundation/scripts'))
from cases import CASES
from hytorch.backends.cuda import backend
from hytorch.backends.cuda import runtime as rt
from hytorch.backends.cuda.reduction import plan_sum


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--iterations', type=int, default=30)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--nvtx', action='store_true')
    parser.add_argument('--eligibility', type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--plan', action='store_true')
    action.add_argument('--run-gpu', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    eligible = json.loads(args.eligibility.read_text())['eligible_cases']
    config = dict(shapes=[dict(case=i, shape=s, axis=a) for i,(s,a) in enumerate(CASES)],
        methods=['cupy', 'native-full', 'native-preallocated', 'torch'], keepdims=False,
        warmup=args.warmup, iterations=args.iterations, repeats=args.repeats,
        atol=1e-5, rtol=1e-4, gpu='NOT_RUN' if args.plan else 'REQUESTED',
        event_scope='CUDA event span, includes possible host submission gaps; NOT pure kernel time',
        kernel_scope='Sum of correlated CUPTI kernel durations from separate Nsight run',
        full_scope='Submission through stream synchronization, includes allocation/layout/ctypes',
        selected_cases=eligible, seed=42, tf32=False, timestamp=datetime.now(timezone.utc).isoformat())
    write_json(args.output/'config.json', config)
    if args.plan:
        print('GPU NOT_RUN: CPU-only benchmark plan written')
        return
    import cupy as cp
    import torch
    cp.cuda.Device(0).use()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config['cuda_visible_devices'] = os.environ['CUDA_VISIBLE_DEVICES']
    write_json(args.output/'config.json', config)
    rng = np.random.default_rng(42)
    rows = []
    errors = []
    with (args.output/'timings.csv').open('x', newline='') as file, backend(reduction='native'):
        writer = None
        for case, (shape, axis) in enumerate(CASES):
            host = rng.standard_normal(shape).astype(np.float32)
            if case not in eligible:
                continue
            x = cp.asarray(host)
            tx = torch.tensor(host, device='cuda')
            layout, outshape, parts = plan_sum(shape, axis)
            y = cp.empty(outshape, cp.float32)
            workspace = cp.empty((layout.outputs, parts), cp.float32) if parts>1 else None
            rt.reduce_sum(x, axis)  # Bind ABI and compile library dependencies before timing.
            def full():
                return rt.sum_array(x, axis)
            def preallocated():
                rt._call('hy_reduce_sum', rt._ptr(x), rt._ptr(y), rt._ptr(workspace), layout, parts)
                return y
            functions = dict(cupy=lambda: x.sum(axis=axis), **{'native-full': full,
                'native-preallocated': preallocated,
                'torch': lambda: tx+0. if axis==() else torch.sum(tx, dim=axis)})
            target = host.sum(axis=axis)
            case_valid = True
            for method, fn in functions.items():
                actual = fn()
                a = actual.detach().cpu().numpy() if method=='torch' else cp.asnumpy(actual)
                delta = a.astype(np.float64)-target.astype(np.float64)
                error = dict(case=case, method=method, max_abs=float(np.abs(delta).max()),
                    relative_l2=float(np.linalg.norm(delta.ravel())/max(np.linalg.norm(target.ravel()),1e-30)),
                    mismatches=int((np.abs(delta)>1e-5+1e-4*np.abs(target)).sum()))
                errors.append(error)
                write_json(args.output/'errors.json', errors)
                if error['mismatches']:
                    np.savez(args.output/('failure-{}-{}.npz'.format(case,method)), x=host, actual=a, reference=target)
                    case_valid = False
                for _ in range(args.warmup):
                    fn()
            if not case_valid:
                continue
            cp.cuda.get_current_stream().synchronize()
            torch.cuda.synchronize()
            methods = list(functions)
            for repeat in range(args.repeats):
                for method in methods[repeat%len(methods):]+methods[:repeat%len(methods)]:
                    for step in range(args.iterations):
                        sync = torch.cuda.synchronize if method=='torch' else cp.cuda.get_current_stream().synchronize
                        sync()
                        before_bytes = cp.get_default_memory_pool().used_bytes()
                        start, end = (torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) if method=='torch' else (cp.cuda.Event(),cp.cuda.Event())
                        label = 'OP/micro:{}|case=c{}_r{}_{}|step={}'.format(
                            method.replace('-','_'), case, repeat, method.replace('-','_'), step)
                        if args.nvtx:
                            cp.cuda.nvtx.RangePush(label)
                        wall = time.perf_counter()
                        start.record()
                        submit = time.perf_counter()
                        value = functions[method]()
                        host_us = (time.perf_counter()-submit)*1e6
                        end.record()
                        sync()
                        wall_us = (time.perf_counter()-wall)*1e6
                        if args.nvtx:
                            cp.cuda.nvtx.RangePop()
                        event_us = (start.elapsed_time(end) if method=='torch' else cp.cuda.get_elapsed_time(start,end))*1000
                        row = dict(case=case, shape=str(shape), axis=str(axis), method=method,
                            repeat=repeat, step=step, event_span_us=event_us, full_sync_wall_us=wall_us,
                            host_submit_us=host_us, workspace_bytes=4*layout.outputs*parts if method.startswith('native') and parts>1 else 0,
                            cupy_live_delta_bytes=cp.get_default_memory_pool().used_bytes()-before_bytes)
                        if writer is None:
                            writer=csv.DictWriter(file,fieldnames=list(row));writer.writeheader()
                        writer.writerow(row);file.flush()
                        rows.append(row)
    write_json(args.output/'summary.json', dict(status='COMPLETED', samples=len(rows),
        pure_kernel_time='Requires separate Nsight correlation export; event_span_us is not pure kernel time',
        gpu='EXECUTED', finished_at_utc=datetime.now(timezone.utc).isoformat()))


if __name__ == '__main__':
    main()
