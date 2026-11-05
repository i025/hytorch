"""Separate GPU allocation instrumentation; never part of operator timing."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from cupy.cuda.memory_hook import MemoryHook
from common import write_json
from cases import CASES
from hytorch.backends.cuda import backend
from hytorch.backends.cuda import runtime
import hytorch
import hytorch.functions as F


class AllocationTrace(MemoryHook):
    name = 'hytorch_operator_allocations'

    def __init__(self):
        self.blocks = {}
        self.live = self.peak = self.requested_total = self.calls = 0

    def malloc_postprocess(self, device_id, size, mem_size, mem_ptr, pmem_id):
        self.blocks[pmem_id] = mem_size
        self.live += mem_size
        self.peak = max(self.peak, self.live)
        self.requested_total += size
        self.calls += 1

    def free_postprocess(self, device_id, mem_size, mem_ptr, pmem_id):
        if pmem_id in self.blocks:
            self.live -= self.blocks.pop(pmem_id)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--operator', choices=['sum', 'relu'], required=True)
    p.add_argument('--eligibility', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--run-gpu', action='store_true', required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    import cupy as cp
    import torch
    cp.cuda.Device(0).use()
    torch.set_num_threads(4)
    rows = []
    if a.operator == 'sum':
        eligible = json.loads(a.eligibility.read_text())['eligible_cases']
        cases = [(i, shape, axis, False) for i, (shape, axis) in enumerate(CASES) if i in eligible]
    else:
        shapes = [(257,), (3,7,11), (4,24,31,98), (4,36,14,47), (4,48,5,22),
                  (4,64,3,20), (4,64,1,18), (4,100), (4,50), (16,32,33,37), (1,16,29,41), (17,31)]
        cases = [(i, shape, None, i == len(shapes)-1) for i, shape in enumerate(shapes)]
    for case, shape, axis, transposed in cases:
        rng = np.random.default_rng(42)
        x = cp.asarray(rng.standard_normal(shape).astype(np.float32))
        if a.operator == 'sum':
            tx = torch.tensor(cp.asnumpy(x), device='cuda')
            functions = [('cupy', lambda: x.sum(axis=axis)),
                         ('native', lambda: runtime.sum_array(x, axis)),
                         ('torch', lambda: tx+0. if axis == () else torch.sum(tx, dim=axis))]
        else:
            g = cp.asarray(rng.standard_normal(shape).astype(np.float32))
            if transposed:
                x, g = x.T, g.T
            node = F.ReLU()
            node.inputs = [hytorch.Variable(x)]
            gy = hytorch.Variable(g)
            functions = [('original', lambda: node.backward(gy)), ('fused', lambda: node.backward(gy))]
        for method, fn in functions:
            with hytorch.no_grad(), backend(reduction='native',
                    relu_backward='fused' if method == 'fused' else 'original'):
                for _ in range(5):
                    value = fn()
                cp.cuda.get_current_stream().synchronize()
                torch.cuda.synchronize()
                del value
                for repeat in range(3):
                    pool = cp.get_default_memory_pool()
                    used, reserved = pool.used_bytes(), pool.total_bytes()
                    trace = AllocationTrace()
                    torch_before = torch.cuda.memory_allocated() if method == 'torch' else None
                    if method == 'torch':
                        torch.cuda.reset_peak_memory_stats()
                    with trace:
                        value = fn()
                        cp.cuda.get_current_stream().synchronize()
                        torch.cuda.synchronize()
                    rows.append(dict(case=case, shape=str(x.shape), axis=str(axis), method=method,
                                     repeat=repeat, layout='transposed' if transposed else 'contiguous',
                                     cupy_allocation_calls=trace.calls,
                                     cupy_total_requested_bytes=trace.requested_total,
                                     cupy_extra_rounded_peak_bytes=trace.peak,
                                     cupy_extra_rounded_live_bytes=trace.live,
                                     cupy_transient_over_return_live_bytes=trace.peak-trace.live,
                                     output_bytes=value.numel()*value.element_size() if method == 'torch' else
                                         (value.data.nbytes if a.operator == 'relu' else value.nbytes),
                                     cupy_pool_used_before=used, cupy_pool_used_after=pool.used_bytes(),
                                     cupy_pool_reserved_before=reserved, cupy_pool_reserved_after=pool.total_bytes(),
                                     torch_extra_peak_bytes=torch.cuda.max_memory_allocated()-torch_before if method == 'torch' else None))
                    path = a.output / 'allocations.csv'
                    exists = path.exists()
                    with path.open('a', newline='') as file:
                        writer = csv.DictWriter(file, fieldnames=list(rows[-1]), lineterminator='\n')
                        if not exists:
                            writer.writeheader()
                        writer.writerow(rows[-1])
                    del value
    write_json(a.output / 'summary.json', dict(status='MEASURED', gpu='EXECUTED', records=len(rows),
               scope='Separate instrumented allocation run, never included in operator latency or training throughput',
               cupy_scope='Peak concurrently live rounded pool allocations created inside the call, including returned output',
               limitations='Pre-existing inputs and cached pool reservations excluded; not an exact whole-device/process physical memory peak'))


if __name__ == '__main__':
    main()
