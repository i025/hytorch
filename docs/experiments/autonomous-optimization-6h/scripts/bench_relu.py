"""Raw per-call CUDA-event and synchronized host latency for complete ReLU backward."""
import argparse
from pathlib import Path
import time

import numpy as np
from common import ROOT, write_csv, write_json
import hytorch
import hytorch.functions as F
from hytorch.backends.cuda import backend, dispatch_stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trace', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import cupy as cp
    cp.cuda.Device(0).use()
    shapes = [(257,), (3, 7, 11), (4, 24, 31, 98), (4, 36, 14, 47),
              (4, 48, 5, 22), (4, 64, 3, 20), (4, 64, 1, 18), (4, 100),
              (4, 50), (16, 32, 33, 37), (1, 16, 29, 41), (17, 31)]
    rows = []
    for case, shape in enumerate(shapes):
        rng = np.random.default_rng(42)
        x, g = [cp.asarray(rng.normal(size=shape).astype(np.float32)) for _ in range(2)]
        layout = 'transposed' if case == len(shapes) - 1 else 'contiguous'
        if layout == 'transposed':
            x, g = x.T, g.T
        node = F.ReLU()
        node.inputs = [hytorch.Variable(x)]
        gy = hytorch.Variable(g)
        for repeat in range(1, 4):
            order = ('original', 'fused') if repeat % 2 else ('fused', 'original')
            for mode in order:
                with hytorch.no_grad(), backend(relu_backward=mode):
                    for _ in range(10):
                        value = node.backward(gy)
                    cp.cuda.get_current_stream().synchronize()
                    for iteration in range(5 if args.trace else 30):
                        start, stop = cp.cuda.Event(), cp.cuda.Event()
                        label = f'OP/micro:{mode}|case={case}|step={(repeat - 1) * 5 + iteration}'
                        if args.trace:
                            cp.cuda.nvtx.RangePush(label)
                        dispatch_stats(reset=True)
                        begin = time.perf_counter()
                        start.record()
                        value = node.backward(gy)
                        stop.record()
                        stop.synchronize()
                        wall = time.perf_counter() - begin
                        if args.trace:
                            cp.cuda.nvtx.RangePop()
                        actual = cp.asnumpy(value.data)
                        np.testing.assert_array_equal(actual, cp.asnumpy(g * (x > 0)))
                        rows.append(dict(case=case, shape=str(x.shape), layout=layout, elements=x.size, mode=mode,
                            repeat=repeat, sample=iteration, event_span_us=cp.cuda.get_elapsed_time(start, stop) * 1000,
                            synchronized_host_us=wall * 1e6, extra_mask_bytes=x.size if mode == 'original' else 0,
                            output_bytes=value.data.nbytes,
                            pool_used_bytes=cp.get_default_memory_pool().used_bytes(),
                            pool_reserved_bytes=cp.get_default_memory_pool().total_bytes()))
    write_csv(args.output / 'raw.csv', rows)
    write_json(args.output / 'summary.json', dict(status='PASSED', measurements=len(rows),
        cases=len(shapes), warmup=10, repeats=3, samples=5 if args.trace else 30,
        correctness='Bitwise equality with original FP32 mask multiply',
        event_scope='Full device-event span including possible host submission gaps; pure kernel sums require CUPTI',
        host_scope='Event record, complete Python backward call and final event synchronization; event object creation excluded'))


if __name__ == '__main__':
    main()
