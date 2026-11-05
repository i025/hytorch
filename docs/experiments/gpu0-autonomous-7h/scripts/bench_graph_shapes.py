"""CPU-only scheduling generalization across held-out graph topologies."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time

import numpy as np
import hytorch
import hytorch.functions as F
from common import write_csv, write_json


def graph(kind, width, elements):
    x = hytorch.Variable(np.linspace(-.2, .2, elements, dtype=np.float32))
    root = x * x
    if kind == 'chain':
        y = root
        for _ in range(width):
            y = F.tanh(y + np.float32(.01))
    elif kind == 'residual':
        y = root
        for _ in range(width):
            y = F.tanh(y) + y * np.float32(.01)
    else:
        nodes = [F.tanh(root + np.float32(i / width)) for i in range(width)]
        if kind == 'tree':
            while len(nodes) > 1:
                nodes = [nodes[i] + nodes[i + 1] for i in range(0, len(nodes), 2)]
            y = nodes[0]
        else:
            y = nodes[0]
            for node in nodes[1:]:
                y = y + node
    return F.sum(y), x


def graph_stats(loss):
    ready = [loss.creator]
    seen = {loss.creator}
    peak = 1
    count = 0
    while ready:
        function = ready.pop()
        count += 1
        for value in function.inputs:
            f = value.creator
            if f is not None and f not in seen:
                seen.add(f)
                ready.append(f)
                ready.sort(key=lambda v: v.generation)
        peak = max(peak, len(ready))
    return dict(functions=count, max_ready_functions=peak)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--schedule', type=Path, required=True)
    p.add_argument('--iterations', type=int, default=30)
    p.add_argument('--repeats', type=int, default=5)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(a.schedule.read_text())
    window = Path(cfg['raw']) / 'gpu-window-01/window.json'
    cutoff = datetime.fromisoformat(cfg['gpu_cutoff_utc']).timestamp()
    def cpu_slot():
        while window.exists() and json.loads(window.read_text())['status'] == 'RUNNING':
            if time.time() >= cutoff:
                raise RuntimeError('CPU experiment stopped at GPU cutoff')
            time.sleep(1)
        if time.time() >= cutoff:
            raise RuntimeError('CPU experiment stopped at GPU cutoff')
    cases = [(kind, width, elements) for elements in (128, 8192) for kind, width in
             [('chain', 256), ('residual', 128), ('tree', 64), ('tree', 256), ('fanout', 64), ('fanout', 256)]]
    rows = []
    checks = []
    summary = []
    arrays = {}
    for case, (kind, width, elements) in enumerate(cases):
        cpu_slot()
        values = {}
        loss, _ = graph(kind, width, elements)
        stats = graph_stats(loss)
        for mode in ('list', 'heap'):
            with hytorch.using_config('backward_schedule', mode):
                loss, x = graph(kind, width, elements)
                loss.backward()
                values[mode] = (loss.data.copy(), x.grad.data.copy())
            for name, value in zip(('loss', 'grad'), values[mode]):
                arrays[f'{case}/{mode}/{name}'] = value
        checks.append(dict(case=case, kind=kind, width=width, elements=elements, **stats,
                           loss_equal=np.array_equal(values['list'][0], values['heap'][0]),
                           gradient_equal=np.array_equal(values['list'][1], values['heap'][1])))
        assert checks[-1]['loss_equal'] and checks[-1]['gradient_equal'], checks[-1]
        for repeat in range(a.repeats):
            for mode in (('list', 'heap') if repeat % 2 == 0 else ('heap', 'list')):
                cpu_slot()
                with hytorch.using_config('backward_schedule', mode):
                    for _ in range(3):
                        loss, _ = graph(kind, width, elements)
                        loss.backward()
                    begin = time.perf_counter()
                    for _ in range(a.iterations):
                        loss, _ = graph(kind, width, elements)
                        loss.backward()
                    wall = time.perf_counter() - begin
                row = dict(case=case, kind=kind, width=width, elements=elements, mode=mode,
                           repeat=repeat, iterations=a.iterations, wall_s=wall,
                           ms_per_step=wall * 1000 / a.iterations, **stats)
                rows.append(row)
                path = a.output / 'timings.csv'
                exists = path.exists()
                with path.open('a', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=list(row), lineterminator='\n')
                    if not exists:
                        writer.writeheader()
                    writer.writerow(row)
        times = {mode: [r['ms_per_step'] for r in rows if r['case'] == case and r['mode'] == mode]
                 for mode in ('list', 'heap')}
        summary.append(dict(case=case, kind=kind, width=width, elements=elements, **stats,
                            list_mean_ms=statistics.mean(times['list']),
                            heap_mean_ms=statistics.mean(times['heap']),
                            list_sd_ms=statistics.stdev(times['list']),
                            heap_sd_ms=statistics.stdev(times['heap']),
                            throughput_change_pct=(statistics.mean(times['list']) / statistics.mean(times['heap']) - 1) * 100))
        print(summary[-1], flush=True)
    write_csv(a.output / 'correctness.csv', checks)
    write_csv(a.output / 'summary.csv', summary)
    np.savez(a.output / 'numerical.npz', **arrays)
    write_json(a.output / 'summary.json', dict(status='PASS', gpu='NOT_RUN', cases=len(cases),
               comparisons=2*len(checks), unequal=0, repeats=a.repeats,
               scope='CPU forward + backward synthetic graph; not GPU/CNN training speedup',
               finished_utc=datetime.now(timezone.utc).isoformat()))


if __name__ == '__main__':
    main()
