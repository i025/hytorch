"""CPU tracemalloc allocation study for the same optional heap scheduler."""
import argparse
import gc
import json
from datetime import datetime, timezone
from pathlib import Path
import time
import tracemalloc

import hytorch
from bench_graph_shapes import graph, graph_stats
from common import write_csv, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--schedule', type=Path, required=True)
    p.add_argument('--warmup', type=int, default=2)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(a.schedule.read_text())
    window = Path(cfg['raw']) / 'gpu-window-01/window.json'
    cutoff = datetime.fromisoformat(cfg['gpu_cutoff_utc']).timestamp()
    rows = []
    for elements in (128, 8192):
        for kind, width in (('chain', 256), ('tree', 256), ('fanout', 256)):
            for repeat in range(3):
                for mode in (('list', 'heap') if repeat % 2 == 0 else ('heap', 'list')):
                    while window.exists() and json.loads(window.read_text())['status'] == 'RUNNING':
                        if time.time() >= cutoff:
                            raise RuntimeError('CPU memory study cutoff')
                        time.sleep(1)
                    if time.time() >= cutoff:
                        raise RuntimeError('CPU memory study cutoff')
                    with hytorch.using_config('backward_schedule', mode):
                        for _ in range(a.warmup):
                            loss, x = graph(kind, width, elements)
                            loss.backward()
                            del loss, x
                    gc.collect()
                    loss, x = graph(kind, width, elements)
                    stats = graph_stats(loss)
                    with hytorch.using_config('backward_schedule', mode):
                        tracemalloc.start()
                        loss.backward()
                        live, peak = tracemalloc.get_traced_memory()
                        tracemalloc.stop()
                    rows.append(dict(kind=kind, width=width, elements=elements, mode=mode,
                                     repeat=repeat, **stats, tracked_live_bytes=live, tracked_peak_bytes=peak))
                    del loss, x
    write_csv(a.output / 'memory.csv', rows)
    write_json(a.output / 'summary.json', dict(status='MEASURED', gpu='NOT_RUN', measurements=len(rows),
               repeats=3, warmup=a.warmup, finished_utc=datetime.now(timezone.utc).isoformat(),
               scope='CPU allocations tracked during backward only; graph built before tracing',
               limitation='Not process RSS, not GPU memory, not total historical allocations; profiler overhead excluded from performance claims'))


if __name__ == '__main__':
    main()
