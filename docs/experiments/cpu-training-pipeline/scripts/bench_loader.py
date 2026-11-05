"""Stream real batches on CPU; each run has an independent process RSS peak."""
import argparse
import json
from datetime import datetime, timezone
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np
from pipeline_common import ROOT, configurations, load_data, load_module, write_csv, write_json
from hytorch.dataloaders import DataLoader


def run(args):
    protocol, dataset = load_data(args.prepared, args.variant, args.baseline_dataset)
    cls = load_module(args.baseline).DataLoader if args.variant == 'existing' else DataLoader
    loader = cls(dataset, 4, shuffle=False, indices=protocol['throughput'],
                 num_workers=args.workers, prefetch_factor=args.prefetch)
    started = datetime.now(timezone.utc).isoformat()
    cpu = time.process_time()
    begin = time.perf_counter()
    iterator = iter(loader)
    startup = time.perf_counter() - begin
    batches, samples = [], 0
    for step in range(loader.max_iter):
        before = time.perf_counter()
        x, t = next(iterator)
        batches.append(dict(step=step+1, samples=len(x), next_ms=(time.perf_counter()-before)*1000))
        samples += len(x)
    before = time.perf_counter()
    assert next(iterator, None) is None
    shutdown = time.perf_counter() - before
    wall = time.perf_counter() - begin
    cpu = time.process_time() - cpu
    write_csv(args.output / 'batches.csv', batches)
    summary = dict(variant=args.variant, repeat=args.repeat, workers=args.workers,
        prefetch_factor=args.prefetch, samples=samples, batches=len(batches),
        wall_s=wall, images_per_s=samples/wall, cpu_s=cpu, process_cpu_pct=100*cpu/wall,
        startup_ms=startup*1000, shutdown_ms=shutdown*1000,
        mean_next_ms=float(np.mean([b['next_ms'] for b in batches])),
        p95_next_ms=float(np.percentile([b['next_ms'] for b in batches], 95)),
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        started_at_utc=started, ended_at_utc=datetime.now(timezone.utc).isoformat(),
        affinity=sorted(os.sched_getaffinity(0)), cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
        scope='CPU hot-cache streaming, no simulated training consumer, no image retention, includes worker startup/exhaustion; no CUDA')
    write_json(args.output / 'summary.json', summary)
    print(str(args.output.name)+': '+str(round(summary['images_per_s'], 2))+' images/s', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline-dataset', type=Path, default=ROOT/'runs/cpu_pipeline_01/baseline_udacity_dataset.py')
    parser.add_argument('--variants', nargs='+', choices=['existing','repaired','inplace','stream'])
    parser.add_argument('--narrow', action='store_true')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--variant', choices=['existing', 'repaired', 'inplace', 'stream'])
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--prefetch', type=int, default=2)
    parser.add_argument('--repeat', type=int, default=1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.variant:
        run(args)
        return
    settings = [(w,2) for w in (0,2,4)] if args.narrow else configurations()
    jobs = [(variant, w, p) for variant in (args.variants or ['existing','repaired','inplace']) for w,p in settings]
    rng = np.random.RandomState(42)
    commands, rows = [], []
    for repeat in range(1, args.repeats+1):
        for position in rng.permutation(len(jobs)):
            variant, workers, prefetch = jobs[position]
            out = args.output / f'{repeat}-{variant}-w{workers}-p{prefetch}'
            command = [sys.executable, __file__, '--prepared', str(args.prepared), '--baseline', str(args.baseline),
                       '--output', str(out), '--baseline-dataset', str(args.baseline_dataset), '--variant', variant, '--workers', str(workers),
                       '--prefetch', str(prefetch), '--repeat', str(repeat)]
            commands.append(dict(order=len(commands)+1, command=command))
            write_json(args.output / 'commands.json', commands)
            subprocess.run(command, check=True)
            row = json.loads((out/'summary.json').read_text())
            row['order'] = len(commands)
            rows.append(row)
            write_csv(args.output / 'runs.csv', rows)
    write_json(args.output / 'status.json', dict(status='COMPLETED', runs=len(rows), gpu='NOT_RUN'))


if __name__ == '__main__':
    main()
