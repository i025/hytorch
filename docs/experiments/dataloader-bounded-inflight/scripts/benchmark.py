"""Matched CPU throughput and per-process peak memory; no instrumentation of queues."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np
from common import ROOT, inputs, loader_module, write_csv, write_json


def measure(args):
    module = loader_module(args.variant, args.baseline)
    dataset, indices = inputs(args.prepared, args.workload, args.size)
    loader = module.DataLoader(dataset, 4, shuffle=False, indices=indices,
                               num_workers=args.workers, prefetch_factor=args.prefetch)
    record = dict(variant=args.variant, workload=args.workload, repeat=args.repeat,
                  workers=args.workers, prefetch_factor=args.prefetch, samples=len(indices),
                  steps=loader.max_iter, image_shape=[3,66,200], dtype='float32',
                  consumer_sleep_ms=4 if args.workload=='real-consumer' else 0,
                  straggler_ms=1000 if args.workload=='straggler' else 0,
                  synthetic_item_sleep_ms=.5 if args.workload=='straggler' else 0,
                  git_version=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  started_at_utc=datetime.now(timezone.utc).isoformat(),
                  affinity=sorted(os.sched_getaffinity(0)), cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'])
    cpu = time.process_time()
    start = time.perf_counter()
    iterator = iter(loader)
    startup = time.perf_counter()-start
    batches = []
    for step in range(loader.max_iter):
        before = time.perf_counter()
        x, t = next(iterator)
        elapsed = time.perf_counter()-before
        batches.append(dict(step=step+1, samples=len(x), next_ms=elapsed*1000))
        if args.workload == 'real-consumer':
            time.sleep(.004)
    before = time.perf_counter()
    assert next(iterator, None) is None
    shutdown = time.perf_counter()-before
    wall = time.perf_counter()-start
    cpu = time.process_time()-cpu
    record.update(wall_s=wall, images_per_s=len(indices)/wall, process_cpu_pct=cpu/wall*100,
        startup_ms=startup*1000, shutdown_ms=shutdown*1000,
        mean_next_ms=float(np.mean([b['next_ms'] for b in batches])),
        p95_next_ms=float(np.percentile([b['next_ms'] for b in batches],95)),
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        ended_at_utc=datetime.now(timezone.utc).isoformat(),
        scope='CPU streaming includes worker startup/exhaustion; RSS is lifetime process high-water; cache not reset; no GPU training')
    write_csv(args.output/'batches.csv', batches)
    write_json(args.output/'summary.json', record)
    print(args.output.name+': '+str(round(record['images_per_s'],2))+' images/s', flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--variant',choices=['baseline','bounded'])
    parser.add_argument('--workload',choices=['real-fast','real-consumer','straggler'],default='real-fast')
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--prefetch',type=int,default=2)
    parser.add_argument('--repeat',type=int,default=1)
    parser.add_argument('--size',type=int,default=512)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    if args.variant:
        measure(args)
        return
    jobs=[]
    for workload in ('real-fast','real-consumer','straggler'):
        settings=([(0,2)]+[(w,p) for w in (1,2,4) for p in (1,2,4)]) if workload=='real-fast' else [(w,p) for w in (2,4) for p in (1,2,4)]
        for w,p in settings:
            for variant in ('baseline','bounded'):
                jobs.append((workload,w,p,variant))
    commands,rows=[],[]
    rng=np.random.RandomState(42)
    for repeat in range(1,4):
        for index in rng.permutation(len(jobs)):
            workload,workers,prefetch,variant=jobs[index]
            output=args.output/f'{repeat}-{workload}-{variant}-w{workers}-p{prefetch}'
            command=[sys.executable,__file__,'--prepared',str(args.prepared),'--baseline',str(args.baseline),
                     '--output',str(output),'--variant',variant,'--workload',workload,'--workers',str(workers),
                     '--prefetch',str(prefetch),'--repeat',str(repeat),'--size',str(args.size)]
            commands.append(dict(order=len(commands)+1,command=command))
            write_json(args.output/'commands.json',commands)
            subprocess.run(command,check=True)
            row=json.loads((output/'summary.json').read_text());row['order']=len(commands)
            rows.append(row);write_csv(args.output/'runs.csv',rows)
    # Independent epoch-length observations, separate from three-repeat matched statistics.
    scaling=[]
    for size in (128,1024):
        for variant in ('baseline','bounded'):
            output=args.output/f'scaling-{size}-{variant}'
            command=[sys.executable,__file__,'--prepared',str(args.prepared),'--baseline',str(args.baseline),
                     '--output',str(output),'--variant',variant,'--workload','straggler','--workers','2',
                     '--prefetch','2','--size',str(size)]
            commands.append(dict(order=len(commands)+1,command=command))
            write_json(args.output/'commands.json',commands)
            subprocess.run(command,check=True)
            scaling.append(json.loads((output/'summary.json').read_text()))
    write_csv(args.output/'scaling.csv',scaling)
    write_json(args.output/'status.json',dict(status='COMPLETED',matched_runs=len(rows),scaling_runs=len(scaling),gpu='NOT_RUN'))


if __name__=='__main__':
    main()
