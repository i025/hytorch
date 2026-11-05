"""Unified future GPU runner. --plan writes CPU-only metadata; GPU execution is explicit."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from pipeline_common import ROOT, load_data, write_json
from training_config import METHODS, configuration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=list(METHODS), required=True)
    parser.add_argument('--model', choices=['donkey', 'spatial'], default='donkey')
    parser.add_argument('--task', choices=['io', 'compute', 'phases'], default='io')
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=256)
    parser.add_argument('--prefetch-factor', type=int, default=2)
    parser.add_argument('--no-fingerprints', action='store_true', required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--plan', action='store_true')
    action.add_argument('--run-gpu', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol, dataset = load_data(args.prepared, 'stream')
    config = configuration(args.method, args.model, protocol, args.prefetch_factor)
    config.update(task=args.task, steps=args.steps, indices=protocol['throughput'][:args.steps*4],
                  git_version=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
                  protocol=str((args.prepared/'protocol.json').resolve()),
                  timing='Same StepProfiler boundary sync, per-step scalar/logging and epoch wall for every method; phases measured separately',
                  preprocessing_implementation='Canonical UdacityDataset for every method')
    write_json(args.output/'config.json', config)
    if args.plan:
        write_json(args.output/'status.json', dict(status='PREPARED', gpu='NOT_RUN'))
        print('CPU-only GPU plan written; no device queried or initialized')
        return
    # Imports that can initialize CUDA are confined to explicitly requested GPU work.
    spec = importlib.util.spec_from_file_location('prior_engine', ROOT/'docs/experiments/autonomous-optimization-6h/scripts/common.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from hytorch.dataloaders import DataLoader
    from hytorch.profiling import StepProfiler
    from hytorch.backends.cuda import dispatch_stats
    engine = module.Engine(config['engine'], args.model)
    config.update(cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
                  started_at_utc=datetime.now(timezone.utc).isoformat())
    write_json(args.output/'config.json', config)
    data = np.load(args.prepared/'throughput-inputs.npz')
    with engine.context():
        wx, wt = engine.convert(data['x'][:4]), engine.convert(data['t'][:4])
        for _ in range(config['warmup_steps']):
            loss = engine.loss(wx, wt)
            engine.backward(loss)
            engine.update()
        engine.sync()
        engine.restore()
        dispatch_stats(reset=True)
        bank = (engine.convert(data['x']), engine.convert(data['t'])) if args.task=='compute' else None
        profiler = StepProfiler(args.output/'steps.jsonl', engine.sync,
                                'phases' if args.task=='phases' else 'throughput')
        if config['engine']=='E':
            engine.torch.cuda.reset_peak_memory_stats()
        engine.sync()
        begin = time.perf_counter()
        with ExitStack() as stack:
            iterator = None
            if bank is None:
                loader = stack.enter_context(DataLoader(dataset, 4, shuffle=False,
                    indices=config['indices'], num_workers=config['workers'],
                    prefetch_factor=config['prefetch_factor']))
                iterator = iter(loader)
            losses = []
            with (args.output/'losses.jsonl').open('x') as stream:
                for step in range(args.steps):
                    profiler.begin()
                    with profiler.phase('data'):
                        x, t = next(iterator) if iterator is not None else (bank[0][step*4:step*4+4], bank[1][step*4:step*4+4])
                    with profiler.phase('transfer'):
                        if bank is None:
                            x, t = engine.convert(x), engine.convert(t)
                    with profiler.phase('forward'):
                        loss = engine.loss(x, t)
                    with profiler.phase('backward'):
                        engine.backward(loss)
                    with profiler.phase('update'):
                        engine.update()
                    row = profiler.end(len(x), 1, step+1)
                    value = engine.value(loss)
                    assert np.isfinite(value), f'Nonfinite loss at step {step+1}'
                    losses.append(value)
                    stream.write(json.dumps(dict(step=step+1, loss=value, timestamp=row['timestamp']))+'\n')
                if iterator is not None:
                    assert next(iterator, None) is None
        engine.sync()
        wall = time.perf_counter()-begin
        np.savez(args.output/'initial_weights.npz', **engine.initial)
        np.savez(args.output/'final_weights.npz', **engine.state())
        write_json(args.output/'summary.json', dict(status='COMPLETED', samples=args.steps*4,
            wall_s=wall, samples_per_sec=args.steps*4/wall, phases=profiler.summary(),
            dispatch_counts=dispatch_stats(), final_loss=losses[-1],
            cupy_reserved_bytes=engine.cp.get_default_memory_pool().total_bytes(),
            torch_peak_bytes=engine.torch.cuda.max_memory_allocated() if config['engine']=='E' else None,
            finished_at_utc=datetime.now(timezone.utc).isoformat()))


if __name__ == '__main__':
    main()
