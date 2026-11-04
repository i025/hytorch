"""Matched compute, complete ordered-I/O, phase and Nsight measurements."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from common import Engine, ROOT, load_protocol, write_json
from hytorch.dataloaders import DataLoader
from hytorch.backends.cuda import dispatch_stats
from hytorch.profiling import StepProfiler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=list('AFIPRJE'), required=True)
    parser.add_argument('--model', choices=['donkey', 'spatial'], default='donkey')
    parser.add_argument('--task', choices=['compute', 'io', 'phases', 'trace'], required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=256)
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--no-fingerprints', action='store_true', required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol, dataset = load_protocol(args.prepared)
    workers = (2 if args.method in ('P', 'J') else 0) if args.workers is None else args.workers
    engine = Engine(args.method, args.model)
    trace = args.task == 'trace'
    phases = args.task == 'phases'
    config = dict(method=args.method, model=args.model, task=args.task, workers=workers,
        git_version=subprocess.check_output(['git', 'log', '-1', '--format=%H'], cwd=ROOT, text=True).strip(),
        steps=args.steps, warmup_steps=10, batch=4, dtype='float32', seed=42, dropout=0,
        tf32=False, learning_rate=.001, optimizer='SGD', no_fingerprints=True,
        protocol=str((args.prepared / 'protocol.json').resolve()),
        indices=protocol['throughput'][:args.steps * 4],
        cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
        resource_condition=os.environ['HYTORCH_RESOURCE_CONDITION'],
        started_at=datetime.now(timezone.utc).isoformat(),
        timing='Epoch wall includes iterator startup, all batches, file logging and final drain; no Nsight for formal throughput',
        j_definition='I + opt-in native ReLU backward + existing ordered CPU AsyncDataLoader, two workers',
        p_definition='I + prefetch only', r_definition='I + native ReLU backward only')
    write_json(args.output / 'config.json', config)
    np.savez(args.output / 'initial_weights.npz', **engine.initial)
    data = np.load(args.prepared / 'throughput-inputs.npz')
    with ExitStack() as stack:
        recorder = None
        if trace:
            spec = importlib.util.spec_from_file_location('autonomous_ranges', ROOT / 'docs/experiments/conv-dweight-fusion/scripts/profile_training.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            recorder = stack.enter_context(module.WeightRanges(engine.cp.cuda.nvtx.RangePush, engine.cp.cuda.nvtx.RangePop, engine.cp))
            class TraceDataset:
                def __len__(self):
                    return len(dataset_source)

                def __getitem__(self, index):
                    # NVTX annotates CPU decode workers; it submits no CUDA work.
                    engine.cp.cuda.nvtx.RangePush('DATA/preprocess|row=' + str(index))
                    try:
                        return dataset_source[index]
                    finally:
                        engine.cp.cuda.nvtx.RangePop()
            dataset_source = dataset
            dataset = TraceDataset()
        stack.enter_context(engine.context())
        wx, wt = engine.convert(data['x'][:4]), engine.convert(data['t'][:4])
        for _ in range(10):
            loss = engine.loss(wx, wt)
            engine.backward(loss)
            engine.update()
        engine.sync()
        engine.restore()
        dispatch_stats(reset=True)
        bank = (engine.convert(data['x']), engine.convert(data['t'])) if args.task == 'compute' else None
        loader = None if bank is not None else DataLoader(dataset, 4, shuffle=False,
            num_workers=workers, indices=config['indices'])
        profiler = StepProfiler(args.output / 'steps.jsonl', engine.sync,
                                'phases' if phases else 'throughput', trace)
        if args.method == 'E':
            engine.torch.cuda.reset_peak_memory_stats()
        engine.sync()
        begin = time.perf_counter()
        iterator = iter(loader) if loader is not None else None
        losses = []
        with (args.output / 'losses.jsonl').open('x') as loss_log:
            for step in range(args.steps):
                profiler.begin()
                with profiler.phase('data'):
                    if bank is None:
                        x, t = next(iterator)
                    else:
                        x, t = bank[0][step * 4:step * 4 + 4], bank[1][step * 4:step * 4 + 4]
                with profiler.phase('transfer'):
                    if bank is None:
                        x, t = engine.convert(x), engine.convert(t)
                with profiler.phase('forward'):
                    loss = engine.loss(x, t)
                with profiler.phase('backward'):
                    engine.backward(loss)
                with profiler.phase('update'):
                    engine.update()
                row = profiler.end(len(x), 1, step + 1)
                value = engine.value(loss)
                assert np.isfinite(value), f'Nonfinite loss at step {step + 1}'
                losses.append(value)
                loss_log.write(json.dumps(dict(step=step + 1, loss=value, timestamp=row['timestamp'])) + '\n')
            if iterator is not None:
                assert next(iterator, None) is None
        engine.sync()
        wall = time.perf_counter() - begin
        final = engine.state()
        np.savez(args.output / 'final_weights.npz', **final)
        write_json(args.output / 'summary.json', dict(status='COMPLETED', wall_s=wall,
            samples=args.steps * 4, samples_per_sec=args.steps * 4 / wall,
            measured_steps=profiler.summary(), final_loss=losses[-1], dispatch_counts=dispatch_stats(),
            cupy_pool_used_bytes=engine.cp.get_default_memory_pool().used_bytes(),
            cupy_pool_reserved_bytes=engine.cp.get_default_memory_pool().total_bytes(),
            torch_peak_allocated_bytes=engine.torch.cuda.max_memory_allocated() if args.method == 'E' else None,
            finished_at=datetime.now(timezone.utc).isoformat()))
        if recorder is not None:
            write_json(args.output / 'instrumentation.json', recorder.summary())
        print(json.dumps(dict(method=args.method, model=args.model, task=args.task,
                             wall_s=wall, samples_per_sec=args.steps * 4 / wall)), flush=True)


if __name__ == '__main__':
    main()
