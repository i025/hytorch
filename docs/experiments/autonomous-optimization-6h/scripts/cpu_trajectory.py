"""Does long FP32 drift also occur with no CUDA kernels at all?"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from common import Engine, compare, load_protocol, result, write_csv, write_json
from diagnose_replay import training_batches, CHECKPOINTS
from benchmarks.bench_cuda_training import torch_model, torch_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol, dataset = load_protocol(args.prepared)
    with np.load(args.cache) as archive:
        cache = {key: archive[key] for key in archive.files}
    engines = {m: Engine(m, device='cpu') for m in 'AE'}
    torch = engines['E'].torch
    double = torch_model('donkey', engines['A'].initial).double()
    optimizer = torch.optim.SGD(double.parameters(), lr=.001)
    losses, errors, local = [], [], []
    initial = engines['A'].initial
    begin = time.perf_counter()
    for step, epoch, indices, x, t in training_batches(protocol, cache, dataset, 1600):
        values = {}
        for method, engine in engines.items():
            with engine.context():
                loss = engine.loss(x if method == 'A' else engine.convert(x),
                                   t if method == 'A' else engine.convert(t))
                engine.backward(loss)
                engine.update()
            values[method] = engine.value(loss)
        optimizer.zero_grad(set_to_none=True)
        loss64 = torch.nn.functional.mse_loss(double(torch.tensor(x, dtype=torch.float64)),
                                              torch.tensor(t, dtype=torch.float64))
        loss64.backward()
        optimizer.step()
        values['FP64'] = loss64.item()
        losses.extend(dict(method=m, step=step, epoch=epoch, loss=value) for m, value in values.items())
        if step in CHECKPOINTS:
            states = {m: e.state() for m, e in engines.items()}
            states['FP64'] = torch_snapshot(double, initial)
            for method, reference in [('A', 'E'), ('A', 'FP64'), ('E', 'FP64')]:
                for key in initial:
                    errors.append(compare(states[method][key], states[reference][key], 'weight/' + key,
                                          method=method, reference=reference, step=step))
            # Reset two temporary CPU engines to the SAME current FP32 state.
            # This does not alter the ongoing free trajectories.
            common = states['A']
            replay = {}
            for method in 'AE':
                engine = Engine(method, device='cpu')
                engine.load_state(common)
                with engine.context():
                    replay[method] = result(engine, x, t)
            for key in replay['A']:
                local.append(compare(replay['A'][key], replay['E'][key], key,
                                     method='A', reference='E', step=step, scope='same-state-CPU'))
            np.savez(args.output / f'cpu-state-{step}.npz', **common)
        if step % 256 == 0:
            print(step, json.dumps(values), flush=True)
    write_csv(args.output / 'losses.csv', losses)
    write_csv(args.output / 'weight-errors.csv', errors)
    write_csv(args.output / 'local-errors.csv', local)
    for m, engine in engines.items():
        np.savez(args.output / ('final-' + m + '.npz'), **engine.state())
    np.savez(args.output / 'final-FP64.npz', **torch_snapshot(double, initial))
    loss_errors = []
    for method, reference in [('A', 'E'), ('A', 'FP64'), ('E', 'FP64')]:
        a, b = ([r['loss'] for r in losses if r['method'] == m] for m in (method, reference))
        delta = np.abs(np.array(a) - b)
        failed = np.flatnonzero(delta > 1e-6 + 1e-5 * np.abs(b))
        loss_errors.append(dict(method=method, reference=reference, mismatches=len(failed),
            first_bad_step=int(failed[0]) + 1 if len(failed) else None, max_abs=float(delta.max())))
    write_csv(args.output / 'loss-errors.csv', loss_errors)
    write_json(args.output / 'summary.json', dict(status='COMPLETED', steps=1600, wall_s=time.perf_counter() - begin,
        loss_errors=loss_errors, local_comparisons=len(local), local_failed_groups=sum(r['mismatches'] > 0 for r in local),
        scope='CPU-only trajectory; different FP32 BLAS/torch CPU reductions, independent FP64 trajectory; no GPU inference'))


if __name__ == '__main__':
    main()
