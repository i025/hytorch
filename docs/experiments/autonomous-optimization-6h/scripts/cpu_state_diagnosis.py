"""Offline common-state CPU replay and independent FP64 full-network reference."""
import argparse
from pathlib import Path
import time

import numpy as np
from common import Engine, ROOT, batch, compare, load_protocol, result, write_csv, write_json
from benchmarks.bench_cuda_training import torch_model, torch_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol, dataset = load_protocol(args.prepared)
    import torch
    torch.set_num_threads(4)
    engines = {m: Engine(m, device='cpu') for m in 'AE'}
    states = {'initial': engines['A'].initial}
    for method in 'AFIE':
        with np.load(ROOT / f'runs/udacity_real_window_02/train-{method}/final_weights.npz') as archive:
            states['final-' + method] = {k: archive[k] for k in archive.files}
    errors, accuracy = [], []
    begin = time.perf_counter()
    for name, state in states.items():
        for step in (1, 863, 1422):
            indices = protocol['epoch_order'][0][(step - 1) * 4:step * 4]
            x, t = batch(dataset, indices)
            values = {}
            for method, engine in engines.items():
                engine.load_state(state)
                with engine.context():
                    values[method] = result(engine, x, t)
                np.savez(args.output / f'{name}-{step}-{method}.npz', **values[method])
            for key in values['A']:
                errors.append(compare(values['A'][key], values['E'][key], key,
                                      state=name, step=step, method='CPU-HyTorch', reference='CPU-PyTorch'))
            model = torch_model('donkey', state).double()
            tx, target = torch.tensor(x, dtype=torch.float64), torch.tensor(t, dtype=torch.float64)
            prediction = model(tx)
            loss = torch.nn.functional.mse_loss(prediction, target)
            loss.backward()
            ref = dict(forward=prediction.detach().numpy(), loss=np.array(loss.item()))
            for key in state:
                layer, attr = key.split('/')
                p = getattr(getattr(model, layer), 'weight' if attr == 'W' else 'bias')
                v = p.grad.numpy()
                ref['grad/' + key] = v.T.copy() if v.ndim == 2 else v.copy()
            with torch.no_grad():
                for p in model.parameters():
                    p -= .001 * p.grad
            ref.update({'weight/' + k: v for k, v in torch_snapshot(model, state).items()})
            np.savez(args.output / f'{name}-{step}-FP64.npz', **ref)
            for method in 'AE':
                for key in ref:
                    accuracy.append(compare(values[method][key], ref[key], key, state=name,
                        step=step, method='CPU-' + method, reference='CPU-FP64'))
            print(name, step, 'complete', flush=True)
    write_csv(args.output / 'cpu-replay-errors.csv', errors)
    write_csv(args.output / 'fp64-network-errors.csv', accuracy)
    write_json(args.output / 'summary.json', dict(status='COMPLETED', wall_s=time.perf_counter() - begin,
        cpu_comparisons=len(errors), cpu_failed_groups=sum(r['mismatches'] > 0 for r in errors),
        fp64_comparisons=len(accuracy), fp64_failed_groups=sum(r['mismatches'] > 0 for r in accuracy),
        scope='CPU-only common initial/final state; not a CUDA replay or intermediate checkpoint recovery',
        gpu_status='NOT_RUN by this CPU-only command', states=list(states), batch_steps=[1, 863, 1422]))


if __name__ == '__main__':
    main()
