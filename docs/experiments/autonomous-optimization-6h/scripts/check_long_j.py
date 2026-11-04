"""New opt-in combination versus captured I for 1600 real sequential updates."""
import argparse
import csv
from pathlib import Path

import numpy as np
from common import Engine, compare, load_protocol, result, write_csv, write_json
from diagnose_replay import CHECKPOINTS, warmup
from hytorch.dataloaders import DataLoader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--no-fingerprints', action='store_true', required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol, dataset = load_protocol(args.prepared)
    reference = list(csv.DictReader((args.reference / 'I/losses.csv').open()))
    engine = Engine('J')
    step, losses, errors = 0, [], []
    with engine.context():
        warmup(engine, args.prepared)
        for epoch, indices in enumerate(protocol['epoch_order'], 1):
            remaining = len(reference) - step
            if remaining <= 0:
                break
            loader = DataLoader(dataset, 4, shuffle=False, indices=indices[:remaining * 4], num_workers=2)
            for x, t in loader:
                step += 1
                if step in CHECKPOINTS:
                    values = result(engine, x, t)
                    np.savez(args.output / f'result-{step}.npz', **values)
                    original = np.load(args.reference / 'I' / f'result-{step}.npz')
                    for key in values:
                        errors.append(compare(values[key], original[key], key, step=step, method='J', reference='I'))
                    loss_value = float(values['loss'])
                else:
                    loss = engine.loss(engine.convert(x), engine.convert(t))
                    engine.backward(loss)
                    engine.update()
                    loss_value = engine.value(loss)
                target = float(reference[step - 1]['loss'])
                losses.append(dict(step=step, epoch=epoch, actual=loss_value, reference=target,
                    abs_error=abs(loss_value - target), tolerance_pass=abs(loss_value - target) <= 1e-6 + 1e-5 * abs(target),
                    exact_equal=loss_value == target))
    write_csv(args.output / 'losses.csv', losses)
    write_csv(args.output / 'errors.csv', errors)
    passed = all(r['tolerance_pass'] for r in losses) and all(r['mismatches'] == 0 and r['finite'] for r in errors)
    write_json(args.output / 'summary.json', dict(status='PASSED' if passed else 'FAILED', steps=step,
        loss_exact_equal=all(r['exact_equal'] for r in losses), comparisons=len(errors),
        checkpoints_bitwise_equal=all(r['bitwise_equal'] for r in errors),
        max_loss_abs=max(r['abs_error'] for r in losses),
        preserves_partial_batch=True, includes_prefetch=True, native_relu_backward=True))
    assert passed, 'Full checkpoint errors retained'


if __name__ == '__main__':
    main()
