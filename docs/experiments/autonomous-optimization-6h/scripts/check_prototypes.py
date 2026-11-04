"""Real Udacity DonkeyNet/SpatialNet forward, gradients, updates and opt-in parity."""
import argparse
import json
from pathlib import Path

import numpy as np
from common import Engine, compare, load_protocol, result, write_csv, write_json
from hytorch.dataloaders import DataLoader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--no-fingerprints', action='store_true', required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol, dataset = load_protocol(args.prepared)
    data = np.load(args.prepared / 'throughput-inputs.npz')
    batches = list(DataLoader(dataset, 4, shuffle=False, indices=protocol['throughput'][:12], num_workers=2))
    for step, (x, t) in enumerate(batches):
        np.testing.assert_array_equal(x, data['x'][step * 4:step * 4 + 4])
        np.testing.assert_array_equal(t, data['t'][step * 4:step * 4 + 4])
    errors = []
    for model in ('donkey', 'spatial'):
        engines = {m: Engine(m, model) for m in 'AFIRJE'}
        for step, (x, t) in enumerate(batches, 1):
            values = {}
            for method, engine in engines.items():
                with engine.context():
                    values[method] = result(engine, x, t)
                np.savez(args.output / f'{model}-{step}-{method}.npz', **values[method])
            for method in 'AFIRJ':
                for reference in ('A', 'I', 'E'):
                    for key, value in values[method].items():
                        row = compare(value, values[reference][key], key, model=model, step=step,
                                      method=method, reference=reference)
                        errors.append(row)
            for method in 'RJ':
                for key in values['I']:
                    np.testing.assert_array_equal(values[method][key], values['I'][key], err_msg=key)
                    assert values[method][key].tobytes() == values['I'][key].tobytes(), key
    write_csv(args.output / 'errors.csv', errors)
    passed = all(r['mismatches'] == 0 and r['finite'] for r in errors)
    write_json(args.output / 'summary.json', dict(status='PASSED' if passed else 'FAILED',
        comparisons=len(errors), max_abs=max(r['max_abs'] for r in errors),
        r_j_i_bitwise_equal=True, models=['donkey', 'spatial'], input='Real Udacity RGB, unchanged protocol'))
    assert passed, 'Full arrays and strict error rows retained'


if __name__ == '__main__':
    main()
