"""Exact real-image checks for every loader setting and unchanged data splits."""
import argparse
import json
from pathlib import Path

import numpy as np
from pipeline_common import configurations, load_data, load_module, write_csv, write_json
from hytorch.dataloaders import DataLoader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with np.load(args.prepared / 'throughput-inputs.npz') as data:
        reference = {'x': data['x'], 't': data['t']}
    rows = []
    for variant in ('existing', 'repaired', 'inplace', 'stream'):
        protocol, dataset = load_data(args.prepared, variant)
        for key, actual in zip(('train', 'validation', 'purged'), dataset.split_indices()):
            np.testing.assert_array_equal(actual, protocol[key])
        cls = load_module(args.baseline).DataLoader if variant == 'existing' else DataLoader
        for workers, prefetch in configurations():
            loader = cls(dataset, 4, shuffle=False, indices=protocol['throughput'],
                         num_workers=workers, prefetch_factor=prefetch)
            offset = 0
            for x, t in loader:
                np.testing.assert_array_equal(x, reference['x'][offset:offset+len(x)])
                np.testing.assert_array_equal(t, reference['t'][offset:offset+len(t)])
                offset += len(x)
            assert offset == 1024
            rows.append(dict(variant=variant, workers=workers, prefetch_factor=prefetch,
                             samples=offset, exact_pixels=True, exact_labels=True,
                             unchanged_split=True, status='PASS'))
    write_csv(args.output / 'exact-batches.csv', rows)
    write_json(args.output / 'summary.json', dict(status='PASS', configurations=len(rows),
        image_presentations=sum(r['samples'] for r in rows),
        comparison='direct array equality, no hashes', gpu='NOT_RUN'))
    print(json.dumps(dict(status='PASS', configurations=len(rows))), flush=True)


if __name__ == '__main__':
    main()
