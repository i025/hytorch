"""Offline protocol audit, ordered-loader equality and preprocessing measurements."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
from common import ROOT, batch, load_protocol, write_csv, write_json
from hytorch.dataloaders import DataLoader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol, dataset = load_protocol(args.prepared)
    indices = protocol['throughput']
    rows = []
    reference = np.load(args.prepared / 'throughput-inputs.npz')
    for repeat in range(3):
        order = [0, 1, 2, 4]
        order = order[repeat:] + order[:repeat]
        for workers in order:
            loader = DataLoader(dataset, 4, shuffle=False, num_workers=workers, indices=indices)
            begin = time.perf_counter()
            values = list(loader)
            elapsed = time.perf_counter() - begin
            x, t = (np.concatenate([b[i] for b in values]) for i in (0, 1))
            np.testing.assert_array_equal(x, reference['x'])
            np.testing.assert_array_equal(t, reference['t'])
            row = dict(repeat=repeat + 1, workers=workers, samples=len(x), wall_s=elapsed,
                       samples_per_sec=len(x) / elapsed, exact_pixels=True, exact_targets=True)
            rows.append(row)
            print(json.dumps(row), flush=True)
    write_csv(args.output / 'loader-timing.csv', rows)
    # Cache only CPU-preprocessed first-epoch arrays for diagnostics, not benchmarks.
    # Includes all 6330 samples and the N=2 final batch, with original order retained.
    begin = time.perf_counter()
    x, t = batch(dataset, protocol['epoch_order'][0])
    np.savez(args.output / 'epoch-1-inputs.npz', x=x, t=t,
             indices=np.array(protocol['epoch_order'][0]))
    write_json(args.output / 'summary.json', dict(status='PASSED', comparisons=len(rows),
        cached_samples=len(x), cache_wall_s=time.perf_counter() - begin,
        cache_scope='CPU-only diagnostic replay; not used for I/O throughput',
        optimizer=dict(name='SGD', learning_rate=.001, momentum=0, weight_decay=0, state='stateless'),
        original_protocol=str((args.prepared / 'protocol.json').resolve()),
        start_utc=datetime.now(timezone.utc).isoformat()))


if __name__ == '__main__':
    main()
