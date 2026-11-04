"""Independent sampled FP64 dot on preserved, identical real dWeight operands."""
import argparse
import json
from pathlib import Path

import numpy as np
from common import write_csv, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for path in sorted(args.raw.glob('dweight-operands-*.npz')):
        data = np.load(path)
        x, gy = data['x'], data['gy']
        n, c, h, w = x.shape
        _, oc, oh, ow = gy.shape
        kh, kw = data['kernel']
        sh, sw = data['stride']
        ph, pw = data['pad']
        rng = np.random.default_rng(42)
        total = data['dw'].size
        random = rng.choice(total, size=min(100, total), replace=False)
        worst = np.argsort(abs(data['dw'].ravel().astype(np.float64) - data['original'].ravel()))[-20:]
        near = np.argsort(abs(data['dw'].ravel()))[:20]
        indices = np.unique(np.concatenate([random, worst, near]))
        nh = np.arange(oh)[:, None] * sh
        nw = np.arange(ow)[None, :] * sw
        for index in indices:
            o, ci, r, s = np.unravel_index(index, data['dw'].shape)
            ih, iw = nh + r - ph, nw + s - pw
            valid = (ih >= 0) & (ih < h) & (iw >= 0) & (iw < w)
            operand = np.zeros((n, oh, ow), np.float64)
            ih, iw = np.broadcast_to(ih, (oh, ow)), np.broadcast_to(iw, (oh, ow))
            operand[:, valid] = x[:, ci, ih[valid], iw[valid]]
            left = gy[:, o].astype(np.float64).ravel()
            right = operand.ravel()
            reference = float(np.dot(left, right))
            absolute_sum = float(np.sum(abs(left * right)))
            for method in ('dw', 'original', 'torch'):
                value = float(data[method].ravel()[index])
                delta = abs(value - reference)
                rows.append(dict(file=path.name, method=method, index=int(index), shape=str(x.shape),
                    reduction_size=n * oh * ow, value=value, fp64_reference=reference,
                    abs_error=delta, relative_error=delta / max(abs(reference), 1e-30),
                    cancellation_ratio=abs(reference) / max(absolute_sum, 1e-30),
                    near_zero=abs(reference) < 1e-5,
                    tolerance_pass=delta <= 1e-5 + 1e-4 * abs(reference),
                    sample_kind='random+largest-FI-difference+near-zero'))
    write_csv(args.output / 'samples.csv', rows)
    summaries = []
    for method in ('dw', 'original', 'torch'):
        rr = [r for r in rows if r['method'] == method]
        summaries.append(dict(method=method, samples=len(rr), max_abs=max(r['abs_error'] for r in rr),
            mean_abs=float(np.mean([r['abs_error'] for r in rr])), failed=sum(not r['tolerance_pass'] for r in rr)))
    write_csv(args.output / 'summary.csv', summaries)
    write_json(args.output / 'summary.json', dict(status='COMPLETED', operand_sets=len(set(r['file'] for r in rows)),
        scope='Sampled scalar FP64 dot, not full-matrix validation', methods=summaries))


if __name__ == '__main__':
    main()
