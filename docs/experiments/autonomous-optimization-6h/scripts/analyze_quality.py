"""Separate offline model-quality observations; no change to data/training protocol."""
import argparse
import csv
from pathlib import Path

import numpy as np
from common import ROOT, write_csv, write_json


def read(path):
    return list(csv.DictReader(path.open()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    old = ROOT / 'docs/experiments/udacity-real-training'
    manifest = read(ROOT / 'runs/udacity_real_prepared_01/manifest.csv')
    distribution = []
    for split in ('train', 'validation', 'purged'):
        values = np.array([float(r['steering']) for r in manifest if r['split'] == split])
        if not len(values):
            continue
        distribution.append(dict(split=split, count=len(values), zero_fraction=float(np.mean(values == 0)),
            mean=float(values.mean()), std=float(values.std()), min=float(values.min()), max=float(values.max()),
            negative_fraction=float(np.mean(values < 0)), positive_fraction=float(np.mean(values > 0))))
    predictions = read(old / 'results/acceptance-01/validation-predictions.csv')
    rows, bins = [], []
    for method in 'AFIE':
        for epoch in range(4):
            rr = [r for r in predictions if r['method'] == method and int(r['epoch']) == epoch]
            x, y = np.array([float(r['prediction']) for r in rr]), np.array([float(r['target']) for r in rr])
            variance_x, variance_y = np.var(x), np.var(y)
            covariance = np.mean((x - x.mean()) * (y - y.mean()))
            mse = np.mean((x - y)**2)
            reconstructed = (x.mean() - y.mean())**2 + variance_x + variance_y - 2 * covariance
            assert abs(mse - reconstructed) < 1e-12
            rows.append(dict(method=method, epoch=epoch, samples=len(x), prediction_mean=float(x.mean()),
                prediction_std=float(x.std()), target_mean=float(y.mean()), target_std=float(y.std()),
                pearson=float(np.corrcoef(x, y)[0, 1]), mse=float(mse), zero_mse=float(np.mean(y * y)),
                mse_minus_zero=float(mse - np.mean(y * y)), bias_squared=float((x.mean() - y.mean())**2),
                prediction_variance=float(variance_x), target_variance=float(variance_y),
                minus_two_covariance=float(-2 * covariance)))
            masks = [('zero', y == 0), ('small_nonzero', (abs(y) > 0) & (abs(y) < .05)),
                     ('medium', (abs(y) >= .05) & (abs(y) < .2)), ('large', abs(y) >= .2)]
            for label, mask in masks:
                if not mask.any():
                    continue
                bins.append(dict(method=method, epoch=epoch, label_bin=label, samples=int(mask.sum()),
                    fraction=float(mask.mean()), mae=float(np.mean(abs(x[mask] - y[mask]))),
                    mse=float(np.mean((x[mask] - y[mask])**2)), zero_mse=float(np.mean(y[mask]**2))))
    write_csv(args.output / 'label-distribution.csv', distribution)
    write_csv(args.output / 'prediction-moments.csv', rows)
    write_csv(args.output / 'steering-bins.csv', bins)
    write_json(args.output / 'summary.json', dict(status='COMPLETED', original_data_unchanged=True,
        source=str(old / 'results/acceptance-01/validation-predictions.csv'),
        scope='Historical held-out validation observations; no hyperparameter or split changes, no closed-loop driving claim'))


if __name__ == '__main__':
    main()
