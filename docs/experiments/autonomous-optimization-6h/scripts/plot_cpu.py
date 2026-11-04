"""Scientific plots from retained CSVs; these are data visualizations, not screenshots."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from common import ROOT


def read(path):
    return list(csv.DictReader(path.open()))


def save(fig, path):
    fig.text(.5, .005, 'Measured CSV data visualization; not a terminal screenshot', ha='center', fontsize=8)
    fig.tight_layout(rect=[0, .025, 1, 1])
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    losses = read(args.raw / 'cpu-trajectory/losses.csv')
    values = {m: np.array([float(r['loss']) for r in losses if r['method'] == m]) for m in ('A', 'E', 'FP64')}
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for m, ref in [('A', 'E'), ('A', 'FP64'), ('E', 'FP64')]:
        ratio = abs(values[m] - values[ref]) / (1e-6 + 1e-5 * abs(values[ref]))
        axes[0].semilogy(np.arange(len(ratio)) + 1, np.maximum(ratio, 1e-5), label=m + '/' + ref)
    axes[0].axhline(1, color='black', ls='--')
    axes[0].set_ylabel('Loss error / unchanged tolerance')
    axes[0].set_title('CPU-only free trajectories: FP32 drift without custom CUDA')
    axes[0].legend()
    local = read(args.raw / 'cpu-trajectory/local-errors.csv')
    steps = sorted({int(r['step']) for r in local})
    for prefix, label in [('forward', 'Forward'), ('grad/', 'Gradients'), ('weight/', 'Updated weights')]:
        maxima = [max(float(r['max_abs']) for r in local if int(r['step']) == s and r['value'].startswith(prefix)) for s in steps]
        axes[1].semilogy(steps, np.maximum(maxima, 1e-12), 'o-', label=label)
    axes[1].set_ylabel('Same-state max absolute error')
    axes[1].set_xlabel('Training step')
    axes[1].legend()
    save(fig, args.output / '01-cpu-long-drift-and-local-replay.png')

    old = read(ROOT / 'docs/experiments/udacity-real-training/results/acceptance-01/losses.csv')
    history = {m: np.array([float(r['loss']) for r in old if r['method'] == m and r['task'] == 'train']) for m in 'AFIE'}
    fig, ax = plt.subplots(figsize=(10, 4))
    for m, ref in [('A', 'E'), ('F', 'I'), ('I', 'E')]:
        ratio = abs(history[m] - history[ref]) / (1e-6 + 1e-5 * abs(history[ref]))
        ax.semilogy(np.arange(len(ratio)) + 1, np.maximum(ratio, 1e-5), label=m + '/' + ref)
    ax.axhline(1, color='black', ls='--')
    ax.set(xlabel='Training step', ylabel='Loss error / unchanged tolerance',
           title='Historical stage 4H GPU trajectories (separate shared-GPU window)')
    ax.legend()
    save(fig, args.output / '02-historical-gpu-long-drift.png')

    weights = read(args.raw / 'cpu-trajectory/weight-errors.csv')
    fig, ax = plt.subplots(figsize=(9, 4))
    for m, ref in [('A', 'E'), ('A', 'FP64'), ('E', 'FP64')]:
        ax.semilogy(steps, [max(float(r['max_abs']) for r in weights if int(r['step']) == s and r['method'] == m and r['reference'] == ref) for s in steps], 'o-', label=m + '/' + ref)
    ax.set(xlabel='Training step', ylabel='Free-trajectory max parameter error', title='Small differences exist before the first strict loss failure')
    ax.legend()
    save(fig, args.output / '03-parameter-drift.png')

    rows = read(args.raw / 'cpu-preparation/loader-timing.csv')
    fig, ax = plt.subplots(figsize=(7, 4))
    for w in [0, 1, 2, 4]:
        values = [float(r['samples_per_sec']) for r in rows if int(r['workers']) == w]
        ax.bar(str(w), np.mean(values), yerr=np.std(values, ddof=1), capsize=5, alpha=.7)
        ax.scatter([str(w)] * len(values), values, color='black', s=20)
    ax.set(xlabel='Existing loader worker threads (0 = serial)', ylabel='Real images / second',
           title='CPU preprocessing only; three rotating repeats, exact pixels/labels')
    save(fig, args.output / '04-cpu-prefetch.png')

    rows = read(args.raw / 'quality-analysis/prediction-moments.csv')
    rows = [r for r in rows if r['method'] == 'I']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = [int(r['epoch']) for r in rows]
    axes[0].plot(epochs, [float(r['prediction_mean']) for r in rows], 'o-', label='Prediction mean')
    axes[0].axhline(float(rows[0]['target_mean']), color='red', label='Validation target mean')
    axes[0].set(xlabel='Epoch', ylabel='Steering', title='Historical I: mean prediction vs held-out labels')
    axes[0].legend()
    axes[1].plot(epochs, [float(r['mse']) for r in rows], 'o-', label='Model MSE')
    axes[1].axhline(float(rows[0]['zero_mse']), color='black', ls='--', label='Zero steering MSE')
    axes[1].set(xlabel='Epoch', ylabel='Validation MSE', title='Quality observations, separate from CUDA performance')
    axes[1].legend()
    save(fig, args.output / '05-quality-observations.png')


if __name__ == '__main__':
    main()
