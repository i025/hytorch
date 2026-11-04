"""Actual matched GPU data and CUPTI plots, strictly separate timing scopes."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from common import ROOT
from plot_cpu import save


def read(path):
    return list(csv.DictReader(path.open()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    stats = read(args.results / 'timing-summary.csv')
    gains = read(args.results / 'speedups.csv')
    raw = read(args.results / 'training-raw.csv')
    methods = ['A', 'F', 'I', 'P', 'R', 'J', 'E', 'Eprefetch']
    for task in ('compute', 'io'):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, model in zip(axes, ('donkey', 'spatial')):
            rr = [r for r in stats if r['model'] == model and r['task'] == task and r['metric'] == 'samples_per_sec']
            present = [m for m in methods if any(r['method'] == m for r in rr)]
            values = [next(r for r in rr if r['method'] == m) for m in present]
            ax.bar(present, [float(r['mean']) for r in values], yerr=[float(r['sd']) for r in values], capsize=4, alpha=.7)
            for m in present:
                points = [float(r['samples_per_sec']) for r in raw if r['method'] == m and r['model'] == model and r['task'] == task]
                ax.scatter([m] * len(points), points, s=15, color='black')
            ax.set(title=model + ': ' + task, ylabel='Samples / second')
        fig.suptitle('Uninstrumented matched training, independent process repeats')
        save(fig, args.output / ('06-throughput-' + task + '.png'))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, model in zip(axes, ('donkey', 'spatial')):
        rr = [r for r in stats if r['model'] == model and r['task'] == 'phases']
        bottom = np.zeros(7)
        for phase in ('data', 'transfer', 'forward', 'backward', 'update'):
            values = [float(next(r for r in rr if r['method'] == m and r['metric'] == phase + '_ms')['mean']) for m in methods[:7]]
            ax.bar(methods[:7], values, bottom=bottom, label=phase)
            bottom += values
        ax.set(title=model + ': independent synchronized phases', ylabel='ms / step')
    axes[1].legend()
    save(fig, args.output / '07-synchronized-phases.png')
    for task in ('compute', 'io'):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, model in zip(axes, ('donkey', 'spatial')):
            comparisons = ['P/I', 'R/I', 'J/I', 'J/A', 'J/E'] + (['J/Eprefetch'] if task == 'io' else [])
            rr = [next(r for r in gains if r['model'] == model and r['task'] == task and r['comparison'] == c) for c in comparisons]
            ax.bar(comparisons, [float(r['throughput_gain_pct']) for r in rr])
            ax.axhline(0, color='black')
            ax.set(title=model + ': ' + task, ylabel='Throughput difference (%)')
        fig.suptitle('Ablation, incremental, cumulative and PyTorch comparisons: same GPU window')
        save(fig, args.output / ('08-gains-' + task + '.png'))

    micro = read(args.raw / 'relu-micro/raw.csv')
    kernel = read(args.results / 'micro-kernel-times.csv')
    cases = sorted({int(r['case']) for r in micro})
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for mode, offset in [('original', -.18), ('fused', .18)]:
        means = lambda metric: [np.mean([float(r[metric]) for r in micro if int(r['case']) == c and r['mode'] == mode]) for c in cases]
        pure = [np.mean([float(r['kernel_us']) for r in kernel if int(r['case']) == c and r['scope'] == mode]) for c in cases]
        axes[0].bar(np.array(cases) + offset, pure, width=.35, label=mode)
        axes[1].bar(np.array(cases) + offset, means('event_span_us'), width=.35, label=mode)
        axes[2].bar(np.array(cases) + offset, means('synchronized_host_us'), width=.35, label=mode)
    for ax in axes:
        ax.legend()
    axes[0].set(ylabel='CUPTI kernel sum (us)', title='ReLU backward: pure kernel, full event span and synchronized host scopes')
    axes[1].set_ylabel('Full event span (us)')
    axes[2].set(xlabel='Generic shape case; final case is transposed (copy cost included)', ylabel='Host call + sync (us)')
    save(fig, args.output / '09-relu-micro-scopes.png')

    phases = read(args.results / 'trace-phases.csv')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels = []
    for model in ('donkey', 'spatial'):
        for method in 'IJ':
            row = next(r for r in phases if r['run'] == f'trace-{model}-{method}' and r['phase'] == 'backward')
            label = model + '-' + method
            labels.append(label)
            axes[0].bar(label, float(row['gpu_kernel_count']) / 32)
            axes[1].bar(label, float(row['gpu_kernel_sum_ms']) * 1000 / 32)
    axes[0].set(ylabel='CUPTI kernels / step', title='Backward launch count (separate Nsight run)')
    axes[1].set(ylabel='CUPTI kernel sum us / step', title='Backward GPU execution sum')
    save(fig, args.output / '10-kernel-launches-and-time.png')

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for ax, method in zip(axes, 'IJ'):
        folder = args.analysis / ('trace-donkey-' + method)
        kernels = read(folder / 'kernel-events.csv')
        ranges = read(folder / 'nvtx-events.csv')
        summary = __import__('json').loads((folder / 'profiling-summary.json').read_text())
        lo = int(summary['training_envelope_start_ns'])
        for phase, y, color in [('data', 4, 'tab:gray'), ('transfer', 3, 'tab:purple'), ('forward', 2, 'tab:blue'), ('backward', 1, 'tab:orange'), ('update', 0, 'tab:green')]:
            rr = [r for r in ranges if r['name'] == phase]
            ax.broken_barh([((int(r['start']) - lo) / 1e6, float(r['duration_ms'])) for r in rr], (y - .15, .3), facecolors=color)
        for r in ranges:
            if r['name'].startswith('DATA/preprocess'):
                ax.broken_barh([((int(r['start']) - lo) / 1e6, float(r['duration_ms']))], (5.0, .3), facecolors='tab:pink')
        rr = [r for r in kernels if r['phase'] in ('forward', 'backward', 'update')]
        ax.broken_barh([((int(r['start']) - lo) / 1e6, float(r['duration_ms'])) for r in rr], (-1.15, .3), facecolors='black')
        ax.set_yticks([-1, 0, 1, 2, 3, 4, 5.15])
        ax.set_yticklabels(['GPU kernels', 'Update', 'Backward', 'Forward', 'H2D', 'Data wait', 'CPU decode'])
        ax.set(title='DonkeyNet ' + method + ': first 50ms of separate Nsight trace', xlim=(0, 50), ylim=(-1.6, 5.6))
    axes[-1].set_xlabel('ms from first training phase; gaps are this process only, not whole-device utilization')
    save(fig, args.output / '11-host-gpu-timeline.png')


if __name__ == '__main__':
    main()
