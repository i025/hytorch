"""Scientific figures from retained CSV/JSON values; these are not screenshots."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--results', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
args = p.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
rows = list(csv.DictReader((args.results / 'samples.csv').open()))
summary = json.loads((args.results / 'summary.json').read_text())
methods = ['numpy_fp32', 'sequential_fma', 'pairwise_fp32', 'block32', 'block64', 'block128', 'fp64_acc_fp32_out']
labels = ['NumPy\nFP32', 'Sequential\nFMA', 'Pairwise\nFP32', 'Block\n32', 'Block\n64', 'Block\n128', 'FP64 acc.\nFP32 out.']
colors = ['#687588', '#c83c36', '#008568', '#2764b2', '#8955ad', '#c18415', '#343b43']
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.dpi': 180, 'axes.titleweight': 'bold'})
source = 'Source: results/samples.csv and summary.json | Stratified samples, not full-matrix acceptance | Data visualization, not a screenshot'


def save(fig, name):
    fig.text(0.02, 0.012, source, fontsize=8, color='#555555')
    target = args.output / name
    with target.open('xb') as stream:
        fig.savefig(stream, format='png', facecolor='white')
    plt.close(fig)
    print(target)


fig, axes = plt.subplots(1, 3, figsize=(16, 5.7), sharey=True)
for ax, n in zip(axes, (1024, 2048, 4096)):
    data = [[float(r['abs_error_fp64']) for r in rows if int(r['n']) == n and r['strategy'] == method] for method in methods]
    zeros = [sum(v == 0 for v in group) for group in data]
    boxes = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False,
                       medianprops={'color': 'black'}, widths=0.6)
    for patch, color in zip(boxes['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.scatter(range(1, 8), [np.mean(v) for v in data], marker='D', s=22, color='black', zorder=3, label='Mean')
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.2)
    ax.set_title(f'K = N = {n} | {len(data[0])} positions')
    ax.tick_params(axis='x', labelsize=8)
    ax.text(0.03, 0.03, 'Zero errors by strategy: ' + ', '.join(map(str, zeros)), transform=ax.transAxes, fontsize=8)
axes[0].set_ylabel('Absolute error relative to FP64 dot')
axes[0].legend(frameon=False)
fig.suptitle('Accumulation accuracy on identical FP32 inputs', fontsize=16)
fig.text(0.5, 0.07, 'Boxes: 25–75%; line: median; whiskers: 1.5 IQR; outliers omitted from display only; diamonds: mean\nLog axis: exact-zero errors counted in annotations; all values retained in statistics', ha='center', fontsize=9)
fig.tight_layout(rect=(0, 0.12, 1, 0.94))
save(fig, 'absolute-errors.png')

fig, axes = plt.subplots(1, 2, figsize=(12, 5.7))
for method, label, color in zip(methods, labels, colors):
    groups = sorted((g for g in summary['cpu']['groups'] if g['stratum'] == 'all' and g['strategy'] == method), key=lambda g: g['n'])
    for ax, key in zip(axes, ('mean_abs_error', 'p95_abs_error')):
        ax.plot([g['n'] for g in groups], [g[key] for g in groups], 'o-', color=color, label=label.replace('\n', ' '))
for ax, title in zip(axes, ('Mean absolute error', '95th percentile absolute error')):
    ax.set(title=title, xlabel='K = N', ylabel='Error relative to FP64 dot', yscale='log', xticks=[1024, 2048, 4096])
    ax.grid(alpha=0.2)
axes[1].legend(fontsize=8, frameon=False)
fig.suptitle('Accuracy versus reduction length | 518 / 515 / 515 sampled positions', fontsize=14)
fig.tight_layout(rect=(0, 0.065, 1, 0.93))
save(fig, 'error-vs-k.png')

worst = [r for r in rows if r['n'] == '4096' and r['stratum'] == 'historical_worst' and r['strategy'] == 'sequential_fma']
fig, axes = plt.subplots(1, len(worst), figsize=(16, 5.8), sharey=True)
for ax, point in zip(axes, worst):
    selected = {r['strategy']: r for r in rows if r['n'] == '4096' and (r['row'], r['col']) == (point['row'], point['col'])}
    ref = float(point['reference_fp64'])
    values = [1e6 * (float(selected[m]['value']) - ref) for m in methods]
    ax.bar(range(7), values, color=colors)
    ax.axhline(0, color='black', lw=0.7)
    tolerance = 1e6 * (1e-4 + 1e-4 * abs(ref))
    for sign in (-1, 1):
        ax.axhline(sign * tolerance, color='#999999', ls='--', lw=0.8)
    ax.set_xticks(range(7), labels, fontsize=8)
    ax.set_title(f"Position ({point['row']}, {point['col']})\nFP64 dot = {ref:.12g}")
    ax.grid(axis='y', alpha=0.2)
    for i, value in enumerate(values):
        ax.annotate(f'{value:.3g}', (i, value), xytext=(0, 4 if value >= 0 else -13), textcoords='offset points', ha='center', fontsize=8)
axes[0].set_ylabel('Signed output difference from FP64 dot (× 1e−6)')
fig.suptitle('Three historical worst positions at N = 4096', fontsize=16)
fig.text(0.5, 0.07, 'Sequential FMA = all seven GPU methods at these positions; dashed lines: unchanged 1e−4 + 1e−4 |FP64 dot|', ha='center', fontsize=9)
fig.tight_layout(rect=(0, 0.12, 1, 0.93))
save(fig, 'historical-worst.png')

fig, axes = plt.subplots(1, 3, figsize=(16, 5.7), sharey=True)
strata = [('random_normal', '#2764b2'), ('near_zero', '#c18415'), ('historical_worst', '#c83c36')]
for ax, n in zip(axes, (1024, 2048, 4096)):
    for stratum, color in strata:
        group = [r for r in rows if int(r['n']) == n and r['strategy'] == 'sequential_fma' and r['stratum'] == stratum]
        ax.scatter([float(r['cancellation_condition']) for r in group], [float(r['relative_error_fp64']) for r in group],
                   s=15, alpha=0.65, color=color, label=stratum.replace('_', ' '))
    ax.set(xscale='log', yscale='log', xlabel='Cancellation condition: sum |a[k]b[k]| / |dot|', title=f'K = N = {n}')
    ax.grid(alpha=0.2)
axes[0].set_ylabel('Sequential FMA relative error to FP64 dot')
axes[0].legend(fontsize=9, frameon=False)
fig.suptitle('Cancellation and relative error | deliberately stratified sampling', fontsize=15)
fig.tight_layout(rect=(0, 0.065, 1, 0.93))
save(fig, 'cancellation.png')
