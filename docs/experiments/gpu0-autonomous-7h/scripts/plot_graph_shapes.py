"""Visualize measured CPU graph timing CSV; these are not GPU measurements."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    rows = list(csv.DictReader(a.input.open()))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, elements in zip(axes, (128, 8192)):
        selected = [r for r in rows if int(r['elements']) == elements]
        positions = list(range(len(selected)))
        for mode, offset, color in [('list', -.19, '#4878b0'), ('heap', .19, '#e68a3f')]:
            ax.bar([v + offset for v in positions], [float(r[mode + '_mean_ms']) for r in selected],
                   yerr=[float(r[mode + '_sd_ms']) for r in selected], width=.38,
                   label=mode, color=color, capsize=3)
        ax.set_xticks(positions)
        ax.set_xticklabels([r['kind'] + '\n' + r['width'] + '\nready=' + r['max_ready_functions'] for r in selected])
        ax.set_title(f'{elements} FP32 elements per node')
        ax.set_ylabel('CPU forward + backward (ms), mean +/- sample SD')
        ax.legend()
    fig.suptitle('CPU synthetic graph topology study: 5 alternating repeats x 30 iterations\nGPU NOT_RUN; source: graph-shapes-01/summary.csv')
    fig.tight_layout()
    fig.savefig(a.output / 'cpu-graph-topologies.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
