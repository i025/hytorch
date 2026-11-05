"""Plot CPU arithmetic error and strict failures from the conditioning CSV."""
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
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, method in zip(axes, ('cpu_model', 'numpy_fp32')):
        for family, marker, color in [('ordinary', 'o', '#4878b0'), ('cancellation', 'x', '#d46a38')]:
            selected = [r for r in rows if r['method'] == method and r['family'] == family]
            ax.scatter([float(r['condition_number']) for r in selected],
                       [float(r['absolute_error']) / float(r['tolerance']) for r in selected],
                       label=family, marker=marker, color=color, alpha=.7)
        ax.axhline(1, color='black', linestyle='--', label='existing tolerance')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('sum(abs(x)) / abs(FP64 sum(x))')
        ax.set_ylabel('Absolute error / (1e-5 + 1e-4 * abs(FP64 reference))')
        ax.set_title(method + '; points above 1 remain FAIL')
        ax.legend()
    fig.suptitle('CPU FP32 Sum conditioning: 4 lengths x 20 seeds x 2 input families\nGPU NOT_RUN; source: sum-conditioning-01/samples.csv')
    fig.tight_layout()
    fig.savefig(a.output / 'cpu-sum-conditioning.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
