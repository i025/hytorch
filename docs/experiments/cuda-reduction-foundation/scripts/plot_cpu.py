"""Render actual CPU reference CSV/NPZ; these figures contain NO GPU measurements."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    rows = list(csv.DictReader((a.input/'comparisons.csv').open()))
    fig, axes = plt.subplots(1,2,figsize=(11,4))
    for ax, ref in zip(axes,('numpy_fp32','fp64')):
        for kind,marker in (('integer','o'),('random','x'),('cancellation','^')):
            selected = [r for r in rows if r['reference']==ref and r['input']==kind and r['keepdims']=='False']
            ax.scatter([int(r['case']) for r in selected],
                [max(float(r['max_abs']),1e-12) for r in selected],label=kind,marker=marker)
        ax.set_yscale('log');ax.set_xlabel('Predeclared case index');ax.set_ylabel('Maximum absolute error')
        ax.set_title('CPU FP32 model vs '+ref);ax.legend(fontsize=8)
    fig.suptitle('CPU arithmetic-model visualization; GPU NOT_RUN (zero error drawn at 1e-12)')
    fig.tight_layout();fig.savefig(a.output/'cpu-errors.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    detail=[]
    for ax,case in zip(axes,(8,23)):
        data=np.load(a.input/f'case-{case}-cancellation-False.npz')
        errors=[float(abs(data[k].astype(np.float64)-data['fp64'])) for k in ('modeled','numpy_fp32')]
        ax.bar(['CPU FP32 model','NumPy FP32'],errors,color=['#4477aa','#cc8844'])
        ax.set_ylabel('Absolute error vs FP64 -> FP32');ax.set_title(f'Cancellation: {data["x"].size} inputs')
        for k in ('modeled','numpy_fp32','fp64'):
            detail.append(dict(case=case,elements=data['x'].size,method=k,value=float(data[k]),
                error_vs_fp64=float(abs(data[k].astype(np.float64)-data['fp64']))))
    fig.suptitle('Retained CPU negative cases; keepdims duplicates omitted; GPU NOT_RUN')
    fig.tight_layout();fig.savefig(a.output/'cpu-cancellation.png',dpi=160);plt.close(fig)
    with (a.output/'cancellation-values.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(detail[0]));w.writeheader();w.writerows(detail)


if __name__=='__main__':
    main()
