"""Reproduce the two available figures; no stage 4D GPU data are implied."""
import argparse
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--experiment',type=Path,required=True)
p.add_argument('--figures',type=Path,required=True)
a=p.parse_args();a.figures.mkdir(parents=True,exist_ok=False)
shutil.copyfile(a.experiment/'results/history/historical-design-evidence.png',a.figures/'historical-design-evidence.png')
rows=json.loads((a.experiment/'results/raw/conv_backward_fusion_cpu_equation_01/summary.json').read_text())['per_case']
fig,axes=plt.subplots(1,2,figsize=(11,4))
for name,color in [('original_numpy','#4477aa'),('pytorch_cpu','#cc4455')]:
    r=[v for v in rows if v['reference']==name]
    axes[0].plot([v['case'] for v in r],[v['max_abs'] for v in r],marker='o',label=name,color=color)
    axes[1].plot([v['case'] for v in r],[v['relative_l2'] for v in r],marker='o',label=name,color=color)
axes[0].set(ylabel='Max absolute error',xlabel='Case');axes[1].set(ylabel='Relative L2 error (sample)',xlabel='Case')
for ax in axes:ax.legend()
fig.suptitle('CPU gather equation samples; NOT CUDA execution')
fig.text(.01,.01,'Source: CPU fmaf sample CSV; not a CUDA result or screenshot.',fontsize=9)
fig.tight_layout(rect=(0,.07,1,1));fig.savefig(a.figures/'cpu-equation-errors.png',dpi=170);plt.close(fig)
