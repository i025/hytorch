"""Actual loss, per-call latency, and independent dInput allocation records."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--results',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)


def read(name):
    with (a.results/(name+'.csv')).open() as f:return list(csv.DictReader(f))


def save(fig,name):
    fig.text(.01,.01,'Source: actual GPU-6 measurement CSV; data visualization, not a screenshot.',fontsize=8)
    fig.tight_layout(rect=(0,.06,1,1));fig.savefig(a.output/name,dpi=170);plt.close(fig)


losses,errors=read('training-losses'),read('training-errors')
fig,axes=plt.subplots(1,2,figsize=(12,4))
for method in 'ABCDFE':
    selected=[r for r in losses if r['method']==method and r['mode']=='throughput' and r['repeat']=='1']
    axes[0].plot(range(1,len(selected)+1),[float(r['loss']) for r in selected],label=method,alpha=.75)
    delta=max(float(r['max_abs']) for r in errors if r['method']==method and r['reference']=='A')
    axes[1].bar(method,delta)
axes[0].set(title='Matched synthetic training, repeat 1',xlabel='Step',ylabel='MSE loss');axes[0].legend()
axes[1].set(title='All modes/repeats vs A',ylabel='Max absolute difference (weights or loss)')
save(fig,'loss-and-correctness.png')

rows=read('micro-latencies')
fig,axes=plt.subplots(2,2,figsize=(12,7))
for ax,layer in zip(axes.flat,['conv2','conv3','conv4','conv5']):
    for method in ['original','fused','torch']:
        selected=[r for r in rows if r['method']==method and r['layer']==layer and r['scope']=='dinput']
        ax.plot(range(1,len(selected)+1),[float(r['event_us']) for r in selected],label=method,lw=1)
    for boundary in [30.5,60.5]:ax.axvline(boundary,color='gray',linestyle=':',lw=.8)
    ax.set(title=layer,xlabel='Call (3 processes x 30 calls)',ylabel='CUDA Event span (us)');ax.legend()
fig.suptitle('All dInput calls retained; Event spans include submission gaps')
save(fig,'latency-variation.png')

rows=read('micro-memory')
fig,ax=plt.subplots(figsize=(9,4))
for i,method in enumerate(['original','fused']):
    values=[np.mean([float(r['temporary_peak_over_retained']) for r in rows if r['method']==method and r['layer']==layer and r['scope']=='dinput'])/2**20
            for layer in ['conv2','conv3','conv4','conv5']]
    bars=ax.bar(np.arange(4)+(i-.5)*.35,values,width=.35,label=method)
    ax.bar_label(bars,fmt='%.3f')
ax.set_xticks(range(4),['conv2','conv3','conv4','conv5'])
ax.set(title='Separate dInput call: CuPy pool requests',ylabel='Temporary peak over retained outputs (MiB)');ax.legend()
save(fig,'dinput-temporary-memory.png')
