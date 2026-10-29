"""Seven figures from actual completed stage 4D CSV/Nsight exports only."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'docs/experiments/fused-sgd-cuda/scripts'))
from analyze_fused_results import csv_rows

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--results',type=Path,required=True)
p.add_argument('--analysis',type=Path,required=True)
p.add_argument('--figures',type=Path,required=True)
a=p.parse_args();a.figures.mkdir(parents=True,exist_ok=False)
assert json.loads((a.results/'completion.json').read_text())['status']=='PASSED'
micro=csv_rows(a.results/'micro-summary.csv')
training=csv_rows(a.results/'training-summary.csv')
raw=csv_rows(a.results/'training-raw.csv')
gains=csv_rows(a.results/'speedups.csv')
categories=csv_rows(a.results/'kernel-categories.csv')
methods='ABCDFE';colors=['#718096','#4477aa','#ee9933','#228866','#cc4455','#aa5599']


def save(fig,name):
    fig.text(.01,.01,'Source: actual stage 4D CSV / Nsight export; visualization, not a screenshot.',fontsize=8)
    fig.tight_layout(rect=(0,.06,1,1));fig.savefig(a.figures/name,dpi=170);plt.close(fig)


def stat(method,metric,mode='throughput'):
    return next(r for r in training if r['method']==method and r['metric']==metric and r['mode']==mode)


fig,axes=plt.subplots(1,2,figsize=(12,5))
for ax,scope in zip(axes,('dinput','full')):
    for i,(method,color) in enumerate(zip(('original','fused','torch'),('#4477aa','#cc4455','#aa5599'))):
        rr=[next(r for r in micro if r['method']==method and r['layer']==layer and r['scope']==scope and r['metric']=='event_us')
            for layer in ('conv2','conv3','conv4','conv5')]
        ax.bar(np.arange(4)+(i-1)*.25,[float(r['mean']) for r in rr],width=.25,
               yerr=[float(r['std']) for r in rr],capsize=3,color=color,label=method)
    ax.set_xticks(range(4),['conv2','conv3','conv4','conv5']);ax.set(title=scope,ylabel='CUDA Event span (us)');ax.legend()
fig.suptitle('Backward latency: 3 process means +/- sample SD')
save(fig,'backward-latency.png')

fig,ax=plt.subplots(figsize=(9,5))
groups=['Copy','GEMM','Col2im','fused_dInput']
for i,method in enumerate('DF'):
    vals=[sum(int(r['kernels']) for r in categories if r['method']==method and r['category']==g)/16 for g in groups]
    bars=ax.bar(np.arange(4)+(i-.5)*.35,vals,width=.35,label=method,color=colors[3+i]);ax.bar_label(bars,fmt='%.0f')
ax.set_xticks(range(4),groups);ax.set(ylabel='Backward kernels / measured step',title='D versus F: actual launches');ax.legend()
save(fig,'kernel-counts.png')

fig,ax=plt.subplots(figsize=(9,5))
for i,metric in enumerate(('event_us','sync_wall_us')):
    vals=[]
    for layer in ('conv2','conv3','conv4','conv5'):
        rr={r['method']:float(r['mean']) for r in micro if r['layer']==layer and r['scope']=='full' and r['metric']==metric}
        vals.append(rr['original']/rr['fused'])
    bars=ax.bar(np.arange(4)+(i-.5)*.35,vals,width=.35,label=metric);ax.bar_label(bars,fmt='%.2fx')
ax.axhline(1,color='black',linestyle='--');ax.set_xticks(range(4),['conv2','conv3','conv4','conv5'])
ax.set(ylabel='Original / fused elapsed time',title='Complete backward per-layer comparison');ax.legend()
save(fig,'per-layer-speedup.png')

fig,ax=plt.subplots(figsize=(10,5))
ax.bar(list(methods),[float(stat(m,'samples_per_sec')['mean']) for m in methods],
    yerr=[float(stat(m,'samples_per_sec')['std']) for m in methods],capsize=4,color=colors)
for m in methods:
    ax.scatter([m]*3,[float(r['samples_per_sec']) for r in raw if r['method']==m and r['mode']=='throughput'],color='black',s=18)
ax.set(title='DonkeyNet: 3 independent runs x 256 steps',ylabel='Samples/s (mean +/- SD)')
save(fig,'training-throughput.png')

fig,ax=plt.subplots(figsize=(10,5));base=float(stat('A','samples_per_sec')['mean'])
bars=ax.bar(list(methods),[float(stat(m,'samples_per_sec')['mean'])/base for m in methods],color=colors)
ax.bar_label(bars,fmt='%.3fx');ax.axhline(1,color='black',linestyle='--')
ax.set(title='Same-window cumulative throughput ratio',ylabel='Relative to A (PyTorch E is a control)')
save(fig,'cumulative-speedup.png')

fig,ax=plt.subplots(figsize=(10,5));rr=gains[:4]
bars=ax.bar([r['comparison'] for r in rr],[float(r['throughput_gain_pct']) for r in rr],color=colors[1:5])
ax.bar_label(bars,fmt='%.2f%%');ax.axhline(0,color='black')
ax.set(title='Same-window increments',ylabel='Throughput change (%)')
save(fig,'incremental-gains.png')

fig,axes=plt.subplots(2,1,figsize=(12,7))
for ax,method in zip(axes,'DF'):
    folder=a.analysis/('trace-training-'+method)
    ranges=[r for r in csv_rows(folder/'nvtx-events.csv') if r['name']=='backward']
    selected=sorted(ranges,key=lambda r:int(r['start']))[len(ranges)//2]
    rid=int(selected['sqlite_rowid']);start=int(selected['start'])
    kernels=[r for r in csv_rows(folder/'kernel-events.csv') if rid in json.loads(r['nvtx_ids'])]
    apis=[r for r in csv_rows(folder/'cuda-api-events.csv') if rid in json.loads(r['nvtx_ids']) and r['kind']=='launch']
    ax.broken_barh([(0,(int(selected['end'])-start)/1000)],(2.3,.45),facecolors='#718096')
    ax.broken_barh([((int(r['start'])-start)/1000,(int(r['end'])-int(r['start']))/1000) for r in apis],(1.3,.45),facecolors='#ee9933')
    for r in kernels:
        color='#cc4455' if 'conv_dinput_gather' in r['name'] else '#4477aa'
        ax.broken_barh([((int(r['start'])-start)/1000,(int(r['end'])-int(r['start']))/1000)],(.3,.45),facecolors=color)
    ax.set_yticks([.52,1.52,2.52],['GPU kernels','Host launch APIs','Host Backward'])
    ax.set(title=method+': middle measured backward, correlation-attributed',xlabel='Microseconds from host range start')
fig.suptitle('Own-operation timeline; white gaps do not establish whole-device idle')
save(fig,'gpu-timeline.png')
