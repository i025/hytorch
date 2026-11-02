"""Actual stage 4G CSV/Nsight visualizations, not screenshots."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
colors=dict(A='#909090',F='#e38130',I='#2c9968',E='#ae579b')
def read(path):
    with path.open() as f:return list(csv.DictReader(f))
def save(fig,name):
    fig.text(.5,.004,'Measured CSV / CUPTI visualization; not a screenshot',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.025,1,1));fig.savefig(a.output/(name+'.png'),dpi=165);plt.close(fig)

micro=read(a.results/'micro/comparisons.csv');x=np.arange(len(micro))
fig,axes=plt.subplots(1,2,figsize=(12,4.6))
for ax,metric,label in zip(axes,('kernel_us','wall_us'),('CUPTI kernel sum','Complete synchronized wall')):
    for method,color in [('original','#2677ae'),('fused','#2c9968'),('torch','#ae579b')]:
        ax.plot(x,[float(r[method+'_'+metric]) for r in micro],'.',label=method,color=color,ms=3)
    ax.set(yscale='log',xlabel='All 132 predeclared shapes',ylabel='µs/call',title=label);ax.legend()
save(fig,'01-dweight-latencies')
fig,ax=plt.subplots(figsize=(9,4.5))
for key,label in [('gpu_original_over_fused','CUPTI GPU'),('wall_original_over_fused','Complete wall')]:
    values=np.sort([float(r[key]) for r in micro]);ax.plot(values,np.arange(1,len(values)+1)/len(values),label=label)
ax.axvline(1,color='black');ax.set(xscale='log',xlabel='Original / fused time',ylabel='Fraction of all 132 shapes',title='All wins and regressions retained');ax.legend()
save(fig,'02-cross-shape-speedups')
parts=read(a.results/'dweight-components.csv');phases=read(a.results/'trace-phases.csv')
fig,axes=plt.subplots(1,2,figsize=(11,4.7))
for ax,model in zip(axes,('donkey','spatial')):
    for shift,method in [(-.2,'F'),(.2,'I')]:
        rr=[r for r in parts if (r['model'],r['method'])==(model,method)]
        values=[sum(float(r[k]) for r in rr)/16 for k in ('kernels','Im2col_count','Copy_count')]
        ax.bar(np.arange(3)+shift,values,.4,label=method,color=colors[method])
    ax.set(xticks=np.arange(3),xticklabels=['All kernels','Im2col','Copy'],ylabel='dWeight kernels / measured step',title=model+' / separate Nsight');ax.legend()
save(fig,'03-kernel-copy-launch-counts')
progress=read(a.results/'training/progress-table.csv');raw=read(a.results/'training/training-raw.csv')
fig,axes=plt.subplots(1,2,figsize=(11,4.6))
for ax,model in zip(axes,('donkey','spatial')):
    rr=[r for r in progress if r['model']==model]
    ax.bar([r['method'] for r in rr],[float(r['samples_per_sec']) for r in rr],
        yerr=[float(r['std']) for r in rr],capsize=3,color=[colors[r['method']] for r in rr])
    for i,r in enumerate(rr):
        values=[float(v['samples_per_sec']) for v in raw if (v['model'],v['method'],v['mode'])==(model,r['method'],'throughput')]
        ax.scatter([i-.08,i,i+.08],values,color='black',s=13)
    ax.set(title=model+' / 3 × 256 measured steps',ylabel='Samples/s; process mean ± sample SD')
save(fig,'04-matched-throughput')
gains=read(a.results/'training/speedups.csv')
fig,ax=plt.subplots(figsize=(8,4.5))
xx=np.arange(3)
for shift,model in [(-.2,'donkey'),(.2,'spatial')]:
    values=[float(next(r['throughput_gain_pct'] for r in gains if (r['model'],r['comparison'])==(model,pair))) for pair in ('I/F','I/A','I/E')]
    ax.bar(xx+shift,values,.4,label=model)
ax.axhline(0,color='black');ax.set(xticks=xx,xticklabels=['I/F incremental','I/A cumulative','I/E gap'],ylabel='Throughput change (%)',title='Ratios from this window only');ax.legend()
save(fig,'05-incremental-and-cumulative')
fig,axes=plt.subplots(1,2,figsize=(11,4.5))
for ax,model in zip(axes,('donkey','spatial')):
    rr=[r for r in progress if r['model']==model]
    ax.bar([r['method'] for r in rr],[100*float(r['versus_E']) for r in rr],color=[colors[r['method']] for r in rr]);ax.axhline(100,color='black')
    ax.set(title=model,ylabel='Percent of PyTorch throughput')
save(fig,'06-pytorch-gap')
windows=read(a.results/'backward-windows.csv');loc=json.loads((a.results/'raw-location.json').read_text())
fig,axes=plt.subplots(2,2,figsize=(12,6))
for i,model in enumerate(('donkey','spatial')):
    extent=0
    for j,method in enumerate('FI'):
        ax=axes[i,j];r=next(v for v in windows if (v['model'],v['method'])==(model,method));start=int(r['start_ns']);end=int(r['end_ns'])
        events=read(Path(loc['analysis'])/('trace-training-'+model+'-'+method)/'kernel-events.csv')
        for k in events:
            if int(r['nvtx_id']) not in json.loads(k['nvtx_ids']):continue
            lo,hi=int(k['start']),int(k['end']);extent=max(extent,(hi-start)/1000)
            color=colors['I'] if 'conv_dweight_' in k['name'] else '#2677ae'
            ax.broken_barh([((lo-start)/1000,(hi-lo)/1000)],(.5,.4),facecolors=color)
        ax.broken_barh([(0,(end-start)/1000)],(0,.3),facecolors='#bbb');extent=max(extent,(end-start)/1000)
        ax.set(title=model+' / '+method+' / first measured backward',xlabel='µs from host range start',yticks=[.15,.7],yticklabels=['host','GPU'])
    for ax in axes[i]:ax.set_xlim(0,extent)
save(fig,'07-backward-timeline')
stats=read(a.results/'training/training-summary.csv')
fig,axes=plt.subplots(1,2,figsize=(11,4.5))
for ax,model in zip(axes,('donkey','spatial')):
    bottom=np.zeros(4)
    for phase in ('forward','backward','update'):
        values=[float(next(r['mean'] for r in stats if (r['model'],r['method'],r['mode'],r['metric'])==(model,m,'phases',phase+'_ms'))) for m in 'AFIE']
        ax.bar(list('AFIE'),values,bottom=bottom,label=phase);bottom+=values
    ax.set(title=model+' / separately synchronized phases',ylabel='ms/step; not throughput wall');ax.legend()
save(fig,'08-synchronized-phases')
losses=read(a.results/'training/training-losses.csv')
fig,axes=plt.subplots(1,2,figsize=(11,4.4))
for ax,model in zip(axes,('donkey','spatial')):
    for method in 'AFIE':
        rr=[r for r in losses if (r['model'],r['method'],r['mode'],r['repeat'])==(model,method,'throughput','1')]
        ax.plot([float(r['loss']) for r in rr],label=method,color=colors[method])
    ax.set(title=model+' / repeat 1',xlabel='Measured step',ylabel='MSE Loss',yscale='log');ax.legend()
save(fig,'09-losses')
mem=read(a.results/'micro/memory.csv');fig,ax=plt.subplots(figsize=(10,4.6))
rr=[r for r in mem if r['method']=='fused' and r['repeat']=='1']
ax.plot([int(r['im2col_bytes'])/2**20 for r in rr],'.',label='Old col logical bytes')
ax.plot([int(r['native_partial_bytes'])/2**20 for r in rr],'.',label='New split partial logical bytes')
ax.set(title='All 132 geometries; workspace can regress',xlabel='Predeclared shape index',ylabel='MiB (logical workspace, not process peak)');ax.legend()
save(fig,'10-workspaces')
print('Wrote 10 measured-data figures')
