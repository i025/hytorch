"""Measured CSV/Nsight and compiler-resource figures, never screenshots."""
import csv
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--profiles',type=Path,default=BASE/'results/profiles-01')
p.add_argument('--output',type=Path,default=BASE/'figures')
a=p.parse_args();OUT=a.output;OUT.mkdir(parents=True,exist_ok=False)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
COLORS=dict(A='#909090',D='#2677ae',F='#e38130',H='#2c9968',E='#ae579b')


def read(path):
    with path.open() as f:return list(csv.DictReader(f))


def save(fig,name):
    fig.text(.5,.004,'Measured CSV / Nsight or compiler-resource visualization; not a screenshot',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.025,1,1));fig.savefig(OUT/(name+'.png'),dpi=165);plt.close(fig)


pure=read(a.profiles/'pure-kernel-comparisons.csv')
profile_status=json.loads((a.profiles/'analysis.json').read_text())
micro=read(BASE/'results/micro-01/comparisons.csv')
fig,ax=plt.subplots(figsize=(7,5))
for split,color in [('history','#2677ae'),('fresh','#2c9968')]:
    rr=[r for r in pure if r['split']==split]
    ax.errorbar([float(r['old_kernel_us']) for r in rr],[float(r['opt_kernel_us']) for r in rr],
        xerr=[float(r['old_std']) for r in rr],yerr=[float(r['opt_std']) for r in rr],fmt='o',ms=3,label=split,color=color,alpha=.7)
limits=[min(min(float(r[k]) for r in pure) for k in ('old_kernel_us','opt_kernel_us'))*.8,
        max(max(float(r[k]) for r in pure) for k in ('old_kernel_us','opt_kernel_us'))*1.2]
ax.plot(limits,limits,color='black',lw=1);ax.set(xscale='log',yscale='log',xlabel='Old Gather CUPTI kernel (µs)',
    ylabel='Four-chain Gather CUPTI kernel (µs)',title='124 shapes; '+str(profile_status['kernel_repeats'])+' profiles × 30 calls; '+profile_status['status']);ax.legend()
save(fig,'01-pure-kernel-latency')

fig,axes=plt.subplots(1,2,figsize=(11,4.5))
for ax,split in zip(axes,('history','fresh')):
    for rows,key,label in [(pure,'gpu_old_over_opt','CUPTI pure kernel'),(micro,'wall_old_over_opt','Complete dInput wall')]:
        values=np.sort([float(r[key]) for r in rows if r['split']==split]);ax.plot(values,np.arange(1,len(values)+1)/len(values),label=label)
    ax.axvline(1,color='black',lw=1);ax.set(xscale='log',xlabel='Old / optimized time',ylabel='Empirical fraction of shapes',title=split);ax.legend()
save(fig,'02-cross-shape-distribution')

rr=[r for r in micro if r['split']=='fresh'];x=np.arange(len(rr))
fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True)
for i,method in enumerate(('original','fused','fused_opt','torch')):
    axes[0].bar(x+(i-1.5)*.2,[float(r[method+'_wall_us']) for r in rr],width=.2,label=method)
axes[0].set(ylabel='Full synchronized wall (µs)',yscale='log',title='All 24 independent geometries; prepared arrays, allocation + submit + wait');axes[0].legend(ncol=4)
axes[1].bar(x,[float(r['wall_old_over_opt']) for r in rr],color=['#2c9968' if float(r['wall_old_over_opt'])>1 else '#b44b46' for r in rr])
axes[1].axhline(1,color='black');axes[1].set(ylabel='Old / optimized wall',xticks=x,xticklabels=[r['case'] for r in rr])
save(fig,'03-complete-dinput-wall')

for folder,name in [('training-02','04-matched-throughput'),('training-01','04b-first-noisy-window')]:
    progress=read(BASE/'results'/folder/'progress-table.csv');raw=read(BASE/'results'/folder/'training-raw.csv')
    fig,axes=plt.subplots(1,2,figsize=(12,4.7))
    for ax,model in zip(axes,('donkey','spatial')):
        rr=[r for r in progress if r['model']==model]
        ax.bar([r['method'] for r in rr],[float(r['samples_per_sec']) for r in rr],yerr=[float(r['std']) for r in rr],
            capsize=3,color=[COLORS[r['method']] for r in rr])
        for i,r in enumerate(rr):
            yy=[float(v['samples_per_sec']) for v in raw if (v['model'],v['mode'],v['method'])==(model,'throughput',r['method'])]
            ax.scatter([i-.08,i,i+.08],yy,color='black',s=15,zorder=4)
        ax.set(title=model+' / '+folder+' / 3 × 256 steps',ylabel='Samples/s (process mean ± sample SD)')
    save(fig,name)

progress=read(BASE/'results/training-02/progress-table.csv');gains=read(BASE/'results/training-02/speedups.csv')
fig,axes=plt.subplots(1,2,figsize=(11,4.5))
for ax,model in zip(axes,('donkey','spatial')):
    rr=[r for r in progress if r['model']==model and r['method']!='E']
    ax.bar([r['method'] for r in rr],[float(r['versus_A']) for r in rr],color=[COLORS[r['method']] for r in rr]);ax.axhline(1,color='black')
    ax.set(title=model+' / second matched window',ylabel='Throughput / non-native current-code A')
save(fig,'05-cumulative-progress')
fig,axes=plt.subplots(1,2,figsize=(11,4.3))
for ax,model in zip(axes,('donkey','spatial')):
    rr=[r for r in progress if r['model']==model]
    ax.bar([r['method'] for r in rr],[100*float(r['versus_E']) for r in rr],color=[COLORS[r['method']] for r in rr]);ax.axhline(100,color='black')
    ax.set(title=model,ylabel='Percent of PyTorch throughput')
save(fig,'06-pytorch-gap')

fig,ax=plt.subplots(figsize=(7,4))
resources=read(BASE/'results/kernel-resources.csv')
ax.bar([r['kernel'] for r in resources],[int(r['registers']) for r in resources],color=['#e38130','#2c9968'])
ax.set(ylabel='Registers/thread (SM80 compiler report)',title='256 threads/block; 0 shared / local / stack / spill bytes')
ax.text(.5,.45,'Occupancy not measured; register count is not achieved occupancy',transform=ax.transAxes,ha='center',fontsize=9)
save(fig,'07-compiler-resources')

profiles=a.profiles
windows=read(profiles/'backward-windows.csv');loc=json.loads((profiles/'raw-location.json').read_text())
fig,axes=plt.subplots(2,2,figsize=(12,6))
for i,model in enumerate(('donkey','spatial')):
    extent=0
    for j,m in enumerate('FH'):
        ax=axes[i,j];r=next(r for r in windows if (r['model'],r['method'])==(model,m));start,end=int(r['start_ns']),int(r['end_ns'])
        data=read(Path(loc['analysis'])/('trace-training-'+model+'-'+m)/'kernel-events.csv')
        for k in data:
            if int(r['nvtx_id']) not in json.loads(k['nvtx_ids']):continue
            lo,hi=int(k['start']),int(k['end']);extent=max(extent,(hi-start)/1000)
            color='#2c9968' if 'gather_opt' in k['name'] else '#e38130' if 'gather' in k['name'] else '#2677ae'
            ax.broken_barh([((lo-start)/1000,(hi-lo)/1000)],(.5,.4),facecolors=color)
        ax.broken_barh([(0,(end-start)/1000)],(0,.3),facecolors='#bbbbbb');extent=max(extent,(end-start)/1000)
        ax.set(title=model+' / '+m+' / first measured backward',xlabel='µs from host range start',yticks=[.15,.7],yticklabels=['host','GPU'])
    for ax in axes[i]:ax.set_xlim(0,extent)
save(fig,'08-backward-timeline')

summary=read(BASE/'results/training-02/training-summary.csv')
fig,axes=plt.subplots(1,2,figsize=(11,4.7))
for ax,model in zip(axes,('donkey','spatial')):
    bottom=np.zeros(5)
    for phase in ('forward','backward','update'):
        values=[float(next(r['mean'] for r in summary if (r['model'],r['method'],r['mode'],r['metric'])==(model,m,'phases',phase+'_ms'))) for m in 'ADFHE']
        ax.bar(list('ADFHE'),values,bottom=bottom,label=phase);bottom+=values
    ax.set(title=model+' / separately synchronized phases',ylabel='ms/step; separate from throughput wall');ax.legend()
save(fig,'09-synchronized-phases')

losses=read(BASE/'results/training-02/training-losses.csv')
fig,axes=plt.subplots(1,2,figsize=(11,4.5))
for ax,model in zip(axes,('donkey','spatial')):
    for m in 'ADFHE':
        rr=[r for r in losses if (r['model'],r['method'],r['mode'],r['repeat'])==(model,m,'throughput','1')]
        ax.plot([float(r['loss']) for r in rr],label=m,color=COLORS[m],alpha=.8)
    ax.set(title=model+' / repeat 1',xlabel='Measured step',ylabel='MSE Loss',yscale='log');ax.legend(ncol=5)
save(fig,'10-loss-correctness')

parts=read(profiles/'gradient-components.csv')
fig,axes=plt.subplots(1,2,figsize=(11,4.7))
for ax,model,layers in zip(axes,('donkey','spatial'),(['conv2','conv3','conv4','conv5'],['conv2','conv3'])):
    x=np.arange(len(layers))
    for shift,m in [(-.2,'F'),(.2,'H')]:
        vals=[np.mean([float(r['kernel_us']) for r in parts if (r['model'],r['method'],r['part'],r['layer'])==(model,m,'dInput',layer)]) for layer in layers]
        ax.bar(x+shift,vals,width=.4,label=m,color=COLORS[m])
    ax.set(title=model+' / actual dInput ancestors in Nsight',ylabel='CUPTI kernel µs/call',xticks=x,xticklabels=layers);ax.legend()
save(fig,'11-training-dinput-kernels')
print('Wrote',len(list(OUT.glob('*.png'))),'measured-data figures')
