"""Static scientific figures from recorded CSVs; these are not screenshots."""
import csv
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'figures';OUT.mkdir(exist_ok=False)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
COLORS=dict(A='#8c8c8c',D='#2677ae',F='#e38130',G='#2c9968',E='#ae579b')


def read(path):
    with path.open() as f:return list(csv.DictReader(f))


def save(fig,name):
    fig.text(.5,.005,'Measured CSV / Nsight visualization; not a terminal screenshot',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.025,1,1));fig.savefig(OUT/(name+'.png'),dpi=165);plt.close(fig)


folders=['calibration-01','held-out-01','validation-2-01','pointwise-validation-01']
shapes=[]
fig,ax=plt.subplots(figsize=(9,5))
for name in folders:
    rows=read(BASE/'results'/name/'comparisons.csv');shapes.extend(rows)
    ax.scatter([float(r['work']) for r in rows],[float(r['wall_speedup']) for r in rows],label=name,s=30,alpha=.8)
ax.axhline(1,color='black',lw=1);ax.axvline(1161216,color='grey',ls='--',label='unchanged work threshold')
ax.set(xscale='log',yscale='log',xlabel='N Cin H W Cout Kh Kw / stride² (work proxy)',ylabel='Original wall / Gather wall',title='100 geometries: work alone does not predict the best algorithm')
ax.legend(fontsize=8,ncol=2);save(fig,'01-cross-shape-wall')

auto=read(BASE/'results/actual-auto-01/comparisons.csv')
fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True)
x=np.arange(len(auto))
for i,method in enumerate(('original','fused','auto','torch')):
    axes[0].bar(x+(i-1.5)*.2,[float(r[method+'_wall_us']) for r in auto],width=.2,label=method,
        yerr=[float(r.get(method+'_std',0)) for r in auto] if method!='torch' else None,capsize=1)
axes[0].set(ylabel='Complete synchronized dInput wall (µs)',yscale='log',title='Fresh pointwise validation: actual per-call Auto overhead included')
axes[0].legend(ncol=4)
axes[1].bar(x,[float(r['actual_auto_speedup']) for r in auto],color=['#2c9968' if r['actual_auto_selected']=='fused' else '#999999' for r in auto])
axes[1].axhline(1,color='black');axes[1].axhline(1.05,color='grey',ls='--')
axes[1].set(ylabel='Original / Auto wall',xticks=x,xticklabels=[r['case'] for r in auto]);save(fig,'02-held-out-auto')

progress=read(BASE/'results/training-01/progress-table.csv');raw=read(BASE/'results/training-01/training-raw.csv')
fig,axes=plt.subplots(1,2,figsize=(12,4.5))
for ax,model in zip(axes,('donkey','spatial')):
    rr=[r for r in progress if r['model']==model]
    ax.bar([r['method'] for r in rr],[float(r['samples_per_sec']) for r in rr],
        yerr=[float(r['std']) for r in rr],capsize=4,color=[COLORS[r['method']] for r in rr])
    for i,r in enumerate(rr):
        yy=[float(v['samples_per_sec']) for v in raw if (v['model'],v['mode'],v['method'])==(model,'throughput',r['method'])]
        ax.scatter([i-.08,i,i+.08],yy,color='black',s=14,zorder=4)
    ax.set(title=model+' CNN: uninstrumented, 3 × 256 steps',ylabel='Samples/s (process mean ± sample SD)')
save(fig,'03-five-way-throughput')

fig,axes=plt.subplots(1,2,figsize=(11,4.3))
gains=read(BASE/'results/training-01/speedups.csv')
for ax,model in zip(axes,('donkey','spatial')):
    rr=[r for r in progress if r['model']==model and r['method']!='E']
    ax.bar([r['method'] for r in rr],[float(r['versus_A']) for r in rr],color=[COLORS[r['method']] for r in rr])
    ax.axhline(1,color='black');ax.set(title=model,ylabel='Throughput / current-code non-native baseline A',ylim=(.9,1.15))
save(fig,'04-cumulative-progress')
fig,axes=plt.subplots(1,2,figsize=(11,4.3))
for ax,model in zip(axes,('donkey','spatial')):
    rr=[r for r in gains if r['model']==model and r['comparison'] in ('F/D','G/D','G/F','G/A')]
    yy=[float(r['throughput_gain_pct']) for r in rr]
    ax.bar([r['comparison'] for r in rr],yy,color=['#2c9968' if v>0 else '#b44b46' for v in yy]);ax.axhline(0,color='black')
    ax.set(title=model,ylabel='Throughput change (%)')
save(fig,'05-incremental-gains')

summary=read(BASE/'results/training-01/training-summary.csv')
fig,axes=plt.subplots(1,2,figsize=(11,4.8))
for ax,model in zip(axes,('donkey','spatial')):
    bottom=np.zeros(5)
    for phase in ('forward','backward','update'):
        vals=[float(next(r['mean'] for r in summary if (r['model'],r['method'],r['mode'],r['metric'])==(model,m,'phases',phase+'_ms'))) for m in 'ADFGE']
        ax.bar(list('ADFGE'),vals,bottom=bottom,label=phase);bottom+=vals
    ax.set(title=model+' (separate synchronized phases experiment)',ylabel='ms/step; not throughput-run wall');ax.legend()
save(fig,'06-synchronized-phases')

fig,axes=plt.subplots(1,2,figsize=(11,4.4))
losses=read(BASE/'results/training-01/training-losses.csv')
for ax,model in zip(axes,('donkey','spatial')):
    for m in 'ADFGE':
        rr=[r for r in losses if (r['model'],r['method'],r['mode'],r['repeat'])==(model,m,'throughput','1')]
        ax.plot([float(r['loss']) for r in rr],label=m,color=COLORS[m],alpha=.8)
    ax.set(title=model+' / repeat 1',xlabel='Measured step',ylabel='MSE Loss',yscale='log');ax.legend(ncol=5)
save(fig,'07-loss-correctness')

trace=BASE/'results/profiles-01'
kk=read(trace/'shape-kernels.csv');cats=read(trace/'kernel-categories.csv')
fig,axes=plt.subplots(1,2,figsize=(11,4.5))
for ax,model in zip(axes,('donkey','spatial')):
    bottom=np.zeros(3)
    for cat in ('Copy','Col2im','GEMM','Gather','other'):
        values=[sum(int(r['kernels']) for r in cats if (r['model'],r['method'],r['category'])==(model,m,cat))/16 for m in 'DFG']
        ax.bar(list('DFG'),values,bottom=bottom,label=cat);bottom+=values
    ax.set(title=model+' / Nsight backward',ylabel='GPU kernels/step (16 measured steps)');ax.legend(fontsize=8)
save(fig,'08-backward-launches')

fig,axes=plt.subplots(1,2,figsize=(11,4.5))
ss=read(BASE/'results/actual-auto-01/summary.csv')
for ax,method in zip(axes,('original','fused')):
    for metric in ('event_us','wall_us'):
        vals=[float(next(r['mean'] for r in ss if (r['case'],r['method'],r['metric'])==(c['case'],method,metric))) for c in auto]
        ax.plot(x,vals,label=metric)
    vals=[np.mean([float(r['kernel_us']) for r in kk if (r['case'],r['method'])==(c['case'],method)]) for c in auto]
    ax.plot(x,vals,label='Nsight kernel sum (separate run)')
    ax.set(title=method+' / final 24 shapes',xlabel='Case index p00–p23',ylabel='µs',yscale='log');ax.legend(fontsize=8)
save(fig,'09-timing-scopes')

windows=read(trace/'backward-windows.csv')
RAW=Path(json.loads((trace/'raw-location.json').read_text())['analysis'])
fig,axes=plt.subplots(2,3,figsize=(15,6))
for i,model in enumerate(('donkey','spatial')):
    for j,m in enumerate('DFG'):
        ax=axes[i,j];rr=next(r for r in windows if (r['model'],r['method'])==(model,m))
        start,end=int(rr['start_ns']),int(rr['end_ns']);data=read(RAW/('trace-training-'+model+'-'+m)/'kernel-events.csv')
        extent=(end-start)/1000
        for r in data:
            lo,hi=int(r['start']),int(r['end'])
            if int(rr['nvtx_id']) in json.loads(r['nvtx_ids']):
                ax.broken_barh([((lo-start)/1000,(hi-lo)/1000)],(.5,.4),facecolors='#e38130' if 'gather' in r['name'] else '#2677ae')
                extent=max(extent,(hi-start)/1000)
        ax.broken_barh([(0,(end-start)/1000)],(0,.3),facecolors='#bbbbbb')
        ax.set(title=model+' / '+m,xlabel='µs from host backward range start',yticks=[.15,.7],yticklabels=['host','GPU'])
        ax.set_xlim(0,extent)
save(fig,'10-backward-timeline')
print('Wrote',len(list(OUT.glob('*.png'))),'measured-data figures')
