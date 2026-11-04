"""Scientific figures from actual shared-GPU records; never screenshots."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--results',type=Path,required=True);p.add_argument('--prepared',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
def read(path):
    with path.open() as f:return list(csv.DictReader(f))
def save(fig,name):
    fig.text(.5,.004,'SHARED_GPU • measured CSV/CUPTI visualization • not a screenshot',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.025,1,1));fig.savefig(a.output/(name+'.png'),dpi=160);plt.close(fig)
colors=dict(A='#888888',F='#e28131',I='#219269',E='#a756a2')
stats=read(a.results/'timing-summary.csv');epochs=read(a.results/'epochs.csv')
def stat(task,method,metric):return next(r for r in stats if (r['task'],r['method'],r['metric'])==(task,method,metric))
fig,axes=plt.subplots(1,2,figsize=(11,4.7))
for ax,task,title in zip(axes,('compute','io'),('Resident GPU inputs','JPEG read + preprocess + training')):
    rr=[stat(task,m,'samples_per_sec') for m in 'AFIE']
    ax.bar(list('AFIE'),[float(r['mean']) for r in rr],yerr=[float(r['std']) for r in rr],capsize=3,color=list(colors.values()))
    ax.set(title=title,ylabel='Samples/s (3 processes × 256 steps)')
save(fig,'01-throughput')
training=[r for r in epochs if r['task']=='train']
fig,axes=plt.subplots(1,2,figsize=(11,4.5))
for ax,key,title in zip(axes,('train_loss','validation_loss'),('Training loss','Temporal validation loss')):
    for m in 'AFIE':
        rr=[r for r in training if r['method']==m and r.get(key)]
        ax.plot([int(r['epoch']) for r in rr],[float(r[key]) for r in rr],'.-',label=m,color=colors[m])
    ax.set(title=title,xlabel='Epoch',ylabel='MSE',yscale='log');ax.legend()
save(fig,'02-train-validation-loss')
fig,axes=plt.subplots(1,2,figsize=(11,4.5))
for ax,key in zip(axes,('steering_mae','steering_rmse')):
    for m in 'AFIE':
        rr=[r for r in training if r['method']==m]
        ax.plot([int(r['epoch']) for r in rr],[float(r[key]) for r in rr],'.-',label=m,color=colors[m])
    ax.set(title=key,xlabel='Epoch',ylabel='Normalized steering units');ax.legend()
save(fig,'03-steering-errors')
fig,ax=plt.subplots(figsize=(8,4.5));bottom=np.zeros(4)
for phase in ('data','transfer','forward','backward','update'):
    values=[float(stat('phases',m,phase+'_ms')['mean']) for m in 'AFIE'];ax.bar(list('AFIE'),values,bottom=bottom,label=phase);bottom+=values
ax.set(title='Separate synchronized phase experiment',ylabel='ms/step');ax.legend()
save(fig,'04-phases')
fig,axes=plt.subplots(1,2,figsize=(11,4.4))
for ax,task in zip(axes,('compute','io')):
    values=[float(stat(task,m,'samples_per_sec')['mean'])/float(stat(task,'A','samples_per_sec')['mean']) for m in 'AFIE']
    ax.bar(list('AFIE'),values,color=list(colors.values()));ax.axhline(1,color='black');ax.set(title=task+' / current-code no Native baseline',ylabel='Throughput / A')
save(fig,'05-cumulative-gains')
fig,ax=plt.subplots(figsize=(8,4.4));x=np.arange(4)
for shift,task in [(-.2,'compute'),(.2,'io')]:
    values=[100*float(stat(task,m,'samples_per_sec')['mean'])/float(stat(task,'E','samples_per_sec')['mean']) for m in 'AFIE']
    ax.bar(x+shift,values,.4,label=task)
ax.set(xticks=x,xticklabels=list('AFIE'),ylabel='Percent of PyTorch throughput');ax.axhline(100,color='black');ax.legend()
save(fig,'06-pytorch-gap')
resources=read(a.results/'resources.csv');x=np.arange(len(resources))
fig,axes=plt.subplots(4,1,figsize=(12,10),sharex=True)
axes[0].plot(x,[float(r['utilization.gpu']) for r in resources],label='Whole GPU1 utilization');axes[0].set_ylabel('%');axes[0].legend()
axes[1].plot(x,[float(r['memory.used']) for r in resources],label='Whole card')
axes[1].plot(x,[float(r['own_memory_mib']) for r in resources],label='Own sampled processes');axes[1].set_ylabel('MiB');axes[1].legend()
axes[2].plot(x,[float(r['power.draw']) for r in resources],label='Power W');axes[2].plot(x,[float(r['temperature.gpu']) for r in resources],label='Temperature °C');axes[2].legend()
axes[3].plot(x,[float(r['clocks.sm']) for r in resources],label='SM MHz');axes[3].plot(x,[float(r['clocks.mem']) for r in resources],label='Memory MHz');axes[3].legend();axes[3].set_xlabel('Resource sample (raw UTC timestamps in CSV)')
axes[0].set_title('GPU1 shared resources; no original-task latency measurement')
save(fig,'07-shared-resources')
loc=json.loads((a.results/'summary.json').read_text());analysis=Path(loc['analysis'])
fig,axes=plt.subplots(1,2,figsize=(12,4.8))
for ax,method in zip(axes,'FI'):
    folder=analysis/('trace-'+method);rr=read(folder/'nvtx-events.csv');kk=read(folder/'kernel-events.csv')
    phases=[next(r for r in rr if r['name']==name) for name in ('data','transfer','forward','backward','update')]
    start=int(phases[0]['start']);ids={int(r['sqlite_rowid']) for r in phases}
    for r in phases:
        ax.broken_barh([((int(r['start'])-start)/1e6,(int(r['end'])-int(r['start']))/1e6)],(0,.3),facecolors='#999999')
    for k in kk:
        if not ids.intersection(json.loads(k['nvtx_ids'])):continue
        color='#219269' if 'conv_dweight_' in k['name'] else '#2678a8'
        ax.broken_barh([((int(k['start'])-start)/1e6,(int(k['end'])-int(k['start']))/1e6)],(.5,.4),facecolors=color)
    ax.set(title=method+' / first measured real-image step',xlabel='ms from data start',yticks=[.15,.7],yticklabels=['Host phases','Own GPU kernels'])
save(fig,'08-nsight-timeline')
manifest=read(a.prepared/'manifest.csv');fig,ax=plt.subplots(figsize=(8,4.4))
for split in ('train','validation'):
    values=[float(r['steering']) for r in manifest if r['split']==split]
    ax.hist(values,bins=np.linspace(-1,1,41),density=True,alpha=.5,label=split)
ax.set(xlabel='Steering',ylabel='Density',title='Center-camera label imbalance');ax.legend()
save(fig,'09-steering-distribution')
fig,axes=plt.subplots(1,2,figsize=(11,4.5))
for ax,key in zip(axes,('train_wall_s','validation_wall_s')):
    for method in 'AFIE':
        rr=[r for r in training if r['method']==method and int(r['epoch'])>0]
        ax.plot([int(r['epoch']) for r in rr],[float(r[key]) for r in rr],'.-',label=method,color=colors[method])
    ax.set(title=key+' / full-data protocol',xlabel='Epoch',ylabel='Seconds');ax.legend()
save(fig,'10-epoch-times')
print('Wrote 10 real-data figures from actual shared-GPU records')
