"""CPU tables and figures for long-run failures, validation baselines and sampling limits."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--results',type=Path,required=True);p.add_argument('--raw',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
def read(path):
    with path.open() as f:return list(csv.DictReader(f))
def write(path,rows):
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def save(fig,name):
    fig.text(.5,.004,'SHARED_GPU • actual CSV visualization • not a screenshot',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.025,1,1));fig.savefig(a.output/(name+'.png'),dpi=160);plt.close(fig)
epochs=read(a.results/'epochs.csv');resources=read(a.results/'resources.csv');samples=[]
for row in resources:row['unix']=datetime.fromisoformat(row['timestamp_utc']).timestamp()
for row in epochs:
    if row['epoch']=='0':continue
    whole=[r for r in resources if r['job']==row['run']]
    inside=[r for r in whole if float(row['started_at_unix'])<=r['unix']<=float(row['finished_at_unix'])]
    samples.append(dict(run=row['run'],epoch=row['epoch'],job_samples=len(whole),measured_epoch_samples=len(inside),
        measured_gpu_util_max=max((int(r['utilization.gpu']) for r in inside),default=None),
        measured_own_memory_max_mib=max((int(r['own_memory_mib']) for r in inside),default=None),
        scope='Sampled; zero samples means NOT_SAMPLED, not zero activity'))
write(a.output/'per-run-resource-samples.csv',samples)
parts=read(a.results/'dweight-components.csv');phases=read(a.results/'trace-phases.csv');totals=[]
for m in 'FI':
    rr=[r for r in parts if r['method']==m];row=dict(method=m)
    for key in ('host_us','kernels','kernel_us','Im2col_count','Im2col_us','Copy_count','Copy_us','GEMM_count','tiles_count','tiles_us','reduce_count','reduce_us'):
        row['dweight_'+key]=sum(float(r[key]) for r in rr)/16
    for phase in ('data','forward','backward','update'):
        r=next(r for r in phases if r['method']==m and r['phase']==phase)
        row[phase+'_host_us']=float(r['host_nvtx_sum_ms'])*1000/16
        row[phase+'_gpu_us']=float(r['gpu_kernel_sum_ms'])*1000/16
        row[phase+'_kernels']=float(r['gpu_kernel_count'])/16
        row[phase+'_sync_api_count']=int(r['sync_api_count'])
    totals.append(row)
write(a.output/'trace-totals.csv',totals)
loss=read(a.results/'losses.csv');fig,axes=plt.subplots(1,3,figsize=(14,4.3))
for ax,(left,right) in zip(axes,[('F','I'),('I','E'),('A','E')]):
    x=np.array([float(r['loss']) for r in loss if r['run']=='train-'+left])
    y=np.array([float(r['loss']) for r in loss if r['run']=='train-'+right]);delta=abs(x-y)
    ax.plot(np.where(delta>0,delta,np.nan),lw=.7,label='Absolute Loss difference')
    ax.plot(1e-6+1e-5*abs(y),lw=.7,label='Unchanged allowed error')
    ax.set(title=left+' vs '+right+' / long-run FAIL',xlabel='Global training step',ylabel='Loss error',yscale='log');ax.legend(fontsize=8)
save(fig,'11-long-run-drift')
baselines=read(a.results/'constant-baselines.csv');fig,axes=plt.subplots(1,3,figsize=(13,4.3))
for ax,metric,key in zip(axes,('MSE','MAE','RMSE'),('validation_loss','steering_mae','steering_rmse')):
    rr=[next(r for r in epochs if r['run']=='train-'+m and r['epoch']=='3') for m in 'AFIE']
    ax.bar(list('AFIE'),[float(r[key]) for r in rr],color=['#888888','#e28131','#219269','#a756a2'])
    field={'MSE':'validation_mse','MAE':'validation_mae','RMSE':'validation_rmse'}[metric]
    ax.axhline(float(baselines[0][field]),color='black',linestyle='--',label='Zero predictor')
    initial=next(r for r in epochs if r['run']=='train-I' and r['epoch']=='0')
    ax.axhline(float(initial[key]),color='#2678a8',linestyle=':',label='Initial model')
    ax.set(title='Epoch 3 temporal validation '+metric,ylabel=metric);ax.legend(fontsize=8)
save(fig,'12-validation-baselines')
original=json.loads((a.raw/'before.json').read_text())['processes'];pids={r['pid'] for r in original}
observations=[json.loads(s) for s in (a.raw/'resources.jsonl').read_text().splitlines()]
activity=read(a.results/'process-activity.csv')
summary=dict(resource_samples=len(resources),max_gpu_memory_mib=max(int(r['memory.used']) for r in resources),
    max_gpu_utilization_pct=max(int(r['utilization.gpu']) for r in resources),max_temperature_c=max(int(r['temperature.gpu']) for r in resources),
    max_power_w=max(float(r['power.draw']) for r in resources),sm_clock_min_mhz=min(int(r['clocks.sm']) for r in resources),
    sm_clock_max_mhz=max(int(r['clocks.sm']) for r in resources),max_own_memory_mib=max(int(r['own_memory_mib']) for r in resources),
    original_memory_mib=sorted({int(p['memory_mib']) for r in observations for p in r['processes'] if p['pid'] in pids}),
    original_sm_available_samples=sum(r['baseline_original']=='True' and bool(r['sm_pct']) for r in activity),
    measured_epochs_without_resource_sample=sum(r['measured_epoch_samples']==0 for r in samples),
    original_latency='NOT_AVAILABLE; no zero-impact assertion',gpu_restarted_for_analysis=False)
(a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
