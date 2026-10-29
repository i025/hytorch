"""Render real full-matrix accuracy, timing, and resource evidence as PNGs."""
import csv
import json
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'figures'
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
colors={'shared32':'#c94d43','block32_tree':'#168a70','cupy':'#437db1','torch':'#8a61aa','numpy_fp32':'#888888'}
labels={'shared32':'Original Shared32','block32_tree':'Block32 + Tree','cupy':'CuPy','torch':'PyTorch','numpy_fp32':'NumPy FP32'}
sizes=(1024,2048,4096);bases=(42,1729,20261029)


def read(name):return list(csv.DictReader((ROOT/'results'/name).open()))


accuracy=read('accuracy-01/full-matrix-summary.csv')
latency=read('benchmark-01/benchmark-summary.csv')
changes=read('accuracy-01/improvement-summary.csv')
trade=read('accuracy-performance.csv')
resource=read('correctness-01/kernel-resources.csv')
hist=read('accuracy-01/error-distributions.csv')


def save(fig,name,source):
    fig.text(0.02,0.012,'Source: '+source+' | Measured data visualization, not a screenshot',fontsize=8,color='#555555')
    with (OUT/name).open('xb') as stream:fig.savefig(stream,format='png',facecolor='white')
    plt.close(fig)
    print(OUT/name)


fig,axes=plt.subplots(1,2,figsize=(13,5.7))
for ax,key,title in zip(axes,('mean_abs_error','max_abs_error'),('Full-matrix mean absolute error','Full-matrix maximum absolute error')):
    for method in ('shared32','block32_tree','cupy','numpy_fp32'):
        groups=[[float(r[key]) for r in accuracy if int(r['n'])==n and r['method']==method and r['scope']=='ALL'] for n in sizes]
        means=np.mean(groups,axis=1);low=np.min(groups,axis=1);high=np.max(groups,axis=1)
        name='CuPy / PyTorch' if method=='cupy' else labels[method]
        ax.errorbar(sizes,means,yerr=[means-low,high-means],marker='o',capsize=4,color=colors[method],
                    linestyle='--' if method=='cupy' else '-',label=name)
    ax.set(title=title,xlabel='N = K',ylabel='Error to GPU FP64 reference',yscale='log',xticks=sizes)
    ax.grid(alpha=0.2)
axes[0].legend(fontsize=9,frameon=False)
fig.suptitle('Accuracy Comparison | all matrix elements, three independent seeds',fontsize=15)
fig.text(0.5,0.065,'Points: mean of three matrix statistics; whiskers: min–max across seeds; CuPy and PyTorch outputs equal in every case',ha='center',fontsize=8)
fig.tight_layout(rect=(0,0.1,1,0.93))
save(fig,'accuracy-comparison.png','accuracy-01/full-matrix-summary.csv')

fig,axes=plt.subplots(2,3,figsize=(14,7.5))
for col,n in enumerate(sizes):
    for row,ref in enumerate(('numpy','fp64')):
        ax=axes[row,col]
        for offset,key,method in [(-0.18,'old_fail_','shared32'),(0.18,'new_fail_','block32_tree')]:
            values=[int(next(r for r in changes if int(r['n'])==n and int(r['seed_base'])==base)[key+ref]) for base in bases]
            bars=ax.bar(np.arange(3)+offset,values,width=0.34,color=colors[method],label=labels[method])
            for bar,value in zip(bars,values):ax.annotate(str(value),(bar.get_x()+bar.get_width()/2,value),xytext=(0,4),textcoords='offset points',ha='center',fontsize=9)
        ax.set(yscale='symlog',ylim=(0,6000),xticks=np.arange(3),xticklabels=['42','1729','20261029'],title=f'N={n} | reference: {"NumPy FP32" if ref=="numpy" else "FP64"}')
        ax.set_xlabel('Seed base (actual RNG seed = base + N)')
        ax.grid(axis='y',alpha=0.2)
        if col==0:ax.set_ylabel('Failing output elements')
axes[0,0].legend(fontsize=9,frameon=False)
fig.suptitle('Tolerance Failures | unchanged atol = rtol = 1e−4',fontsize=15)
fig.text(0.5,0.055,'Symmetric-log count axis includes zero. At N=4096, base 42: all 2,418 original failures resolved; one new NumPy-reference failure.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,0.08,1,0.94))
save(fig,'tolerance-failures.png','accuracy-01/improvement-summary.csv')

fig,axes=plt.subplots(1,3,figsize=(14,5.5))
methods=('shared32','block32_tree','cupy','torch')
for ax,n in zip(axes,sizes):
    for i,method in enumerate(methods):
        values=np.array([float(r['mean_ms']) for r in latency if int(r['n'])==n and r['method']==method])
        mean=np.mean(values)
        ax.bar(i,mean,color=colors[method],alpha=0.8)
        ax.errorbar(i,mean,yerr=[[mean-values.min()],[values.max()-mean]],color='black',capsize=4)
        ax.scatter(np.full(3,i)+[-0.07,0,0.07],values,s=16,color='black',zorder=3)
        ax.annotate(f'{mean:.3g}',(i,values.max()),xytext=(0,6),textcoords='offset points',ha='center')
    ax.set(title=f'N = {n}',ylabel='CUDA-event latency (ms)',xticks=range(4),xticklabels=['Shared32','Block32\n+ Tree','CuPy','PyTorch'])
    ax.margins(y=0.2);ax.grid(axis='y',alpha=0.2)
fig.suptitle('Latency Comparison | 5 warmups + 10 measurements per method and seed',fontsize=14)
fig.text(0.5,0.065,'Bars: mean of all 30 timings; dots: three seed-group means; whiskers: min–max of group means. Preallocated buffers; no H2D/D2H.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,0.1,1,0.92))
save(fig,'latency-comparison.png','benchmark-01/benchmark-summary.csv; benchmark-raw.csv')

fig,ax=plt.subplots(figsize=(9,6))
for n,color,marker in zip(sizes,('#437db1','#a276ba','#168a70'),('o','s','^')):
    rows=[r for r in trade if int(r['n'])==n]
    ax.scatter([float(r['latency_new_over_old']) for r in rows],[float(r['mean_error_reduction']) for r in rows],s=90,color=color,marker=marker,label=f'N={n} (3 seeds)')
ax.axvline(1,color='#999999',ls='--')
ax.set(xlim=(0.98,1.57),ylim=(0,9.1),xlabel='New latency / Shared32 latency (lower is faster)',
       ylabel='Shared32 mean error / new mean error (higher is more accurate)',title='Accuracy–Performance Tradeoff | full-matrix FP64 reference')
ax.grid(alpha=0.2);ax.legend(frameon=False)
ax.text(1.025,8.5,'Same kernel version in every run',fontsize=9)
fig.tight_layout(rect=(0,0.06,1,1))
save(fig,'accuracy-performance-tradeoff.png','accuracy-performance.csv')

fig,axes=plt.subplots(1,2,figsize=(14,6))
for offset,method in [(-0.18,'shared32'),(0.18,'block32_tree')]:
    group=[r for r in hist if r['n']=='4096' and r['seed_base']=='42' and r['method']==method and r['scope']=='ALL' and r['metric']=='abs_error_fp64']
    values=[int(r['count']) for r in group]
    axes[0].bar(np.arange(len(group))+offset,values,width=0.35,color=colors[method],label=labels[method])
axes[0].set(yscale='log',xticks=range(10),xticklabels=['[0,\n1e-14)','1e-14\n–1e-12','1e-12\n–1e-10','1e-10\n–1e-8','1e-8\n–1e-7','1e-7\n–1e-6','1e-6\n–1e-5','1e-5\n–1e-4','1e-4\n–1e-3','≥1e-3'],
    ylabel='Output element count',title='Absolute error distribution | N=4096, base 42')
axes[0].tick_params(axis='x',labelsize=8);axes[0].legend(frameon=False,fontsize=9);axes[0].grid(axis='y',alpha=0.2)
bottom=np.zeros(3)
for key,color in [('improved','#168a70'),('equal','#aaaaaa'),('worsened','#c94d43')]:
    values=np.array([int(next(r for r in changes if r['n']==str(n) and r['seed_base']=='42')[key])/(n*n)*100 for n in sizes])
    axes[1].bar(range(3),values,bottom=bottom,color=color,label=key)
    for i,(value,base) in enumerate(zip(values,bottom)):
        axes[1].text(i,base+value/2,f'{value:.1f}%',ha='center',va='center',fontsize=9)
    bottom+=values
axes[1].set(xticks=range(3),xticklabels=sizes,ylim=(0,110),ylabel='Share of all output positions (%)',title='Per-position change relative to FP64 | base 42')
axes[1].legend(loc='upper left',ncol=3,fontsize=8,frameon=False)
fig.suptitle('Error Distribution | improvements and regressions both retained',fontsize=15)
fig.text(0.5,0.055,'Histogram zero-count bins have no bar; no epsilon added. Right panel compares absolute errors, not closeness to NumPy.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,0.1,1,0.93))
save(fig,'error-distribution.png','accuracy-01/error-distributions.csv; improvement-summary.csv')

fig,axes=plt.subplots(1,4,figsize=(15,5))
for ax,key,title,factor in zip(axes,('registers_per_thread','shared_bytes_per_block','local_bytes_per_thread','theoretical_occupancy'),
    ('Registers per thread','Shared memory per block (KiB)','Local memory per thread (bytes)','Theoretical occupancy (%)'),(1,1/1024,1,100)):
    values=[float(r[key])*factor for r in resource]
    ax.bar([0,1],values,color=[colors['shared32'],colors['block32_tree']])
    for i,value in enumerate(values):ax.text(i,value+max(max(values),1)*0.04,f'{value:g}',ha='center')
    ax.set(xticks=[0,1],xticklabels=['Shared32','Block32\n+ Tree'],title=title,ylim=(0,max(max(values),1)*1.3))
    ax.grid(axis='y',alpha=0.2)
fig.suptitle('Register / Resource Comparison | sm_80, 1,024 threads per block',fontsize=15)
fig.text(0.5,0.085,'New kernel: 0 spill-store bytes, 0 spill-load bytes (ptxas). Both: 2 theoretical resident blocks/SM. Achieved occupancy NOT MEASURED: NCU permission blocked.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,0.14,1,0.91))
save(fig,'register-resource-comparison.png','correctness-01/kernel-resources.csv; environment/build.log')

records=[json.loads(line) for line in (ROOT/'logs/benchmark-monitor-01/gpu-load.jsonl').read_text().splitlines()]
times=np.array([datetime.fromisoformat(r['timestamp_utc']).timestamp() for r in records]);times-=times[0]
fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True)
axes[0].plot(times,[float(r['gpu']['clocks.sm']) for r in records],marker='.',color='#437db1')
axes[0].set(ylabel='SM clock (MHz)',title='Observed GPU clocks and load during formal timing')
axes[1].plot(times,[float(r['gpu']['utilization.gpu']) for r in records],marker='.',color='#168a70',label='nvidia-smi GPU utilization')
axes[1].set(ylabel='GPU utilization (%)',xlabel='Seconds from first monitor sample',ylim=(0,105))
axes[1].legend(frameon=False)
for ax in axes:ax.grid(alpha=0.2)
fig.text(0.5,0.065,'Clock was not locked. GPU utilization is a sampled device-level signal, not achieved SM occupancy. No external compute process observed.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,0.1,1,1))
save(fig,'hardware-monitor.png','logs/benchmark-monitor-01/gpu-load.jsonl')
