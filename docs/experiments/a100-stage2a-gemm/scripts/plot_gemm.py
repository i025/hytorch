"""Plot only measured GPU GEMM CSV data; no fabricated timings or interpolation."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ORDER = ['cupy','basic','blocked-16','blocked-32','shared-16','shared-32']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with (args.input/'gemm.csv').open() as stream:
        rows=list(csv.DictReader(stream))
    gpu=[r for r in rows if r['hardware']=='gpu']
    if not gpu:
        parser.error('No GPU measurements: cannot generate GPU comparison figures')
    args.output.mkdir(parents=True,exist_ok=False)
    for row in gpu:
        row['label']=row['implementation']+('-'+row['tile'] if row['tile'] else '')
    sizes=sorted({int(r['size']) for r in gpu})
    colors=dict(zip(ORDER,plt.get_cmap('tab10').colors))
    plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.2})

    def selected(label):
        return sorted([r for r in gpu if r['label']==label],key=lambda r:int(r['size']))

    def save(fig,name):
        fig.text(.5,.012,'Measured data; x marks numerical-check failures; no smoothing',ha='center',fontsize=9)
        fig.tight_layout(rect=[0,.045,1,1])
        fig.savefig(args.output/name,dpi=170)
        plt.close(fig)

    for key,ylabel,title,name,log,labels in [
        ('mean_ms','Mean GPU compute latency (ms)','GEMM latency comparison','gemm-latency.png',True,ORDER),
        ('speedup_cupy','CuPy mean / implementation mean','Speedup over CuPy (device-resident compute)','speedup-vs-cupy.png',True,ORDER[1:]),
        ('gflops','GFLOPS = 2N³ / mean latency','Effective GPU GFLOPS','gemm-gflops.png',True,ORDER)]:
        fig,ax=plt.subplots(figsize=(9,5))
        for label in labels:
            data=selected(label)
            ax.plot([int(r['size']) for r in data],[float(r[key]) for r in data],'o-',label=label,color=colors[label])
            failed=[r for r in data if r['correct']!='True']
            ax.scatter([int(r['size']) for r in failed],[float(r[key]) for r in failed],marker='x',color='red',s=100,zorder=5)
        if key=='speedup_cupy': ax.axhline(1,color='black',linestyle='--',label='1× CuPy')
        ax.set(xlabel='Square matrix size N',ylabel=ylabel,title=title)
        if log: ax.set_yscale('log')
        ax.set_xticks(sizes); ax.legend(ncol=2); save(fig,name)

    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,method in zip(axes,['blocked','shared']):
        for tile in ('16','32'):
            label=method+'-'+tile
            data=selected(label)
            ax.errorbar([int(r['size']) for r in data],[float(r['mean_ms']) for r in data],
                        yerr=[float(r['std_ms']) for r in data],fmt='o-',capsize=4,label='tile '+tile,color=colors[label])
            failed=[r for r in data if r['correct']!='True']
            ax.scatter([int(r['size']) for r in failed],[float(r['mean_ms']) for r in failed],marker='x',color='red',s=100)
        ax.set(xlabel='Square matrix size N',ylabel='Mean ± sample std (ms)',title=method,yscale='log')
        ax.set_xticks(sizes); ax.legend()
    save(fig,'tile-comparison.png')

    fig,axes=plt.subplots(1,len(sizes),figsize=(5*len(sizes),5),squeeze=False)
    for ax,size in zip(axes[0],sizes):
        data=[next(r for r in selected(label) if int(r['size'])==size) for label in ORDER]
        x=np.arange(len(data)); width=.24
        for offset,key,label in [(-width,'h2d_ms','H2D: both inputs'),(0,'mean_ms','GEMM compute'),(width,'d2h_ms','D2H: output')]:
            ax.bar(x+offset,[float(r[key]) for r in data],width=width,label=label)
        ax.set(yscale='log',ylabel='Separately measured latency (ms)',title=f'N={size}')
        ax.set_xticks(x,[r['label']+('*' if r['correct']!='True' else '') for r in data],rotation=45,ha='right')
        ax.legend(fontsize=8)
    fig.suptitle('Pageable transfers versus device-resident compute; sums are NOT measured end-to-end\n* = numerical-check failure')
    save(fig,'transfer-vs-compute.png')

    fig,axes=plt.subplots(1,len(sizes),figsize=(5*len(sizes),5),squeeze=False)
    for ax,size in zip(axes[0],sizes):
        data=[next(r for r in selected(label) if int(r['size'])==size) for label in ORDER]
        values=[json.loads(r['repeats_ms']) for r in data]
        ax.boxplot(values,labels=[r['label']+('*' if r['correct']!='True' else '') for r in data],showfliers=False)
        for i,items in enumerate(values,1):
            ax.scatter(np.full(len(items),i),items,s=15,alpha=.65,color=colors[data[i-1]['label']])
        ax.set(yscale='log',ylabel='Individual compute latency (ms)',title=f'N={size}')
        ax.tick_params(axis='x',rotation=45)
    fig.suptitle('Latency variability: all repeats retained; * = numerical-check failure')
    save(fig,'latency-variability.png')


if __name__=='__main__':
    main()
