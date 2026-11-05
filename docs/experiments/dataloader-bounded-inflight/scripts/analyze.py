"""Summarize actual CPU measurements and separate diagnostic wait accounting."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from common import write_csv, write_json


def read(path):
    return list(csv.DictReader(Path(path).open()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    results,figures=args.report/'results',args.report/'figures'
    runs=read(args.raw/'benchmark-01/runs.csv')
    grouped=[]
    keys=sorted({(r['workload'],int(r['workers']),int(r['prefetch_factor']),r['variant']) for r in runs})
    metrics=('images_per_s','wall_s','peak_rss_mib','mean_next_ms','p95_next_ms','startup_ms','shutdown_ms','process_cpu_pct')
    for workload,w,p,variant in keys:
        values=[r for r in runs if (r['workload'],int(r['workers']),int(r['prefetch_factor']),r['variant'])==(workload,w,p,variant)]
        row=dict(workload=workload,workers=w,prefetch_factor=p,variant=variant,repeats=len(values))
        for metric in metrics:
            numbers=[float(r[metric]) for r in values]
            row[metric+'_mean']=statistics.mean(numbers);row[metric+'_sd']=statistics.stdev(numbers)
        grouped.append(row)
    write_csv(results/'timing-summary.csv',grouped)
    lookup={(r['workload'],r['workers'],r['prefetch_factor'],r['variant']):r for r in grouped}
    comparisons=[]
    for workload,w,p in sorted({k[:3] for k in lookup}):
        old,new=lookup[workload,w,p,'baseline'],lookup[workload,w,p,'bounded']
        comparisons.append(dict(workload=workload,workers=w,prefetch_factor=p,
            baseline_images_per_s=old['images_per_s_mean'],bounded_images_per_s=new['images_per_s_mean'],
            throughput_change_pct=(new['images_per_s_mean']/old['images_per_s_mean']-1)*100,
            wall_speedup=old['wall_s_mean']/new['wall_s_mean'],
            rss_saved_mib=old['peak_rss_mib_mean']-new['peak_rss_mib_mean'],
            baseline_peak_rss_mib=old['peak_rss_mib_mean'],bounded_peak_rss_mib=new['peak_rss_mib_mean']))
    write_csv(results/'comparisons.csv',comparisons)
    diagnostics=read(args.raw/'diagnostics-01/summary.csv')
    write_csv(results/'queue-summary.csv',diagnostics)
    suite=ET.parse(results/'cpu-regression-01.xml').getroot().find('testsuite')
    summary=dict(status='COMPLETED_CPU_ONLY',baseline_commit='dc77d065d58b13e69a67bdc098147964a9e15642',
        production_source_commit=runs[0]['git_version'],window='num_workers * prefetch_factor',
        cpu_tests={k:int(suite.attrib[k]) for k in ('tests','failures','errors','skipped')},
        original_failing_bound_tests=4,matched_runs=len(runs),scaling_runs=len(read(args.raw/'benchmark-01/scaling.csv')),
        diagnostic_configurations=len(diagnostics),real_data_configurations=20,gpu='NOT_RUN',
        throughput_comparisons=comparisons)
    write_json(results/'summary.json',summary)
    plt.rcParams.update({'figure.dpi':140,'font.size':10})
    def save(name):
        plt.figtext(.5,.01,'Measured CPU data visualization; no GPU training; error bars are sample SD of 3 independent runs unless stated',ha='center',fontsize=7)
        plt.tight_layout(rect=(0,.04,1,.97));plt.savefig(figures/name);plt.close()
    settings=[(w,p) for w in (2,4) for p in (1,2,4)]
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for offset,variant in ((-.18,'baseline'),(.18,'bounded')):
        values=[lookup['straggler',w,p,variant] for w,p in settings]
        axes[0].bar(np.arange(6)+offset,[r['peak_rss_mib_mean'] for r in values],.36,
                     yerr=[r['peak_rss_mib_sd'] for r in values],capsize=3,label=variant)
        dv=[next(r for r in diagnostics if r['workload']=='straggler' and int(r['workers'])==w and int(r['prefetch_factor'])==p and r['variant']==variant) for w,p in settings]
        axes[1].bar(np.arange(6)+offset,[float(r['pending_peak_mib']) for r in dv],.36,label=variant)
    for ax in axes:
        ax.set_xticks(range(6));ax.set_xticklabels([f'{w}/{p}' for w,p in settings]);ax.set_xlabel('Workers / prefetch');ax.legend()
    axes[0].set_ylabel('Lifetime process peak RSS (MiB)');axes[1].set_ylabel('Peak array bytes in pending (MiB), diagnostic n=1')
    fig.suptitle('Delayed first batch: process memory and exact pending payload are different metrics')
    save('01-memory-bound.png')
    fig,ax=plt.subplots(figsize=(9,4.8))
    for variant in ('baseline','bounded'):
        path=args.raw/f'diagnostics-01/straggler-{variant}-w2-p2/pending-events.csv'
        values=read(path)
        ax.plot([float(r['elapsed_ms']) for r in values],[int(r['pending']) for r in values],label=variant)
    ax.axhline(4,ls='--',color='black',lw=.8,label='bounded window = 4')
    ax.set_xlabel('Diagnostic elapsed ms');ax.set_ylabel('Pending batches immediately after insertion')
    ax.set_title('Same delayed head: out-of-order buffer growth');ax.legend();save('02-pending-timeline.png')
    real_settings=[(0,2)]+[(w,p) for w in (1,2,4) for p in (1,2,4)]
    fig,ax=plt.subplots(figsize=(11,4.8))
    for shift,variant in ((-.18,'baseline'),(.18,'bounded')):
        values=[lookup['real-fast',w,p,variant] for w,p in real_settings]
        ax.bar(np.arange(10)+shift,[r['images_per_s_mean'] for r in values],.36,
               yerr=[r['images_per_s_sd'] for r in values],capsize=3,label=variant)
    ax.set_xticks(range(10));ax.set_xticklabels([f'{w}/{p}' for w,p in real_settings])
    ax.set_xlabel('Workers / prefetch');ax.set_ylabel('Real Udacity images/s');ax.set_title('Uninstrumented streaming, same images and preprocessing');ax.legend()
    save('03-real-throughput.png')
    fig,ax=plt.subplots(figsize=(10,4.8))
    matrix=np.array([[next(r['throughput_change_pct'] for r in comparisons if r['workload']==load and r['workers']==w and r['prefetch_factor']==p) for w,p in settings]for load in ('real-fast','real-consumer','straggler')])
    limit=max(abs(matrix.min()),abs(matrix.max()))
    im=ax.imshow(matrix,cmap='RdBu',vmin=-limit,vmax=limit,aspect='auto')
    ax.set_xticks(range(6));ax.set_xticklabels([f'{w}/{p}' for w,p in settings]);ax.set_xlabel('Workers / prefetch')
    ax.set_yticks(range(3));ax.set_yticklabels(['real / fast consumer','real / 4ms consumer','synthetic / delayed head'])
    for y in range(3):
        for x in range(6):ax.text(x,y,f'{matrix[y,x]:+.1f}%',ha='center',va='center')
    fig.colorbar(im,ax=ax,label='Bounded / baseline mean throughput - 1 (%)')
    ax.set_title('Keep slowdowns: finite buffering trades look-ahead for bounded memory')
    save('04-throughput-tradeoffs.png')
    fig,axes=plt.subplots(1,3,figsize=(12,4.7))
    for ax,metric in zip(axes,('result_get_ms','result_put_ms','index_get_ms')):
        for offset,variant in ((-.18,'baseline'),(.18,'bounded')):
            selected=[next(r for r in diagnostics if r['workload']==load and r['variant']==variant and int(r['workers'])==2 and int(r['prefetch_factor'])==2)for load in ('real-fast','real-consumer','straggler')]
            ax.bar(np.arange(3)+offset,[float(r[metric]) for r in selected],.36,label=variant)
        ax.set_xticks(range(3));ax.set_xticklabels(['real','consumer','head']);ax.set_title(metric);ax.set_ylabel('Summed diagnostic wall ms');ax.legend()
    fig.suptitle('Queue durations overlap across workers and include lock/scheduling waits')
    save('05-queue-waits.png')
    scaling=read(args.raw/'benchmark-01/scaling.csv')
    fig,ax=plt.subplots(figsize=(8,4.8))
    for variant in ('baseline','bounded'):
        sizes=[128,512,1024];values=[]
        for size in sizes:
            if size==512:values.append(lookup['straggler',2,2,variant]['peak_rss_mib_mean'])
            else:values.append(float(next(r['peak_rss_mib'] for r in scaling if r['variant']==variant and int(r['samples'])==size)))
        ax.plot([n//4 for n in sizes],values,marker='o',label=variant)
    ax.set_xlabel('Epoch batches (128/512/1024 samples; batch=4)');ax.set_ylabel('Lifetime process peak RSS (MiB)')
    ax.set_title('Epoch-length observations: endpoints n=1, middle mean n=3');ax.legend();save('06-epoch-memory.png')
    print(json.dumps({k:v for k,v in summary.items() if k!='throughput_comparisons'}))


if __name__=='__main__':
    main()
