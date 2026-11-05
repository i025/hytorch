"""Summarize actual CPU CSVs and render scientific plots, without GPU imports."""
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
from pipeline_common import write_csv, write_json


def read(path):
    return list(csv.DictReader(Path(path).open()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    results,figures=args.report/'results',args.report/'figures'
    runs=read(args.raw/'loader-grid-01/runs.csv')
    metrics=('images_per_s','wall_s','startup_ms','shutdown_ms','mean_next_ms',
             'process_cpu_pct','peak_rss_mib')
    grouped=[]
    for variant in ('existing','repaired','inplace'):
        for workers,prefetch in [(0,2)]+[(w,p) for w in (1,2,4) for p in (1,2,4)]:
            rows=[r for r in runs if r['variant']==variant and int(r['workers'])==workers and int(r['prefetch_factor'])==prefetch]
            row=dict(variant=variant,workers=workers,prefetch_factor=prefetch,repeats=len(rows))
            for metric in metrics:
                values=[float(r[metric]) for r in rows]
                row[metric+'_mean']=statistics.mean(values)
                row[metric+'_sd']=statistics.stdev(values)
            grouped.append(row)
    write_csv(results/'loader-summary.csv',grouped)
    lookup={(r['variant'],r['workers'],r['prefetch_factor']):r for r in grouped}
    comparisons=[]
    for w,p in [(0,2)]+[(w,p) for w in (1,2,4) for p in (1,2,4)]:
        for new,old,label in (('repaired','existing','lifecycle_fix'),('inplace','repaired','normalization_candidate')):
            a,b=lookup[new,w,p],lookup[old,w,p]
            ratios=[]
            for repeat in range(1,4):
                nr=next(r for r in runs if r['variant']==new and int(r['workers'])==w and int(r['prefetch_factor'])==p and int(r['repeat'])==repeat)
                br=next(r for r in runs if r['variant']==old and int(r['workers'])==w and int(r['prefetch_factor'])==p and int(r['repeat'])==repeat)
                ratios.append(float(nr['images_per_s'])/float(br['images_per_s']))
            comparisons.append(dict(change=label,workers=w,prefetch_factor=p,
                throughput_ratio=a['images_per_s_mean']/b['images_per_s_mean'],
                throughput_gain_pct=(a['images_per_s_mean']/b['images_per_s_mean']-1)*100,
                wall_speedup=b['wall_s_mean']/a['wall_s_mean'],
                repeat_ratios=ratios,all_repeats_faster=all(r>1 for r in ratios)))
    write_csv(results/'incremental-comparisons.csv',comparisons)
    stages=read(args.raw/'profile-01/preprocess-stages.csv')
    keys=('file_read_ms','jpeg_decode_ms','rgb_ms','crop_ms','resize_ms','normalize_chw_ms')
    stage_summary=[dict(stage=k,mean_ms=statistics.mean(float(r[k]) for r in stages),
        total_ms=sum(float(r[k]) for r in stages)) for k in keys]
    write_csv(results/'preprocess-summary.csv',stage_summary)
    summary=dict(status='COMPLETED_CPU_ONLY',runs=len(runs),samples_per_run=1024,
        variants=dict(existing='Baseline loader and canonical preprocessing at 9909e54',
                      repaired='Only loader lifecycle repair; canonical preprocessing unchanged',
                      inplace='Experiment-only in-place normalization; not integrated'),
        normalization_configs_faster_mean=sum(r['throughput_ratio']>1 for r in comparisons if r['change']=='normalization_candidate'),
        normalization_configs_slower_mean=sum(r['throughput_ratio']<1 for r in comparisons if r['change']=='normalization_candidate'),
        normalization_configs_all_repeats_faster=sum(r['all_repeats_faster'] for r in comparisons if r['change']=='normalization_candidate'),
        gpu='NOT_RUN',gpu_training_speedups=dict(J_I=None,J_A=None,J_E=None),
        measurements=grouped,comparisons=comparisons,preprocess=stage_summary)
    summary['stream_comparisons'] = json.loads((results/'stream-comparisons.json').read_text())
    summary['stream_prototype_runs'] = 30
    summary['stream_integrated_runs'] = 18
    summary['total_independent_runs'] = len(runs)+30+18
    summary['open_stream_default'] = False
    suite = ET.parse(results/'host-regression-03.xml').getroot().find('testsuite')
    summary['cpu_tests'] = {key: int(suite.attrib[key]) for key in ('tests','failures','errors','skipped')}
    summary['exact_real_data_configurations'] = 40
    summary['future_policy_checks'] = 14
    write_json(results/'summary.json',summary)
    plt.rcParams.update({'figure.dpi':140,'font.size':10})
    def save(name):
        plt.figtext(.5,.01,'Source: measured CPU CSVs; data visualization, not a terminal screenshot or GPU training result',ha='center',fontsize=7)
        plt.tight_layout(rect=(0,.04,1,1))
        plt.savefig(figures/name)
        plt.close()
    x=np.arange(4)
    fig,ax=plt.subplots(figsize=(9,4.8))
    for offset,variant in enumerate(('existing','repaired','inplace')):
        values=[lookup[variant,w,2] for w in (0,1,2,4)]
        ax.bar(x+(offset-1)*.25,[r['images_per_s_mean'] for r in values],.25,
               yerr=[r['images_per_s_sd'] for r in values],capsize=3,label=variant)
    ax.set_xticks(x);ax.set_xticklabels(['sync','1 worker','2 workers','4 workers'])
    ax.set_ylabel('Images/s (mean +/- sample SD, n=3)');ax.set_title('Existing async versus this change and rejected candidate; prefetch=2')
    ax.legend();save('01-loader-throughput.png')
    fig,ax=plt.subplots(figsize=(8,4.8))
    for w in (1,2,4):
        values=[lookup['repaired',w,p] for p in (1,2,4)]
        ax.errorbar([1,2,4],[r['images_per_s_mean'] for r in values],
                    yerr=[r['images_per_s_sd'] for r in values],marker='o',capsize=4,label=f'{w} workers')
    ax.set_xticks([1,2,4]);ax.set_xlabel('Prefetch factor');ax.set_ylabel('Images/s; mean +/- SD, n=3')
    ax.set_title('Prefetch depth, repaired loader (streaming consumer)');ax.legend();save('02-prefetch-depth.png')
    fig,ax=plt.subplots(figsize=(8,4.8))
    ax.barh([s['stage'].replace('_ms','') for s in stage_summary],[s['mean_ms'] for s in stage_summary])
    ax.set_xlabel('Mean ms/image; n=1024');ax.set_title('Explicit-read CPU diagnostic; timing excluded from formal throughput')
    save('03-preprocess-stages.png')
    fig,ax=plt.subplots(figsize=(10,4.8))
    labels=[f'{w}/{p}' for w,p in [(0,2)]+[(w,p) for w in (1,2,4) for p in (1,2,4)]]
    for offset,label in enumerate(('lifecycle_fix','normalization_candidate')):
        values=[r for r in comparisons if r['change']==label]
        ax.bar(np.arange(10)+(offset-.5)*.35,[r['throughput_gain_pct'] for r in values],.35,label=label)
    ax.axhline(0,color='black',linewidth=.8);ax.set_xticks(range(10));ax.set_xticklabels(labels)
    ax.set_xlabel('Workers / prefetch');ax.set_ylabel('Ratio of mean throughput - 1 (%)')
    ax.set_title('Actual incremental changes; no established general throughput win');ax.legend();save('04-incremental-results.png')
    queues=read(args.raw/'profile-01/queue-summary.csv')
    selected=[r for r in queues if int(r['prefetch_factor'])==2]
    fig,ax=plt.subplots(figsize=(9,4.8))
    for offset,key in enumerate(('result_get_ms','result_put_ms','batch_assembly_ms')):
        ax.bar(np.arange(len(selected))+(offset-1)*.25,[float(r[key])/256 for r in selected],.25,label=key)
    ax.set_xticks(range(len(selected)));ax.set_xticklabels([r['workers']+' workers' for r in selected])
    ax.set_ylabel('Sum of instrumented durations / 256 (ms)')
    ax.set_title('Diagnostic only: queue and assembly times overlap across threads');ax.legend();save('05-queue-diagnostics.png')
    before=json.loads((results/'lifecycle-before.json').read_text())
    after=json.loads((results/'lifecycle-after.json').read_text())
    fig,ax=plt.subplots(figsize=(7.5,4.8))
    ax.bar(['before: 2 workers leaked','after: all joined'],[before['shutdown_s']*1000,after['shutdown_s']*1000])
    ax.set_yscale('log');ax.set_ylabel('Shutdown milliseconds, logarithmic scale')
    ax.set_title('Full bounded queue synthetic reproducer; one observation per version')
    save('06-worker-shutdown.png')
    fig,axes=plt.subplots(1,2,figsize=(10,4.8))
    for ax,key,label in zip(axes,('process_cpu_pct','peak_rss_mib'),('Process CPU % (100% = one core)','Whole-process lifetime peak RSS, MiB')):
        vals=[lookup['repaired',w,2] for w in (0,1,2,4)]
        ax.bar(range(4),[v[key+'_mean'] for v in vals],yerr=[v[key+'_sd'] for v in vals],capsize=3)
        ax.set_xticks(range(4));ax.set_xticklabels(['sync','1','2','4']);ax.set_xlabel('Workers');ax.set_ylabel(label)
    fig.suptitle('CPU and memory, independent child per run; initialization included in RSS')
    save('07-cpu-memory.png')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('measurements','comparisons','preprocess')}))


if __name__=='__main__':
    main()
