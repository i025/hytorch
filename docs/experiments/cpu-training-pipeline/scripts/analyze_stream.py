"""Keep prototype and integrated stream-open measurements separate."""
import argparse
import csv
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pipeline_common import write_csv, write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    rows=[]
    comparisons=[]
    for name in ('stream-grid-01','stream-integrated-01'):
        runs=list(csv.DictReader((args.raw/name/'runs.csv').open()))
        for workers in (0,2,4):
            pair={}
            for variant in ('repaired','stream'):
                values=[r for r in runs if r['variant']==variant and int(r['workers'])==workers]
                row=dict(experiment=name,variant=variant,workers=workers,repeats=len(values))
                for key in ('images_per_s','wall_s','process_cpu_pct','peak_rss_mib','mean_next_ms'):
                    measurements=[float(r[key]) for r in values]
                    row[key+'_mean']=statistics.mean(measurements)
                    row[key+'_sd']=statistics.stdev(measurements)
                rows.append(row)
                pair[variant]=row
            ratios=[]
            for rep in range(1,pair['repaired']['repeats']+1):
                a=next(r for r in runs if r['variant']=='repaired' and int(r['workers'])==workers and int(r['repeat'])==rep)
                b=next(r for r in runs if r['variant']=='stream' and int(r['workers'])==workers and int(r['repeat'])==rep)
                ratios.append(float(b['images_per_s'])/float(a['images_per_s']))
            comparisons.append(dict(experiment=name,workers=workers,repeats=len(ratios),
                throughput_gain_pct=(pair['stream']['images_per_s_mean']/pair['repaired']['images_per_s_mean']-1)*100,
                wall_speedup=pair['repaired']['wall_s_mean']/pair['stream']['wall_s_mean'],
                repeat_throughput_ratios=ratios,faster_repeats=sum(r>1 for r in ratios)))
    write_csv(args.report/'results/stream-summary.csv',rows)
    write_json(args.report/'results/stream-comparisons.json',comparisons)
    fig,axes=plt.subplots(1,2,figsize=(11,4.6))
    for ax,name in zip(axes,('stream-grid-01','stream-integrated-01')):
        for shift,variant in ((-.18,'repaired'),(.18,'stream')):
            subset=[r for r in rows if r['experiment']==name and r['variant']==variant]
            ax.bar([i+shift for i in range(3)],[r['images_per_s_mean'] for r in subset],.36,
                   yerr=[r['images_per_s_sd'] for r in subset],capsize=3,label=variant)
        ax.set_xticks(range(3));ax.set_xticklabels(['sync','2 workers','4 workers'])
        ax.set_title(name);ax.set_ylabel('Images/s; mean +/- sample SD');ax.legend()
    fig.suptitle('Explicit file-open: unchanged JPEG/math; CPU-only matched experiments')
    fig.text(.5,.01,'Source: independent process CSVs; not GPU training. Prototype n=5, integrated opt-in API n=3.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,.96))
    fig.savefig(args.report/'figures/08-explicit-file-open.png',dpi=140)
    print(comparisons)


if __name__=='__main__':
    main()
