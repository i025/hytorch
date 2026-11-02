"""CPU-only tables from preserved correctness, memory and correlated traces."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET


def read(path):
    with path.open() as f:return list(csv.DictReader(f))


def write(path,rows):
    with path.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,required=True);p.add_argument('--results',type=Path,required=True)
    a=p.parse_args();root=a.results
    phases=read(root/'trace-phases.csv');parts=read(root/'dweight-components.csv')
    windows=read(root/'backward-windows.csv');totals=[];layers=[]
    for model in ('donkey','spatial'):
        for method in 'AFIE':
            dw=[r for r in parts if (r['model'],r['method'])==(model,method)]
            row=dict(model=model,method=method)
            for metric in ('host_us','kernel_us','kernels','Copy_count','Im2col_count','GEMM_count','tiles_count','reduce_count'):
                row['dweight_'+metric]=sum(float(r[metric]) for r in dw)/16
            for layer in sorted({r['layer'] for r in dw}):
                rr=[r for r in dw if r['layer']==layer]
                layers.append(dict(model=model,method=method,layer=layer,calls=len(rr),
                    **{k:statistics.mean(float(r[k]) for r in rr) for k in ('host_us','kernel_us','kernels')}))
            backward=next(r for r in phases if (r['model'],r['method'],r['phase'])==(model,method,'backward'))
            for metric in ('gpu_kernel_count','gpu_kernel_sum_ms','launch_api_count','launch_api_ms','cuda_api_count','cuda_api_sum_ms','host_nvtx_sum_ms'):
                row['backward_'+metric+'_per_step']=float(backward[metric])/16
            rr=[r for r in windows if (r['model'],r['method'])==(model,method)]
            for metric in ('host_us','kernel_union_us','kernel_span_us','between_own_kernels_us'):
                row['backward_'+metric]=statistics.mean(float(r[metric]) for r in rr)
            totals.append(row)
    write(root/'trace-totals.csv',totals);write(root/'dweight-layers.csv',layers)
    tests=[]
    for name in ('correctness-new','correctness-regression'):
        rr=ET.parse(a.raw/(name+'.xml')).getroot().findall('.//testcase')
        row=dict(suite=name,total=len(rr),failures=sum(r.find('failure') is not None for r in rr),
            errors=sum(r.find('error') is not None for r in rr),skipped=sum(r.find('skipped') is not None for r in rr))
        assert row['failures']==row['errors']==row['skipped']==0;tests.append(row)
    errors=read(a.raw/'correctness-new-arrays/comparisons.csv');numeric=[]
    for label,test in [('all',lambda r:True),('dWeight',lambda r:'_dWeight_' in r['case']),
                       ('dInput',lambda r:'_dInput_' in r['case']),('dBias',lambda r:'_dBias_' in r['case'])]:
        rr=[r for r in errors if test(r)]
        assert all(r['finite']=='True' and int(r['mismatches'])==0 for r in rr)
        numeric.append(dict(scope=label,comparisons=len(rr),max_abs=max(float(r['max_abs']) for r in rr),
            max_relative_l2=max(float(r['relative_l2'] or 0) for r in rr),mismatches=0))
    memory=read(root/'micro/memory.csv');mem=[]
    for method in ('original','fused','torch'):
        rr=[r for r in memory if r['method']==method]
        mem.append(dict(method=method,maximum_measured_peak_bytes=max(int(r['peak_live_bytes']) for r in rr),
            maximum_temporary_bytes=max(int(r['temporary_peak_bytes']) for r in rr)))
    rr=[r for r in memory if r['method']=='fused' and r['repeat']=='1']
    workspace=dict(smaller_than_old_col=sum(int(r['native_partial_bytes'])<int(r['im2col_bytes']) for r in rr),
        equal_to_old_col=sum(int(r['native_partial_bytes'])==int(r['im2col_bytes']) for r in rr),
        larger_than_old_col=sum(int(r['native_partial_bytes'])>int(r['im2col_bytes']) for r in rr),
        max_new_over_old=max(int(r['native_partial_bytes'])/int(r['im2col_bytes']) for r in rr))
    result=dict(gpu_tests=tests,numerical=numeric,memory=mem,logical_workspace=workspace)
    with (root/'acceptance-checks.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
