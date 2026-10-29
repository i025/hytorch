"""Derive report numbers and measured dispatch host ranges from saved tables."""
import csv
import json
from pathlib import Path
import statistics

BASE=Path(__file__).resolve().parents[1]


def read(path):
    with path.open() as f:return list(csv.DictReader(f))


summary={};errors=[]
for name in ('calibration-01','held-out-01','validation-2-01','pointwise-validation-01','actual-auto-01'):
    rows=read(BASE/'results'/name/'errors.csv');errors.extend(rows)
    summary[name]=dict(comparisons=len(rows),max_abs=max(float(r['max_abs']) for r in rows))
assert all(int(r['mismatches'])==0 and r['finite']=='True' for r in errors)
summary['shape_comparisons']=len(errors)
rows=read(BASE/'results/actual-auto-01/comparisons.csv')
for mode in ('fused','original'):
    rr=[r for r in rows if r['actual_auto_selected']==mode]
    summary['auto_'+mode]=dict(count=len(rr),minimum_ratio=min(float(r['actual_auto_speedup']) for r in rr),
        maximum_ratio=max(float(r['actual_auto_speedup']) for r in rr),
        mean_wall_delta_us=statistics.mean(float(r['auto_wall_us'])-float(r['original_wall_us']) for r in rr))
raw=Path(json.loads((BASE/'results/profiles-01/raw-location.json').read_text())['analysis'])
dispatch=[]
for model in ('donkey','spatial'):
    rows=read(raw/('trace-training-'+model+'-G')/'nvtx-events.csv')
    rr=[r for r in rows if r['name']=='OP/dispatch:conv_dinput' and r['phase']=='backward']
    summary[model+'_dispatch_host']=dict(calls=len(rr),mean_us=statistics.mean(float(r['duration_ms'])*1000 for r in rr),
        timing='NVTX instrumented host scope, not the uninstrumented dispatch overhead')
    dispatch.extend(dict(model=model,nvtx_id=r['sqlite_rowid'],host_us=float(r['duration_ms'])*1000) for r in rr)
    summary[model+'_trace']={}
    for method in 'DFG':
        rows=read(BASE/'results/profiles-01/trace-phases.csv')
        r=next(r for r in rows if (r['model'],r['method'],r['phase'])==(model,method,'backward'))
        summary[model+'_trace'][method]=dict(kernels_per_step=int(r['gpu_kernel_count'])/16,
            gpu_kernel_us_per_step=float(r['gpu_kernel_sum_ms'])*1000/16,host_nvtx_ms_per_step=float(r['host_nvtx_sum_ms'])/16)
with (BASE/'results/profiles-01/dispatch-host.csv').open('x',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(dispatch[0]));writer.writeheader();writer.writerows(dispatch)
with (BASE/'results/report-numbers.json').open('x') as f:json.dump(summary,f,indent=2);f.write('\n')
print(json.dumps(summary,indent=2))
