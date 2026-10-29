"""Index completed A-D traces from the interrupted GPU6 profiling window."""
import argparse
import csv
import json
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--analysis',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
rows=[]
for method in 'ABCD':
    folder=a.analysis/('trace-training-'+method)
    summary=json.loads((folder/'profiling-summary.json').read_text())
    phases=list(csv.DictReader((folder/'phase-summary.csv').open()))
    back=next(r for r in phases if r['phase']=='backward')
    count=int(back['nvtx_count'])
    assert count==16 and summary['unmatched_kernel_api_count']==0
    kernels=[r for r in csv.DictReader((folder/'kernel-events.csv').open()) if r['phase']=='backward']
    row=dict(method=method,measured_steps=count,
        whole_trace_kernels=summary['whole_trace_kernel_count'],
        measured_phase_kernels=summary['training_phase_submitted_kernel_count'],
        measured_kernels_per_step=summary['training_phase_submitted_kernel_count']/count,
        backward_kernels_per_step=int(back['gpu_kernel_count'])/count,
        backward_kernel_us_per_step=float(back['gpu_kernel_sum_ms'])*1000/count,
        backward_host_nvtx_us_per_step=float(back['host_nvtx_sum_ms'])*1000/count,
        source=str(folder),scope='PARTIAL set A-D; F/E NOT_RUN; instrumented only')
    for category in ('Copy','Col2im','GEMM'):
        selected=[r for r in kernels if ('cupy_copy' in r['name'] if category=='Copy' else
            r['name'].startswith('col2im') if category=='Col2im' else 'gemm' in r['name'].lower())]
        row[category+'_kernels_per_step']=len(selected)/count
        row[category+'_kernel_us_per_step']=sum(float(r['duration_ms']) for r in selected)*1000/count
    rows.append(row)
with a.output.open('x',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print('Saved 4 completed traces; no F/E or cross-window comparison')
