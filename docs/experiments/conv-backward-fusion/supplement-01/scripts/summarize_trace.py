"""Summarize the complete GPU0 trace set, separately from GPU6 throughput."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT/'docs/experiments/conv-backward-fusion/scripts'))
from analyze_results import arrays,identical,error,csv_rows,write_csv,json_file

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--raw',type=Path,required=True)
p.add_argument('--analysis',type=Path,required=True)
p.add_argument('--results',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
phases=csv_rows(a.results/'trace-phases.csv')
parts=csv_rows(a.results/'gradient-components.csv')
windows=csv_rows(a.results/'backward-windows.csv')
allocations=csv_rows(a.results/'allocation-requests.csv')
rows,components,errors,checks=[],[],[],[]
base=a.raw/'trace-training-A'
initial,inputs,weights=[arrays(base/n) for n in ('initial_weights.npz','comparison_inputs.npz','final_weights.npz')]
def losses(path):
    return np.array([json.loads(v)['loss'] for v in (path/'losses.jsonl').read_text().splitlines()])
base_loss=losses(base)
for method in 'ABCDFE':
    path=a.raw/('trace-training-'+method)
    check=dict(method=method,initial_equal=identical(arrays(path/'initial_weights.npz'),initial),
        inputs_equal=identical(arrays(path/'comparison_inputs.npz'),inputs))
    assert check['initial_equal'] and check['inputs_equal']
    checks.append(check)
    actual=arrays(path/'final_weights.npz')
    references=[('A',weights,base_loss)]
    if method=='F':references.append(('D',arrays(a.raw/'trace-training-D/final_weights.npz'),losses(a.raw/'trace-training-D')))
    for reference,values,ref_loss in references:
        for name in actual:
            errors.append(dict(method=method,reference=reference,parameter=name,**error(actual[name],values[name])))
        errors.append(dict(method=method,reference=reference,parameter='loss',**error(losses(path),ref_loss)))
    pp={r['phase']:r for r in phases if r['method']==method}
    back=pp['backward'];count=int(back['nvtx_count']);assert count==16
    summary=json_file(a.analysis/('trace-training-'+method)/'profiling-summary.json')
    rr=[r for r in windows if r['method']==method]
    malloc=[r for r in allocations if r['method']==method and r['phase']=='backward' and r['kind']=='malloc']
    row=dict(method=method,steps=count,whole_trace_kernels=summary['whole_trace_kernel_count'],
        gpu_kernels_per_step=summary['training_phase_submitted_kernel_count']/count,
        backward_kernels_per_step=int(back['gpu_kernel_count'])/count,
        backward_gpu_us=float(back['gpu_kernel_sum_ms'])*1000/count,
        backward_host_nvtx_us=float(back['host_nvtx_sum_ms'])*1000/count,
        phase_launch_api_calls_per_step=sum(int(pp[phase]['launch_api_count']) for phase in ('forward','backward','update'))/count,
        backward_launch_api_us=float(back['launch_api_ms'])*1000/count,
        backward_malloc_requests_per_step=sum(int(r['calls']) for r in malloc)/count if malloc else None,
        backward_malloc_requested_bytes_per_step=sum(int(r['requested_bytes']) for r in malloc)/count if malloc else None)
    for field in ('own_kernel_span_us','own_kernel_union_us','gap_between_own_kernels_us'):
        row[field]=float(np.mean([float(r[field]) for r in rr])) if method!='E' else None
    row['kernel_window_scope']='Same-thread NVTX ancestry; E not applicable because worker-thread phase attribution does not supply ancestry IDs'
    rows.append(row)
for layer in ('conv2','conv3','conv4','conv5'):
    for method in 'DF':
        rr=[r for r in parts if (r['method'],r['part'],r['layer'])==(method,'dInput',layer)]
        assert len(rr)==16
        components.append(dict(method=method,layer=layer,ranges=len(rr),
            kernels_per_range=float(np.mean([int(r['kernels']) for r in rr])),
            kernel_us=float(np.mean([float(r['kernel_us']) for r in rr])),
            host_us=float(np.mean([float(r['host_us']) for r in rr]))))
micro=csv_rows(a.results/'micro-kernel-launches.csv');mm=[]
for method in ('original','fused','torch'):
    for layer in ('conv2','conv3','conv4','conv5'):
        for scope in ('dinput','full'):
            rr=[r for r in micro if (r['method'],r['layer'],r['scope'])==(method,layer,scope)]
            assert len(rr)==3
            row=dict(method=method,layer=layer,scope=scope,captured_calls=len(rr))
            for key in ('kernels','kernel_us','copy_kernels','col2im_kernels','gemm_kernels','gather_kernels'):
                row[key]=float(np.mean([float(r[key]) for r in rr]))
            mm.append(row)
write_csv(a.output/'trace-training-summary.csv',rows)
write_csv(a.output/'trace-dinput-summary.csv',components)
write_csv(a.output/'trace-micro-summary.csv',mm)
write_csv(a.output/'trace-numerical-errors.csv',errors)
with (a.output/'trace-checks.json').open('x') as f:json.dump(checks,f,indent=2);f.write('\n')
assert all(r['passed'] for r in errors)
print(json.dumps(dict(training=rows,dinput=components,
    instrumented_numerical_comparisons=len(errors),max_abs=max(r['max_abs'] for r in errors)),indent=2))
