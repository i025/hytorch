"""Rank measured historical kernel families; do not invent node-level timings."""
import csv
import json
from pathlib import Path

DOC = Path(__file__).resolve().parents[1]
history = DOC/'results/historical-nsys-fixed-01'
summary = json.loads((history/'summary.json').read_text())
with (history/'kernel-summary.csv').open() as stream:
    rows = list(csv.DictReader(stream))
rows = [r for r in rows if r['scope']=='submitted_in_training_nvtx' and r['phase']=='backward']
if not rows:
    raise RuntimeError('Expected explicit training-scope backward kernel statistics')
locations = {
    'cuBLAS': 'hytorch/functions_conv.py:Deconv2d.forward,Conv2DGradW.forward; hytorch/functions.py:MatMul.forward(native=False)',
    'col2im': 'hytorch/functions_conv.py:_col2im_gpu',
    'im2col': 'hytorch/functions_conv.py:_im2col_gpu',
    'copy': 'CuPy array layout materialization; exact parent node NOT_MEASURED',
    'elementwise': 'hytorch/core.py arithmetic; functions.py ReLU/Tanh/MSE; exact split NOT_MEASURED',
    'reduction': 'CuPy reductions including Conv/Linear bias; exact split NOT_MEASURED',
}
groups = {}
for row in rows:
    family = row['family']
    group = groups.setdefault(family, dict(kernel_family=family, calls=0, gpu_total_ms=0.))
    group['calls'] += int(row['count'])
    group['gpu_total_ms'] += float(row['total_ms'])
backward = next(r for r in summary['phase_summary'] if r['phase']=='backward')
output = []
for rank, group in enumerate(sorted(groups.values(), key=lambda g:-g['gpu_total_ms']), 1):
    family = group['kernel_family'].lower()
    matches = [value for key,value in locations.items() if key.lower() in family]
    assert matches, family
    output.append(dict(rank=rank, **group,
        mean_kernel_us=group['gpu_total_ms']*1000/group['calls'],
        backward_gpu_time_percent=100*group['gpu_total_ms']/backward['gpu_kernel_sum_ms'],
        training_nvtx_envelope_percent=100*group['gpu_total_ms']/summary['training_nvtx_envelope_ms'],
        implementation=matches[0], scope='historical instrumented 16 steps; GPU duration sum, not CPU node duration'))
with (DOC/'results/operator-hotspots.csv').open('x', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(output[0]), lineterminator='\n')
    writer.writeheader(); writer.writerows(output)
status = dict(stage='4A', overall_status='PARTIAL_GPU_BLOCKED',
    baseline='aad4d0562fd968083a727099e478e7db64d35dd0',
    branch='experiment/donkeynet-backward-profiling',
    source_analysis='COMPLETED', historical_trace_reanalysis='PASS',
    cpu_instrumentation_tests=dict(passed=10, failed=0, skipped=0),
    new_matched_training='NOT_RUN', new_operator_nvtx='NOT_RUN',
    gpu_instrumentation_equivalence='NOT_RUN', memory_pool_callbacks='NOT_RUN',
    profiler_overhead_measurement='NOT_RUN', achieved_occupancy='BLOCKED_NO_PERMISSION',
    attempted_physical_gpu=6, gpu_jobs_started=0,
    gpu_block_reason='The selected device was occupied at the pre-launch guard; subsequent snapshot showed all eight occupied',
    planned_measurement=dict(samples=256, batch=4, epochs=4, steps_per_run=256,
        warmup=10, independent_repeats=3, methods=['cupy','combined','torch'], modes=['throughput','phases'], executed=False),
    historical_measurement=dict(samples=64, batch=4, steps_per_run=16, warmup=2,
        throughput_repeats=3, phases_repeats=1, trace_method='combined', physical_gpu=2),
    conv_backward_total_fraction=None,
    conv_backward_fraction_status='NOT_MEASURED: no historical node NVTX; col2im+im2col gives only a measured subset',
    kernel_family_ranking=output,
    historical_trace_summary='historical-nsys-fixed-01/summary.json',
    new_kernel_optimization='NOT_RUN_BY_DESIGN', production_kernels_modified=False)
(DOC/'results/profiling-summary.json').write_text(json.dumps(status, indent=2))
print(json.dumps(status, indent=2))
