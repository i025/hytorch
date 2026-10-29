"""Check retained numerical evidence and export reference-decision disagreements."""
import argparse
import csv
import json
import math
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--results', type=Path, required=True)
args = p.parse_args()
root = args.results
rows = list(csv.DictReader((root / 'samples.csv').open()))
repeat = list(csv.DictReader((root / 'completed-03/samples.csv').open()))
assert rows == repeat, 'Serialization-failed and completed runs differ numerically'
assert len(rows) == 10836
assert all(math.isfinite(float(r[k])) for r in rows for k in ('value', 'reference_fp64', 'abs_error_fp64', 'relative_error_fp64'))
gpu = list(csv.DictReader((root / 'gpu-01/gpu-samples.csv').open()))
assert len(gpu) == len(rows)
summary = json.loads((root / 'summary.json').read_text())
assert summary['gpu_status'] == 'EXECUTED'
assert all(r['unequal_to_sequential_fma'] == 0 for r in summary['sampled_gpu_vs_cpu_fma'] if r['strategy'] not in ('cupy', 'torch'))
assert all(r['unequal_elements'] == 0 for r in summary['full_matrix_gpu_comparisons'] if r['n'] >= 2048 and r['b'] != 'numpy_fp32')
disagreements = [r for r in rows if r['strategy'] == 'sequential_fma' and r['pass_original_numpy'] != r['pass_same_tolerance_fp64']]
with (root / 'reference-decision-disagreements.csv').open('x', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(disagreements)
result = dict(status='COMPLETED', cpu_rows=len(rows), gpu_rows=len(gpu),
              cpu_attempt_02_and_03_rows_exactly_equal=True, finite_values=True,
              reference_decisions=[dict(n=n,
                  numpy_pass_fp64_fail=sum(r['n'] == str(n) and r['pass_original_numpy'] == 'True' for r in disagreements),
                  numpy_fail_fp64_pass=sum(r['n'] == str(n) and r['pass_original_numpy'] == 'False' for r in disagreements))
                  for n in (1024, 2048, 4096)])
with (root / 'audit.json').open('x') as stream:
    json.dump(result, stream, indent=2)
print(json.dumps(result, indent=2))
