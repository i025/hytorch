"""Validate real CPU format-run records; no fabricated GPU inputs or timings."""
import csv
import json
from pathlib import Path
import statistics
import numpy as np

root=Path(__file__).resolve().parents[1]/'checks/cpu-format'
with (root/'gemm.csv').open() as stream: rows=list(csv.DictReader(stream))
with (root/'raw-measurements.csv').open() as stream: raw=list(csv.DictReader(stream))
summary=json.loads((root/'summary.json').read_text())
assert summary['gpu_status']=='NOT_EXECUTED'
assert summary['warmup']==5 and summary['repeats']==10
assert summary['all_correct'] and summary['first_measured_beneficial_size'] is None
assert len(rows)==2 and len(raw)==20
for row in rows:
    values=json.loads(row['repeats_ms'])
    size=int(row['size'])
    independent=[float(r['latency_ms']) for r in raw if int(r['size'])==size]
    assert values==independent and len(values)==10
    assert row['implementation']=='numpy' and row['hardware']=='cpu'
    assert row['speedup_cupy']=='' and row['correct']=='True'
    for key,value in [('mean_ms',statistics.mean(values)),('std_ms',statistics.stdev(values)),
                      ('median_ms',statistics.median(values)),('p95_ms',np.percentile(values,95)),
                      ('gflops',2*size**3/(statistics.mean(values)*1e6))]:
        assert np.isclose(float(row[key]),value,rtol=1e-12),key
    assert json.loads(row['h2d_repeats_ms'])==json.loads(row['d2h_repeats_ms'])==[]
print('PASS: 2 real CPU rows / 20 raw measurements; mean/std/median/P95/GFLOPS agree.')
print('GPU status remains NOT_EXECUTED; no CuPy speedup or GPU winner fabricated.')
