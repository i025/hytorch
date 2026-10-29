"""Check recorded event scopes and direct SQLite totals without GPU execution."""
import argparse
import json
from pathlib import Path
import sqlite3
from analyze import csv_rows, write_csv

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--raw',type=Path,required=True)
p.add_argument('--analysis',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a = p.parse_args()
expected = dict(A=152,B=148,C=117,D=114,E=94)
checks,removed = [],[]
for method,count in expected.items():
    folder = a.analysis/('trace-'+method)
    kernels = csv_rows(folder/'kernel-events.csv')
    measured = [r for r in kernels if r['phase'] in ['data','transfer','forward','backward','update']]
    with sqlite3.connect(str(a.raw/('trace-'+method+'.sqlite'))) as db:
        total, = db.execute('SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL').fetchone()
    assert total==len(kernels)==count*26
    assert len(measured)==count*16
    assert all(r['api_event_id']!='unmatched' for r in kernels)
    nvtx = csv_rows(folder/'nvtx-events.csv')
    if method in 'CD':
        for r in nvtx:
            if r['name']=='backward':
                ids=[k for k in kernels if int(r['sqlite_rowid']) in json.loads(k['nvtx_ids'])]
                assert len(ids)==(77 if method=='C' else 74)
        first=[r for r in measured if any(s.startswith('OP/part:Conv2d.dInput|layer=conv1|')
                                         for s in json.loads(r['operator_path']))]
        assert len(first)==(48 if method=='C' else 0)
        removed.extend(dict(method=method,kernel=r['name'],gpu_ms=r['duration_ms'],
                            correlationId=r['correlationId'],api_event_id=r['api_event_id']) for r in first)
    checks.append(dict(method=method,sqlite_total=total,measured_total=len(measured),kernels_per_step=count))
write_csv(a.output/'first-dinput-events.csv',removed)
with (a.output/'event-scope-review.json').open('x') as file:
    json.dump(dict(status='PASSED',checks=checks,scope='Raw SQLite totals and measured phase event scopes; CPU only'),file,indent=2)
print(json.dumps(checks,indent=2))
