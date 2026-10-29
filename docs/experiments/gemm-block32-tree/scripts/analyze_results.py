"""Audit full-position maps, derive comparisons, and inspect exceptional positions."""
import csv
import json
import math
from pathlib import Path
import re
import sqlite3
import sys
import numpy as np
from read_change_map import read_map

ROOT=Path(__file__).resolve().parents[4]
DEST=ROOT/'docs/experiments/gemm-block32-tree'
RESULTS=DEST/'results'
sys.path.insert(0,str(ROOT/'docs/experiments/gemm-fp32-accuracy/scripts'))
from accuracy import ulp_distance


def read(path):
    return list(csv.DictReader(path.open()))


def save(path,rows):
    with path.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


accuracy=read(RESULTS/'accuracy-01/full-matrix-summary.csv')
timing=read(RESULTS/'benchmark-01/benchmark-summary.csv')
changes=read(RESULTS/'accuracy-01/improvement-summary.csv')
derived=[];audits=[];exceptional=[]
for row in changes:
    n,base=int(row['n']),int(row['seed_base'])
    a={r['method']:r for r in accuracy if r['n']==str(n) and r['seed_base']==str(base) and r['scope']=='ALL'}
    t={r['method']:r for r in timing if r['n']==str(n) and r['seed_base']==str(base)}
    old,new=a['shared32'],a['block32_tree']
    derived.append(dict(n=n,seed_base=base,seed=n+base,
        mean_error_reduction=float(old['mean_abs_error'])/float(new['mean_abs_error']),
        max_error_reduction=float(old['max_abs_error'])/float(new['max_abs_error']),
        rmse_reduction=float(old['rmse'])/float(new['rmse']),
        latency_new_over_old=float(t['block32_tree']['mean_ms'])/float(t['shared32']['mean_ms']),
        latency_new_over_cupy=float(t['block32_tree']['mean_ms'])/float(t['cupy']['mean_ms']),
        latency_new_over_torch=float(t['block32_tree']['mean_ms'])/float(t['torch']['mean_ms']),
        mean_error_reduction_vs_cupy=float(a['cupy']['mean_abs_error'])/float(new['mean_abs_error']),
        mean_error_reduction_vs_torch=float(a['torch']['mean_abs_error'])/float(new['mean_abs_error']),
        improved_fraction=int(row['improved'])/(n*n),worsened_fraction=int(row['worsened'])/(n*n)))
    raw=ROOT/f'runs/gemm_block32_tree_accuracy_raw_01/n{n}-seed{base+n}'
    published=read_map(RESULTS/f'accuracy-01/n{n}-seed{base+n}-improvement-map.bin.gz',n)
    original=np.load(raw/'improvement-map.npy',mmap_mode='r')
    assert np.array_equal(published,original),'Published map does not match raw map'
    reference=np.load(raw/'reference_fp64.npy',mmap_mode='r')
    old_error=np.abs(np.load(raw/'shared32.npy',mmap_mode='r').astype(np.float64)-reference)
    new_error=np.abs(np.load(raw/'block32_tree.npy',mmap_mode='r').astype(np.float64)-reference)
    actual=np.where(new_error<old_error,2,np.where(new_error>old_error,0,1)).astype(np.uint8)
    assert np.array_equal(actual,published),'Per-position classifications do not match retained raw outputs'
    for code,key in enumerate(('worsened','equal','improved')):
        assert int(np.count_nonzero(published==code))==int(row[key])
    assert float(new_error.mean())==float(new['mean_abs_error'])
    assert float(new_error.max())==float(new['max_abs_error'])
    audits.append(dict(n=n,seed_base=base,positions=n*n,published_map_matches_raw=True,
                       all_positions_recomputed=True,new_mean_and_max_exact=True))

# Validate the single introduced NumPy failure directly against CPU fsum.
points=read(RESULTS/'accuracy-01/error-positions.csv')
introduced=[r for r in points if r['block32_tree_fail_numpy']=='True' and r['shared32_fail_numpy']=='False']
for row in introduced:
    n,base,i,j=map(int,(row['n'],row['seed_base'],row['row'],row['col']))
    raw=ROOT/f'runs/gemm_block32_tree_accuracy_raw_01/n{n}-seed{base+n}'
    a=np.load(raw/'a.npy',mmap_mode='r');b=np.load(raw/'b.npy',mmap_mode='r')
    fsum=math.fsum(float(x)*float(y) for x,y in zip(a[i],b[:,j]))
    exceptional.append(dict(n=n,seed_base=base,row=i,col=j,cpu_fsum=fsum,
        gpu_fp64=float(row['reference_fp64']),gpu_fp64_fsum_error=abs(float(row['reference_fp64'])-fsum),
        numpy_value=float(row['numpy_fp32']),numpy_fsum_error=abs(float(row['numpy_fp32'])-fsum),
        shared32_value=float(row['shared32']),shared32_fsum_error=abs(float(row['shared32'])-fsum),
        block32_tree_value=float(row['block32_tree']),block32_tree_fsum_error=abs(float(row['block32_tree'])-fsum),
        numpy_tolerance=1e-4+1e-4*abs(float(row['numpy_fp32'])),
        new_minus_numpy_abs=float(row['block32_tree_abs_numpy'])))
save(RESULTS/'accuracy-performance.csv',derived)
save(RESULTS/'map-audit.csv',audits)
if exceptional:save(RESULTS/'introduced-failure-fsum.csv',exceptional)
rounding=[r for r in read(RESULTS/'accuracy-01/reference-cross-checks.csv') if r['rounded32_match']=='False']
for row in rounding:
    row['rounded_reference_ulp_distance']=ulp_distance(float(row['gpu_fp64']),float(row['cpu_fsum']))
if rounding:save(RESULTS/'reference-rounding-differences.csv',rounding)

# Static executable instructions, not dynamic execution counts.
instructions=[]
for method,path,symbol in [('shared32',ROOT/'docs/experiments/gemm-fp32-accuracy/evidence/cuda-sass.log','_Z11gemm_sharedILi32EEvPKfS1_Pfiii'),
                           ('block32_tree',DEST/'logs/new-sass.log','_Z17gemm_block32_treePKfS0_Pfiii')]:
    content=path.read_text()
    section=content.split('Function : '+symbol,1)[1].split('Function :',1)[0]
    ops=re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]*)\s',section)
    instructions.append(dict(method=method,static_instructions=len(ops),FFMA=ops.count('FFMA'),FADD=ops.count('FADD'),
        shared_loads=sum(op.startswith('LDS') for op in ops),shared_stores=sum(op.startswith('STS') for op in ops),
        local_loads=sum(op.startswith('LDL') for op in ops),local_stores=sum(op.startswith('STL') for op in ops),
        barriers=sum(op.startswith('BAR') for op in ops),branches=ops.count('BRA'),source=str(path.relative_to(ROOT))))
save(RESULTS/'instruction-counts.csv',instructions)

# Exact captured kernel intervals with names and actual launch/resource attributes.
db=sqlite3.connect(str(ROOT/'runs/gemm_block32_tree_nsys_02.sqlite'))
db.row_factory=sqlite3.Row
events=[]
for row in db.execute('SELECT k.*,s.value AS kernel_name FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id ORDER BY k.start'):
    d=dict(row)
    keep=('start','end','deviceId','contextId','streamId','gridX','gridY','gridZ','blockX','blockY','blockZ',
          'staticSharedMemory','dynamicSharedMemory','registersPerThread','localMemoryPerThread','kernel_name')
    event={key:d[key] for key in keep}
    event['duration_ms']=(event['end']-event['start'])/1e6
    events.append(event)
save(RESULTS/'nsys-02/kernel-intervals.csv',events)
db.close()
result=dict(status='PASSED',audited_positions=sum(r['positions'] for r in audits),
    all_maps_lossless=True,all_raw_position_classifications_confirmed=True,
    introduced_numpy_failures=exceptional,rounded_reference_differences=rounding,
    instruction_counts=instructions,nsys_captured_kernel_count=len(events),kernel_versions=1)
with (RESULTS/'audit.json').open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
print(json.dumps(result,indent=2))
