"""Full small-matrix GPU/CPU Block32-tree equivalence, including partial tiles."""
import argparse
from pathlib import Path
import ctypes as ct
import numpy as np
import cupy as cp
from common import setup, metadata, write_csv, write_json, load_library
from accuracy import blocked_fma

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--library', required=True)
p.add_argument('--output', type=Path, required=True)
args = p.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
kernel = setup(args.library)
rows = []
details = []
for m, n, k in [(3,5,31), (7,11,32), (33,35,33), (5,9,65), (17,3,127),
                (1,1,1), (2,3,255), (2,2,4095), (2,2,4096)]:
    rng = np.random.default_rng(9000 + k)
    a = rng.standard_normal((m,k), dtype=np.float32)
    b = rng.standard_normal((k,n), dtype=np.float32)
    reference = np.array([[blocked_fma(a[i], b[:,j], 32) for j in range(n)] for i in range(m)], dtype=np.float32)
    ca, cb = cp.asarray(a), cp.asarray(b)
    out = cp.empty((m,n), dtype=cp.float32)
    kernel.call(ca, cb, out)
    actual = cp.asnumpy(out)
    mismatch = int(np.count_nonzero(actual.view(np.uint32) != reference.view(np.uint32)))
    rows.append(dict(case='normal', m=m,n=n,k=k,elements=m*n,bitwise_mismatches=mismatch,
                     max_abs_difference=float(np.max(np.abs(actual.astype(np.float64)-reference))),passed=mismatch==0))
    for i in range(m):
        for j in range(n):
            details.append(dict(case=f'normal-{m}-{n}-{k}',row=i,col=j,cpu=float(reference[i,j]),gpu=float(actual[i,j])))

# Put cancellation in distinct 32-term blocks to exercise the final tree topology.
for k in (65,127,225):
    a = np.zeros((2,k),dtype=np.float32)
    b = np.ones((k,3),dtype=np.float32)
    a[0,0] = 1e8; a[0,32] = -1e8; a[0,64] = 1
    a[1,:] = np.resize(np.array([-1,1+2**-23],np.float32),k)
    b[:,1] = np.resize(np.array([1,1-2**-23],np.float32),k)
    reference=np.array([[blocked_fma(a[i],b[:,j],32) for j in range(3)] for i in range(2)],np.float32)
    out=cp.empty((2,3),cp.float32)
    kernel.call(cp.asarray(a),cp.asarray(b),out)
    actual=cp.asnumpy(out)
    mismatch=int(np.count_nonzero(actual.view(np.uint32)!=reference.view(np.uint32)))
    rows.append(dict(case='cancellation',m=2,n=3,k=k,elements=6,bitwise_mismatches=mismatch,
                     max_abs_difference=float(np.max(np.abs(actual.astype(np.float64)-reference))),passed=mismatch==0))
    for i in range(2):
        for j in range(3):
            details.append(dict(case=f'cancellation-{k}',row=i,col=j,cpu=float(reference[i,j]),gpu=float(actual[i,j])))

write_csv(args.output/'correctness.csv',rows)
write_csv(args.output/'correctness-values.csv',details)
properties=cp.cuda.runtime.getDeviceProperties(0)
resource_rows=[]
baseline=ct.cast(getattr(load_library(),'_Z11gemm_sharedILi32EEvPKfS1_Pfiii'),ct.c_void_p)
for method,symbol in [('shared32',baseline),('block32_tree',None)]:
    resource=kernel.resources(symbol)
    resource.update(method=method,threads_per_block=1024,
        max_threads_per_sm=properties['maxThreadsPerMultiProcessor'],
        theoretical_occupancy=resource['theoretical_active_blocks_per_sm']*1024/properties['maxThreadsPerMultiProcessor'],
        achieved_occupancy='NOT_MEASURED')
    resource_rows.append(resource)
write_csv(args.output/'kernel-resources.csv',resource_rows)
write_json(args.output/'metadata.json',dict(**metadata(),status='PASSED' if all(r['passed'] for r in rows) else 'FAILED',
    cases=len(rows),elements=sum(r['elements'] for r in rows),resources=resource_rows))
print(rows,flush=True)
print(resource_rows,flush=True)
assert all(r['passed'] for r in rows),'GPU does not reproduce CPU Block32 + Tree'
