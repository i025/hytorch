"""Sample original matrices on CPU, preserving every output and reference error."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import time
import numpy as np
from accuracy import sequential_fma,pairwise_product,blocked_fma,sequential_fp64,metrics,summarize

ROOT=Path(__file__).resolve().parents[4]
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
args.output.mkdir(parents=True,exist_ok=True)
old=json.loads((ROOT/'docs/experiments/a100-gpu-window/analysis/gemm-diagnosis-01/fp64-dot-samples.json').read_text())
rows=[];positions=[];per_size=[]
started=time.time()
for n in (1024,2048,4096):
    rng=np.random.default_rng(42+n)
    a=rng.standard_normal((n,n),dtype=np.float32);b=rng.standard_normal((n,n),dtype=np.float32)
    numpy32=np.matmul(a,b)
    chosen={}
    for record in old:
        if record['n']==n: chosen[(record['row'],record['col'])]='historical_worst'
    near=np.argpartition(np.abs(numpy32).ravel(),255)[:256]
    for index in sorted(near,key=lambda i:float(abs(numpy32.ravel()[i]))):
        chosen.setdefault(tuple(int(v) for v in np.unravel_index(index,(n,n))),'near_zero')
    sampler=np.random.default_rng(1000000+42+n)
    random_count=0
    while random_count<256:
        position=tuple(int(v) for v in sampler.integers(0,n,size=2))
        if position not in chosen and abs(float(numpy32[position]))>1.0:
            chosen[position]='random_normal';random_count+=1
    historical_mismatch=[]
    for record in old:
        if record['n']==n and float(numpy32[record['row'],record['col']])!=record['numpy_fp32']:
            historical_mismatch.append([record['row'],record['col']])
    assert not historical_mismatch,'Original NumPy inputs/reference not reproduced'
    print(json.dumps(dict(n=n,positions=len(chosen),historical_numpy_exact=True)),flush=True)
    for index,((i,j),stratum) in enumerate(chosen.items(),1):
        x=a[i,:];y=np.ascontiguousarray(b[:,j])
        reference=float(np.dot(x.astype(np.float64),y.astype(np.float64)))
        product64=x.astype(np.float64)*y.astype(np.float64)
        fsum=math.fsum(float(v) for v in product64)
        sum_abs=float(np.sum(np.abs(product64),dtype=np.float64))
        fp64_sequential=sequential_fp64(x,y)
        common=dict(n=n,row=i,col=j,stratum=stratum,near_zero_fp64=abs(reference)<=1e-2,
            reference_fp64=reference,reference_fsum64=fsum,dot_fsum_abs_difference=abs(reference-fsum),
            numpy_fp32=float(numpy32[i,j]),sum_abs_products=sum_abs,
            cancellation_condition=sum_abs/abs(reference) if reference else None,
            fp64_sequential_before_cast=fp64_sequential)
        positions.append(common)
        results=dict(numpy_fp32=numpy32[i,j],sequential_fma=sequential_fma(x,y),pairwise_fp32=pairwise_product(x,y),
            block32=blocked_fma(x,y,32),block64=blocked_fma(x,y,64),block128=blocked_fma(x,y,128),
            fp64_acc_fp32_out=np.float32(fp64_sequential))
        for strategy,actual in results.items():
            assert np.asarray(actual).dtype==np.float32
            rows.append(dict(**common,strategy=strategy,**metrics(actual,reference,numpy32[i,j])))
        if index%128==0: print(f'N={n}: {index}/{len(chosen)} positions',flush=True)
    per_size.append(dict(n=n,samples=len(chosen),full_matrix_fp64=False,historical_numpy_exact=True))
    del a,b,numpy32
with (args.output/'samples.csv').open('x',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
with (args.output/'sample-positions.json').open('x') as f:json.dump(positions,f,indent=2,allow_nan=False)
summary=dict(status='COMPLETED',scope='STRATIFIED SAMPLES ONLY, not full-matrix acceptance',
    gpu_status='NOT_RUN',started_at=started,finished_at=time.time(),
    seed='np.random.default_rng(42+N); standard_normal((N,N), dtype=float32), A then B',
    selection='All unique historical worst coordinates; 256 smallest |NumPy FP32 output|; 256 seeded random positions with |NumPy output|>1, disjoint priority in that order',
    near_zero_definition='abs(FP64 dot)<=0.01; relative errors retained and explicitly labelled',
    random_selection_seed='1000000+42+N',atol=1e-4,rtol=1e-4,
    ulp_definition='Distance in ordered binary32 representations to round32(FP64 dot), signed zero unified',
    fp64_method='NumPy dot of exact binary32-to-binary64 conversions; cross-check math.fsum of binary64 exact products',
    strategies=dict(numpy_fp32='Original full NumPy FP32 matmul, 4 OpenBLAS threads',sequential_fma='libm.fmaf ascending K; one binary32 rounding per multiply-add',
        pairwise_fp32='Explicit binary32 products followed by adjacent balanced tree of binary32 adds',
        blocks='32/64/128 ascending-K fmaf within each block, then binary32 balanced tree over partial sums',
        fp64_acc_fp32_out='Independent ascending-K binary64 product/add, one final binary32 conversion'),
    sizes=per_size,reference_max_dot_fsum_difference=max(p['dot_fsum_abs_difference'] for p in positions),
    reference_rounded32_disagreements=sum(bool(np.float32(p['reference_fp64'])!=np.float32(p['reference_fsum64'])) for p in positions),
    groups=summarize(rows))
with (args.output/'cpu-summary.json').open('x') as f:json.dump(summary,f,indent=2,allow_nan=False)
print(json.dumps(dict(status=summary['status'],positions=len(positions),rows=len(rows),elapsed_s=time.time()-started)),flush=True)
