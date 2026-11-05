"""Run CPU numerical references and descriptor mapping; save every error statistic."""
import argparse
import csv
import ctypes as ct
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from hytorch.backends.cuda.reduction import plan_sum, SumLayout
from cases import CASES
from reference import modeled_sum, reference_sum


def compare(value, reference):
    delta=value.astype(np.float64)-reference.astype(np.float64)
    limit=1e-5+1e-4*np.abs(reference.astype(np.float64))
    return dict(max_abs=float(np.abs(delta).max()),
                relative_l2=float(np.linalg.norm(delta.ravel())/max(np.linalg.norm(reference.ravel()),1e-30)),
                mismatches=int((np.abs(delta)>limit).sum()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    rows=[];rng=np.random.default_rng(42)
    for case,(shape,axis) in enumerate(CASES):
        for kind in ('integer','random','cancellation'):
            size=int(np.prod(shape,dtype=int))
            if kind=='integer':x=(np.arange(size,dtype=np.float32)%17).reshape(shape)
            else:
                x=rng.standard_normal(size).astype(np.float32).reshape(shape)
                if kind=='cancellation':
                    x.ravel()[::2]*=np.float32(-1)
                    x-=x.mean(dtype=np.float64).astype(np.float32)
            for keepdims in (False,True):
                actual=modeled_sum(x,axis,keepdims)
                numpy=x.sum(axis=axis,keepdims=keepdims)
                fp64=reference_sum(x,axis,keepdims)
                np.savez(a.output/f'case-{case}-{kind}-{keepdims}.npz',
                    x=x,modeled=actual,numpy_fp32=numpy,fp64=fp64)
                layout,output,parts=plan_sum(shape,axis,keepdims)
                # Verify the descriptor's full Cartesian mapping independently of transpose.
                def offsets(count,ndim,dims,strides):
                    index=np.arange(count,dtype=np.int64);result=np.zeros(count,np.int64)
                    for d in range(ndim-1,-1,-1):
                        result+=(index%dims[d])*strides[d];index//=dims[d]
                    return result
                o=offsets(layout.outputs,layout.out_ndim,layout.out_shape,layout.out_stride)
                r=offsets(layout.reduction,layout.reduce_ndim,layout.reduce_shape,layout.reduce_stride)
                mapping=(o[:,None]+r[None,:]).ravel()
                assert np.array_equal(np.sort(mapping),np.arange(x.size))
                for ref_name,target in (('numpy_fp32',numpy),('fp64',fp64)):
                    stats=compare(actual,np.asarray(target))
                    row=dict(case=case,shape=str(shape),axis=str(axis),keepdims=keepdims,input=kind,
                        reference=ref_name,parts=parts,output_elements=actual.size,atol=1e-5,rtol=1e-4,
                        **stats,status='PASS' if stats['mismatches']==0 else 'FAIL')
                    rows.append(row)
                    if row['mismatches']:
                        np.savez(a.output/f'failure-c{case}-{kind}-{keepdims}-{ref_name}.npz',x=x,actual=actual,reference=target)
    with (a.output/'comparisons.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    result=dict(status='PASS' if all(r['status']=='PASS'for r in rows) else 'NUMERICAL_FAILURES_RETAINED',
        comparisons=len(rows),failed=sum(r['status']=='FAIL'for r in rows),
        max_abs=max(r['max_abs']for r in rows),descriptor_bytes=ct.sizeof(SumLayout),
        scope='CPU FP32 arithmetic and index model only, not CUDA execution',gpu='NOT_RUN',
        finished_at_utc=datetime.now(timezone.utc).isoformat())
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    sys.exit(1 if result['failed'] else 0)


if __name__=='__main__':main()
