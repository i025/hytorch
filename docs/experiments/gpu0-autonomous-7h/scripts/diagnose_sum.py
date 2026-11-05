"""Retain original tolerances and distinguish rounding from CPU-model disagreement."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from common import ROOT,FOUNDATION,write_json,write_csv
from cases import CASES
from reference import modeled_sum,reference_sum
from hytorch.backends.cuda import backend
from hytorch.backends.cuda.runtime import sum_array


def stats(a,b):
    delta=a.astype(np.float64)-np.asarray(b,dtype=np.float64)
    return dict(max_abs=float(np.abs(delta).max()),
        relative_l2=float(np.linalg.norm(delta.ravel())/max(np.linalg.norm(np.asarray(b,dtype=np.float64).ravel()),1e-30)),
        mismatches=int((np.abs(delta)>1e-5+1e-4*np.abs(b)).sum()),unequal=int((a!=b).sum()),
        finite=bool(np.isfinite(a).all() and np.isfinite(b).all()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--run-gpu',action='store_true',required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    import cupy as cp
    import torch
    cp.cuda.Device(0).use();torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rows=[];rng=np.random.default_rng(42);model_disagreements=0
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
                with backend(reduction='native'):
                    actual=cp.asnumpy(sum_array(cp.asarray(x),axis,keepdims))
                tx=torch.tensor(x,device='cuda')
                target=tx+0. if axis==() else torch.sum(tx,dim=axis,keepdim=keepdims)
                refs=dict(cupy=cp.asnumpy(cp.asarray(x).sum(axis=axis,keepdims=keepdims)),
                    torch=target.cpu().numpy(),numpy=x.sum(axis=axis,keepdims=keepdims),
                    fp64=reference_sum(x,axis,keepdims),cpu_model=modeled_sum(x,axis,keepdims))
                for ref,value in refs.items():
                    row=dict(case=case,shape=str(shape),axis=str(axis),input=kind,keepdims=keepdims,
                        reference=ref,atol=1e-5,rtol=1e-4,**stats(actual,value))
                    row['status']='PASS' if row['finite'] and not row['mismatches'] else 'FAIL'
                    rows.append(row)
                    if ref=='cpu_model':model_disagreements+=row['unequal']
                np.savez(a.output/f'case-{case}-{kind}-{keepdims}.npz',x=x,actual=actual,**refs)
    write_csv(a.output/'errors.csv',rows)
    eligible=[case for case in range(len(CASES)) if all(r['status']=='PASS'
        for r in rows if r['case']==case and r['reference'] in ('cupy','torch','fp64'))]
    ordinary=[r for r in rows if r['input']!='cancellation' and r['reference'] in ('cupy','torch','fp64')]
    value=dict(status='PASS' if all(r['status']=='PASS' for r in rows) else 'NUMERICAL_FAILURES_RETAINED',
        comparisons=len(rows),failed=sum(r['status']=='FAIL' for r in rows),
        cpu_model_unequal_elements=model_disagreements,eligible_cases=eligible,
        ordinary_inputs_pass=all(r['status']=='PASS' for r in ordinary),
        gpu='EXECUTED',scope='Full tensors, three deterministic input families; no tolerance relaxation',
        interpretation='FP32 order attribution requires model agreement; failed cases remain FAIL even if explained')
    write_json(a.output/'summary.json',value);print(json.dumps(value),flush=True)
    # A model disagreement is an implementation investigation, never a rounding waiver.
    assert model_disagreements==0,'GPU differs from declared FP32 CPU arithmetic model'


if __name__=='__main__':main()
