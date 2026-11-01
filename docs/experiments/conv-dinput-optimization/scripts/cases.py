"""Reuse all 4E geometries and predeclare a separate unseen verification set."""
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
spec=importlib.util.spec_from_file_location('stage4e_cases',ROOT/'docs/experiments/conv-shape-dispatch/scripts/cases.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
FRESH=[
    (1,3,17,5,13,1,1,1,1,0,0),(4,16,31,9,21,1,1,1,1,0,0),
    (16,32,65,7,9,1,1,1,1,0,0),(1,64,32,19,23,1,1,2,2,1,1),
    (4,3,64,17,29,3,3,1,1,1,1),(16,16,16,11,19,3,3,2,2,0,0),
    (1,32,64,29,41,3,3,1,1,0,0),(4,64,32,13,23,3,3,2,2,1,1),
    (16,3,32,9,21,5,5,1,1,1,1),(1,16,64,23,39,5,5,2,2,0,0),
    (4,32,16,17,25,5,5,1,1,0,0),(16,64,64,7,19,5,5,2,2,1,1),
    (2,8,3,11,23,1,1,1,1,0,0),(8,24,7,9,25,3,3,2,2,1,1),
    (1,48,33,15,29,5,5,1,1,1,1),(4,8,63,21,35,3,3,1,1,0,0),
    (16,24,19,9,15,1,1,2,2,0,0),(2,64,47,19,31,5,5,2,2,1,1),
    (1,3,16,13,27,2,3,1,2,0,1),(4,16,32,15,23,3,5,2,1,1,0),
    (16,32,64,7,17,5,3,1,2,0,1),(1,64,17,17,33,3,1,2,1,1,0),
    (4,3,1,7,13,1,1,1,1,0,0),(16,16,2,9,17,3,3,2,2,1,1),
]


def cases(split):
    if split=='history':
        return [r for s in ('calibration','held-out','validation-2','validation-final') for r in previous.cases(s)]
    result=[]
    for i,(n,ci,co,h,w,kh,kw,sh,sw,ph,pw) in enumerate(FRESH):
        result.append(dict(case='o%02d'%i,split='fresh',shape=(n,ci,h,w),weight_shape=(co,ci,kh,kw),
            stride=(sh,sw),pad=(ph,pw),work=n*ci*h*w*co*kh*kw/(sh*sw)))
    return result


if __name__=='__main__':
    import sys
    with Path(sys.argv[1]).open('x') as f:json.dump({s:cases(s) for s in ('history','fresh')},f,indent=2);f.write('\n')
