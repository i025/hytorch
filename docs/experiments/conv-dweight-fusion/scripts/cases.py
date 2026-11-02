"""All 124 previous generic geometries plus eight actual CNN operator shapes."""
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
spec=importlib.util.spec_from_file_location('stage4f_cases',ROOT/'docs/experiments/conv-dinput-optimization/scripts/cases.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)


def cases():
    result=previous.cases('history')+previous.cases('fresh')
    layers=[((4,3,66,200),(24,3,5,5),2,0),((4,24,31,98),(36,24,5,5),2,0),
        ((4,36,14,47),(48,36,5,5),2,0),((4,48,5,22),(64,48,3,3),1,0),
        ((4,64,3,20),(64,64,3,3),1,0),((4,3,16,16),(16,3,3,3),1,1),
        ((4,16,16,16),(32,16,3,3),2,1),((4,32,8,8),(32,32,1,1),1,0)]
    for i,(shape,weight,stride,pad) in enumerate(layers):
        result.append(dict(case='n%02d'%i,split='network',shape=shape,weight_shape=weight,stride=stride,pad=pad))
    return result


if __name__=='__main__':
    import sys
    with Path(sys.argv[1]).open('x') as f:json.dump(cases(),f,indent=2);f.write('\n')
