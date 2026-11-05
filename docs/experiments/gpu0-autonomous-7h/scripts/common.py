"""Reuse the canonical 4H states and existing kernels; keep loader ablations explicit."""
import importlib.util
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[4]
FOUNDATION=ROOT/'docs/experiments/cuda-reduction-foundation/scripts'
sys.path.insert(0,str(FOUNDATION))
spec=importlib.util.spec_from_file_location('foundation_common',FOUNDATION/'training_common.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
from hytorch.backends.cuda import backend

METHODS=('A','I','P','R','S','J','E','E-prefetch')


class Engine(previous.Engine):
    def __init__(self,method,model='donkey',device='cuda',sum_enabled=False,relu_enabled=True):
        mode=('E' if method=='E-prefetch' else 'I' if method=='P' else
              ('SR' if sum_enabled and relu_enabled else 'S' if sum_enabled else 'R' if relu_enabled else 'I')
              if method=='J' else method)
        super().__init__(mode,model,device)


def configuration(method,model,protocol,steps=256,sum_enabled=False,relu_enabled=True):
    return dict(method=method,model=model,steps=steps,warmup_steps=10,batch=4,seed=42,
        dtype='float32',lr=.001,dropout=0,tf32=False,open_stream=True,
        workers=2 if method in ('P','J','E-prefetch') else 0,prefetch_factor=2,
        indices=protocol['throughput'][:steps*4],preprocessing=protocol['preprocessing'],
        sum_native=method=='S' or (method=='J' and sum_enabled),
        relu_native=method=='R' or (method=='J' and relu_enabled),
        j_definition='I + same bounded CPU prefetch + independently accepted Sum/ReLU only',
        no_fingerprints=True)


compare=previous.compare
result=previous.result
write_json=previous.write_json
write_csv=previous.write_csv
