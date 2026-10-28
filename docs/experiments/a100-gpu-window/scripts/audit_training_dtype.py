"""Record observed training dtypes and enforce the requested pure FP32 path."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
import cupy as cp
from benchmarks.bench_cuda_training import TinyNet,initialize,parameters,SyntheticDataset
from donkeycar_hy.model import DonkeyNet
import hytorch.functions as F
from hytorch.optimizers import SGD
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--devices',nargs='+',choices=['cpu','cuda'],default=['cpu','cuda']);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
rows=[]
for device in a.devices:
 for kind,shape in [('tiny',(3,16,16)),('donkey',(3,66,200))]:
    model=TinyNet() if kind=='tiny' else DonkeyNet(dropout_rate=0.)
    initialize(model,shape,42);data=SyntheticDataset(4,shape,42)
    if device=='cuda': model.to_gpu();x=cp.asarray(data.x);t=cp.asarray(data.t)
    else: x=data.x;t=data.t
    prediction=model(x);loss=F.mean_square_error(prediction,t);loss.backward()
    gradients={k:str(v.grad.data.dtype) for k,v in parameters(model).items()};SGD(lr=.001).setup(model).update()
    rows.append(dict(device=device,model=kind,prediction=str(prediction.data.dtype),loss=str(loss.data.dtype),gradients=gradients,
                     updated={k:str(v.data.dtype) for k,v in parameters(model).items()}))
(a.output/'dtypes.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
assert all(r['prediction']==r['loss']=='float32' and set(r['gradients'].values())==set(r['updated'].values())=={'float32'} for r in rows), 'Training dtype promoted from FP32; original evidence retained'
