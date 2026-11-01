"""Compare failing baseline forward paths against independent CPU FP64."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import cupy as cp
import torch

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
import hytorch.functions_conv as FC

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
rows=[]
for name in ('c23','v19'):
    with np.load(a.inputs/('test_shape_gradients[False-'+name+']-inputs.npz')) as data:
        x,w,b=[data[k] for k in ('input','weight','bias')]
    reference=torch.nn.functional.conv2d(torch.tensor(x,dtype=torch.float64),torch.tensor(w,dtype=torch.float64),torch.tensor(b,dtype=torch.float64)).numpy()
    cx,cw,cb=[cp.asarray(v) for v in (x,w,b)]
    values={'cupy':cp.asnumpy(FC.conv2d(cx,cw,cb).data)}
    tx,tw,tb=[torch.tensor(v,device='cuda') for v in (x,w,b)]
    for enabled in (True,False):
        with torch.backends.cudnn.flags(enabled=enabled,benchmark=False,deterministic=True,allow_tf32=False):
            values['torch_cudnn' if enabled else 'torch_native']=torch.nn.functional.conv2d(tx,tw,tb).cpu().numpy()
    np.savez(a.output/(name+'.npz'),input=x,weight=w,bias=b,reference_fp64=reference,**values)
    for method,v in values.items():
        d=np.abs(v.astype(np.float64)-reference)
        rows.append(dict(case=name,method=method,max_abs=float(d.max()),mismatches=int((d>1e-5+1e-4*np.abs(reference)).sum()),atol=1e-5,rtol=1e-4))
(a.output/'comparison.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
