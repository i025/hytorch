import json,importlib
from pathlib import Path
import torch
m=importlib.import_module('torch.optim.sgd');counts={}
for name in ['_multi_tensor_sgd','_single_tensor_sgd','_fused_sgd']:
 f=getattr(m,name)
 def wrapped(*args,_name=name,_f=f,**kw):counts[_name]=counts.get(_name,0)+1;return _f(*args,**kw)
 setattr(m,name,wrapped)
p=torch.nn.Parameter(torch.ones(16,device='cuda'));p.grad=torch.ones_like(p);o=torch.optim.SGD([p],lr=.001);o.step();torch.cuda.synchronize()
Path('runs/a100_20261106/final_training_optimization/raw/torch_dispatch.json').write_text(json.dumps(dict(counts=counts,defaults=o.defaults),indent=2))
print(counts)
