"""Matched paired training; separate wall and Nsight evidence, no production hooks."""
import sys,time,json,csv,argparse,contextlib
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import numpy as np
import cupy as cp
import torch
import hytorch.functions as F
from hytorch.backends.cuda.runtime import backend
from hytorch.optimizers import SGD
from benchmarks.bench_cuda_training import SyntheticDataset,initialize,parameters,snapshot,torch_model,state_hash
from hytorch.dataloaders import DataLoader
from donkeycar_hy.model import DonkeyNet
from legacy import legacy
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--trace',choices=['before_native','after_native','after_cupy','torch']);a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
torch.manual_seed(42);torch.set_num_threads(4)
cp.cuda.Device(0).use();sync=cp.cuda.get_current_stream().synchronize
model=DonkeyNet(dropout_rate=0.);initial=initialize(model,(3,66,200),42);model.to_gpu();opt=SGD(.001).setup(model)
tm=torch_model('donkey',initial).cuda();to=torch.optim.SGD(tm.parameters(),lr=.001)
tsaved={k:v.detach().clone() for k,v in tm.state_dict().items()}
ds=SyntheticDataset(256,(3,66,200),42)
@contextlib.contextmanager
def phase(name):
    cp.cuda.nvtx.RangePush('phase:'+name)
    try:yield
    finally:cp.cuda.nvtx.RangePop()
def restore(mode):
    if mode=='torch':tm.load_state_dict(tsaved);to.zero_grad(set_to_none=True);torch.cuda.synchronize()
    else:
        for k,v in parameters(model).items():v.data=cp.asarray(initial[k].copy())
        model.cleargrads();sync()
def step(mode,batch):
    with phase('transfer'):
        if mode=='torch':x,t=[torch.from_numpy(v).cuda() for v in batch]
        else:x,t=map(cp.asarray,batch)
    with phase('forward'):
        loss=((tm(x)-t)**2).mean() if mode=='torch' else F.mean_square_error(model(x),t)
    with phase('backward'):
        if mode=='torch':to.zero_grad(set_to_none=True);loss.backward()
        else:model.cleargrads();loss.backward()
    with phase('update'):
        if mode=='torch':to.step()
        else:opt.update()
    return loss
rows=[];finals={};audit=[]
modes=[a.trace] if a.trace else ['before_native','after_native','after_cupy','torch']
for mode in modes:
    restore(mode)
    with legacy(mode=='before_native'), backend(matmul='shared',conv='im2col',pool='native') if 'native' in mode else backend():
        batch=next(iter(DataLoader(ds,4,shuffle=False)))
        for _ in range(10):step(mode,batch)
    torch.cuda.synchronize()
for repeat in range(1 if a.trace else 5):
    for mode in (modes if repeat%2==0 else modes[::-1]):
        restore(mode)
        with legacy(mode=='before_native'), backend(matmul='shared',conv='im2col',pool='native') if 'native' in mode else backend():
            if a.trace:cp.cuda.runtime.profilerStart()
            torch.cuda.synchronize();start=time.perf_counter();count=0
            for epoch in range(1 if a.trace else 3):
                it=iter(DataLoader(ds,4,shuffle=False))
                for i in range(10 if a.trace else 64):
                    cp.cuda.nvtx.RangePush(f'step:{count}')
                    with phase('data'):batch=next(it)
                    loss=step(mode,batch)
                    cp.cuda.nvtx.RangePop();count+=1
            with phase('final_sync'):torch.cuda.synchronize()
            elapsed=time.perf_counter()-start
            if a.trace:cp.cuda.runtime.profilerStop()
        if mode=='torch':
            final={}
            for k,pv in parameters(model).items():
                layer,n=k.split('/');v=getattr(getattr(tm,layer),'weight' if n=='W' else 'bias')
                ar=v.detach().cpu().numpy();final[k]=ar.T.copy() if ar.ndim==2 else ar.copy()
                audit.append(dict(mode=mode,repeat=repeat,parameter=k,param_dtype=str(v.dtype),grad_dtype=str(v.grad.dtype),sgd_delta_dtype=str((.001*v.grad).dtype)))
            value=float(loss.detach().cpu())
        else:
            final=snapshot(model);value=float(loss.data.item())
            for k,v in parameters(model).items():audit.append(dict(mode=mode,repeat=repeat,parameter=k,param_dtype=str(v.dtype),grad_dtype=str(v.grad.dtype),sgd_delta_dtype=str((.001*v.grad.data).dtype)))
        finals[(mode,repeat)]=final
        row=dict(mode=mode,repeat=repeat,steps=count,samples=count*4,wall_s=elapsed,samples_per_s=count*4/elapsed,loss=value,final_hash=state_hash(final))
        rows.append(row);print(json.dumps(row),flush=True)
if not a.trace:
    for r in rows:
        own=finals[(r['mode'],r['repeat'])];ref=finals[('torch',r['repeat'])]
        diff=np.concatenate([(own[k]-ref[k]).ravel() for k in own]);den=np.concatenate([ref[k].ravel() for k in own])
        r.update(weight_max_abs=float(np.abs(diff).max()),weight_relative_l2=float(np.linalg.norm(diff)/np.linalg.norm(den)),weights_allclose=all(np.allclose(own[k],ref[k],atol=2e-5,rtol=2e-4) for k in own),loss_abs_vs_torch=abs(r['loss']-next(x['loss'] for x in rows if x['mode']=='torch' and x['repeat']==r['repeat'])))
    assert all(r['weights_allclose'] for r in rows)
    assert all(r['loss_abs_vs_torch']<=2e-5 for r in rows)
    assert all(x['grad_dtype'] in ('float32','torch.float32') and x['sgd_delta_dtype'] in ('float32','torch.float32') for x in audit if x['mode']!='before_native')
with (out/'training.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(out/'parameter_dtype.json').write_text(json.dumps(audit,indent=2))
(out/'method.json').write_text(json.dumps(dict(seed=42,batch=4,samples=256,epochs=3,warmup=10,repeats=5,initial_hash=state_hash(initial),tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32,cudnn_enabled=torch.backends.cudnn.enabled,cudnn_deterministic=True,cudnn_benchmark=False,torch=torch.__version__,cupy=cp.__version__,lr=.001,dropout=0,wall='includes sequential host data loader, pageable H2D, all computation and final sync; no per-step logging/loss read/sync',nvtx='phase annotations shared by every mode; CPU elapsed is not active CPU scheduling time'),indent=2))
