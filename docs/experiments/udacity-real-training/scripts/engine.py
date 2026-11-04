"""Reuse the matched DonkeyNet and SGD implementation without changing kernels."""
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
import hytorch
from hytorch import cuda
import hytorch.functions as F
from hytorch.backends.cuda import backend
from hytorch.optimizers import SGD
from donkeycar_hy.model import DonkeyNet
from benchmarks.bench_cuda_training import initialize,parameters,snapshot,torch_model,torch_snapshot

CONFIGS=dict(A={},F=dict(matmul='shared',conv='im2col',pool='cupy',tile=16,conv_backward='fused'),
             I=dict(matmul='shared',conv='im2col',pool='cupy',tile=16,conv_backward='fused',conv_dweight='fused'),E={})


class Engine:
    def __init__(self,method,device='cuda'):
        self.method,self.device=method,device
        hy=DonkeyNet(dropout_rate=0.);self.initial=initialize(hy,(3,66,200),42)
        self.cp=None
        if device=='cuda':
            import cupy as cp
            cp.cuda.Device(0).use();self.cp=cp
        if method=='E':
            import torch
            self.torch=torch;torch.set_num_threads(4);torch.manual_seed(42)
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
            self.model=torch_model('donkey',self.initial).to('cuda:0' if self.cp is not None else 'cpu')
            self.optimizer=torch.optim.SGD(self.model.parameters(),lr=.001)
        else:
            self.model=hy
            if self.cp is not None:self.model.to_gpu()
            self.optimizer=SGD(.001,fused=method in 'FI').setup(self.model)

    def context(self):return backend(**CONFIGS[self.method])

    def convert(self,value):
        if self.method=='E':return self.torch.from_numpy(value).to('cuda:0' if self.cp is not None else 'cpu')
        return self.cp.asarray(value) if self.cp is not None else value

    def sync(self):
        if self.cp is not None:
            if self.method=='E':self.torch.cuda.synchronize(0)
            else:self.cp.cuda.get_current_stream().synchronize()

    def update(self):
        if self.method=='E':self.optimizer.step()
        else:self.optimizer.update()

    def loss(self,x,t):
        if self.method=='E':return self.torch.nn.functional.mse_loss(self.model(x),t)
        if self.method in 'FI':x=hytorch.Variable(x,requires_grad=False)
        return F.mean_square_error(self.model(x),t)

    def backward(self,loss):
        if self.method=='E':self.optimizer.zero_grad(set_to_none=True)
        else:self.model.cleargrads()
        loss.backward()

    def value(self,loss):
        return float(loss.detach().cpu()) if self.method=='E' else float(cuda.as_numpy(loss.data))

    def state(self):
        return torch_snapshot(self.model,self.initial) if self.method=='E' else snapshot(self.model)

    def gradients(self):
        if self.method!='E':return {k:cuda.as_numpy(v.grad.data).copy() for k,v in parameters(self.model).items()}
        result={}
        for k in self.initial:
            layer,name=k.split('/');v=getattr(getattr(self.model,layer),'weight' if name=='W' else 'bias').grad.detach().cpu().numpy()
            result[k]=v.T.copy() if v.ndim==2 else v.copy()
        return result

    def restore(self):
        if self.method!='E':
            for k,v in parameters(self.model).items():v.data=self.convert(self.initial[k].copy())
            self.model.cleargrads()
        else:
            with self.torch.no_grad():
                for k,v in self.initial.items():
                    layer,name=k.split('/');target=getattr(getattr(self.model,layer),'weight' if name=='W' else 'bias')
                    target.copy_(self.convert(v.T.copy() if v.ndim==2 else v.copy()))
            self.optimizer.zero_grad(set_to_none=True)

    def predict(self,x):
        if self.method=='E':
            with self.torch.no_grad():return self.model(self.convert(x)).detach().cpu().numpy()
        with hytorch.no_grad():return cuda.as_numpy(self.model(self.convert(x)).data)
