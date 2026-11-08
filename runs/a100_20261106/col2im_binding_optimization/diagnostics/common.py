"""Diagnostic-only helpers; original CuPy kernel source is read verbatim via AST."""
import ast,inspect,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import numpy as np
import cupy as cp
from hytorch import cuda
from hytorch import functions_conv as FC
from hytorch.backends.cuda import runtime as rt

def cupy_kernel():
    tree=ast.parse(inspect.getsource(FC._col2im_gpu))
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='ElementwiseKernel')
    return eval(compile(ast.Expression(node),'<unchanged-col2im-expression>','eval'),{'cuda':cuda})
ORIGINAL=cupy_kernel()
def run(mode,col,shape,stride,pad,out):
    if mode=='native':return rt.col2im(col,shape,stride,pad,out=out)
    n,c,kh,kw,oh,ow=col.shape;h,w=shape[2:];sh,sw=stride;ph,pw=pad
    ORIGINAL(col.reduced_view(),h,w,oh,ow,kh,kw,sh,sw,ph,pw,1,1,out)
    return out

def cases():
    specs=[('conv1',(4,3,66,200),(5,5),(2,2),(0,0)),('conv2',(4,24,31,98),(5,5),(2,2),(0,0)),('conv3',(4,36,14,47),(5,5),(2,2),(0,0)),('conv4',(4,48,5,22),(3,3),(1,1),(0,0)),('conv5',(4,64,3,20),(3,3),(1,1),(0,0)),('overlap',(2,7,31,37),(5,5),(1,1),(2,2)),('nonoverlap',(3,5,30,33),(3,3),(3,3),(0,0)),('asymmetric',(2,3,17,23),(3,5),(2,3),(1,2)),('holes',(2,4,19,22),(1,1),(3,3),(0,0)),('narrow',(1,9,7,41),(5,1),(1,2),(2,0)),('tiny',(1,1,3,4),(1,1),(1,1),(0,0)),('large_channels',(2,64,32,35),(3,3),(2,2),(1,1))]
    return [dict(name=name,shape=shape,kernel=kernel,stride=stride,pad=pad,layout=layout) for name,shape,kernel,stride,pad in specs for layout in ['training','contiguous','negative']]
def make(case,seed):
    n,c,h,w=case['shape'];kh,kw=case['kernel'];sh,sw=case['stride'];ph,pw=case['pad'];oh=(h+2*ph-kh)//sh+1;ow=(w+2*pw-kw)//sw+1
    host=np.random.default_rng(seed).normal(size=(n,c,kh,kw,oh,ow)).astype(np.float32)
    layout=case['layout']
    if layout=='training':col=cp.asarray(host.transpose(1,2,3,0,4,5).copy()).transpose(3,0,1,2,4,5)
    elif layout=='negative':col=cp.asarray(host[...,::-1].copy())[...,::-1]
    else:col=cp.asarray(host)
    return col
