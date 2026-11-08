"""Standalone benchmark adapters only. No production module is edited."""
import os
from types import SimpleNamespace
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='4'
os.environ['NVIDIA_TF32_OVERRIDE']='0'
from pathlib import Path
import sys,json,csv,time,contextlib,hashlib,itertools,gc
R=Path(__file__).resolve().parents[4];sys.path.insert(0,str(R));O=Path(__file__).resolve().parents[1]
import torch
import numpy as np
import cupy as cp
import hytorch as hy
import hytorch.functions as F
from hytorch import functions_conv as FC
from hytorch.backends.cuda import runtime as rt
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
SEED=20261108

def write(name,rows):
 if not rows:return
 p=O/name;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def read(name):return list(csv.DictReader((O/name).open()))
def host(x):
 if isinstance(x,hy.Variable):x=x.data
 return cp.asnumpy(x) if isinstance(x,cp.ndarray) else x
@contextlib.contextmanager
def config(mode):
 native=mode.startswith('native') or mode in ('shared_old','register_old')
 with rt.backend(matmul='shared' if native else 'cupy',conv='im2col' if native else 'cupy',pool='native' if native else 'cupy',col2im='native' if native else 'cupy',dweight='packed' if native else 'cupy',metadata_cache=True),hy.core.using_config('compact_function',True):yield

def specs():
 base=[('conv1',(4,3,66,200),24,(5,5),(2,2),(0,0)),('conv2',(4,24,31,98),36,(5,5),(2,2),(0,0)),('conv3',(4,36,14,47),48,(5,5),(2,2),(0,0)),('conv4',(4,48,5,22),64,(3,3),(1,1),(0,0)),('conv5',(4,64,3,20),64,(3,3),(1,1),(0,0)),('overlap',(2,7,31,37),13,(5,5),(1,1),(2,2)),('nonoverlap',(3,5,30,33),11,(3,3),(3,3),(0,0)),('asymmetric',(2,3,17,23),7,(3,5),(2,3),(1,2)),('holes',(2,4,19,22),9,(1,1),(3,3),(0,0)),('narrow',(1,9,7,41),5,(5,1),(1,2),(2,0)),('tiny',(1,1,3,4),2,(1,1),(1,1),(0,0)),('large_channels',(2,64,32,35),96,(3,3),(2,2),(1,1))]
 return [dict(id=i,x=tuple(x),oc=oc,k=tuple(k),s=tuple(s),p=tuple(p)) for i,x,oc,k,s,p in base]
def dims(s):
 for key in ("x","k","s","p"):s[key]=tuple(s[key])
 n,c,h,w=s['x'];kh,kw=s['k'];sh,sw=s['s'];ph,pw=s['p'];return n,c,h,w,kh,kw,sh,sw,ph,pw,(h+2*ph-kh)//sh+1,(w+2*pw-kw)//sw+1

def window(x,s):
 """Independent vectorized CPU reference, logical N,C,KH,KW,OH,OW."""
 n,c,h,w,kh,kw,sh,sw,ph,pw,oh,ow=dims(s)
 pad=np.pad(x,((0,0),(0,0),(ph,ph),(pw,pw)))
 z=np.lib.stride_tricks.sliding_window_view(pad,(kh,kw),axis=(2,3))[:,:,::sh,::sw,:,:]
 return z[:,:,:oh,:ow].transpose(0,1,4,5,2,3)
def matrix(x,s):
 n,c,h,w,kh,kw,sh,sw,ph,pw,oh,ow=dims(s)
 return window(x,s).transpose(0,4,5,1,2,3).reshape(n*oh*ow,c*kh*kw)
def fold_ref(col,s):
 """Independent FP64 PyTorch fold; asymmetric padding handled by explicit crop."""
 n,c,h,w,kh,kw,sh,sw,ph,pw,oh,ow=dims(s)
 a=np.ascontiguousarray(col.reshape(n,c*kh*kw,oh*ow),dtype=np.float64)
 out=torch.nn.functional.fold(torch.from_numpy(a),(h+2*ph,w+2*pw),(kh,kw),stride=(sh,sw)).numpy()
 return out[:,:,ph:ph+h,pw:pw+w]
def cpu_conv(x,w,b,s):
 n,c,h,wi,kh,kw,sh,sw,ph,pw,oh,ow=dims(s)
 y=matrix(x,s)@w.reshape(s['oc'],-1).T
 if b is not None:y+=b
 return y.reshape(n,oh,ow,s['oc']).transpose(0,3,1,2)
def cpu_dw(x,gy,s):return (gy.transpose(1,0,2,3).reshape(s['oc'],-1)@matrix(x,s)).reshape(s['oc'],s['x'][1],*s['k'])
def native_im(x,s):
 # Adapter to existing ABI; retains per-call dtype/device/geometry checks.
 x,=rt._arrays(x);g=rt._geometry(x,s['k'],s['s'],s['p']);n,c,h,w,kh,kw,sh,sw,ph,pw,oh,ow=g
 shape=(n*oh*ow,c*kh*kw);rt._shape_ok(shape);out=cp.empty(shape,cp.float32)
 rt._call('hy_im2col',rt._ptr(x),rt._ptr(out),*g);return out
class Adapter(hy.core.Function):
 """Diagnostic Function shell for APIs lacking this exact production operation."""
 def __init__(self,fn):self.fn=fn
 def forward(self,*xs):return self.fn(*xs)

def layout(a,xp,kind):
 if kind=='training':return a.transpose(1,2,3,0,4,5).copy().transpose(3,0,1,2,4,5)
 if kind=='negative':return a[...,::-1].copy()[...,::-1]
 if kind=='sliced':
  b=xp.empty((*a.shape[:-1],a.shape[-1]*2),dtype=a.dtype);b[...,::2]=a;return b[...,::2]
 return xp.ascontiguousarray(a)

def cases():
 out=[]
 actual=json.loads((O/'raw/workload.json').read_text()) if (O/'raw/workload.json').exists() else {'gemms':[]}
 seen=set()
 for g in actual['gemms']:
  m,n,k=g['M'],g['N'],g['K']
  if (m,n,k) not in seen:out.append(dict(op='gemm',id=f'real_{m}_{n}_{k}',mnk=[m,n,k],origin='instrumented',layout='contiguous'));seen.add((m,n,k))
 for m,n,k in [(v,v,v) for v in (256,512,1024,2048,4096)]+[(63,65,67),(65,63,129),(1025,127,259),(8192,16,75),(8,4096,257),(511,513,129),(1024,512,256)]:
  if (m,n,k) not in seen:out.append(dict(op='gemm',id=f'ext_{m}_{n}_{k}',mnk=[m,n,k],origin='extension',layout='contiguous'))
 for s in specs():
  for op in ('im2col','conv','dweight'):
   for l in (['contiguous','sliced'] if s['id'] in ('asymmetric','tiny') else ['contiguous']):out.append(dict(op=op,id=s['id']+'_'+l,spec=s,layout=l,origin='real' if s['id'].startswith('conv') else 'extension'))
  for l in ('training','contiguous','negative'):out.append(dict(op='col2im',id=s['id']+'_'+l,spec=s,layout=l,origin='real' if s['id'].startswith('conv') else 'extension'))
 for s in [dict(id='tiny_net_pool',x=[4,4,16,16],oc=4,k=[2,2],s=[2,2],p=[0,0]),dict(id='overlap_pool',x=[2,7,31,37],oc=7,k=[3,3],s=[1,1],p=[1,1]),dict(id='asym_pool',x=[1,3,11,17],oc=3,k=[3,5],s=[2,3],p=[1,2]),dict(id='negative_pool',x=[1,2,7,9],oc=2,k=[3,3],s=[1,1],p=[1,1])]:
  for op in ('pool_forward','pool_backward'):out.append(dict(op=op,id=s['id'],spec=s,layout='contiguous',origin='tiny_net' if s['id']=='tiny_net_pool' else 'extension'))
 return out

def prepare(case,gpu=True):
 seed=SEED+int(hashlib.sha256((case['op']+case['id']).encode()).hexdigest()[:8],16);rng=np.random.default_rng(seed);op=case['op'];s=case.get('spec');d={}
 if op=='gemm':
  m,n,k=case['mnk'];d['a']=rng.normal(0,.25,(m,k)).astype('f');d['b']=rng.normal(0,.25,(k,n)).astype('f');d['out']=np.empty((m,n),'f');ref=d['a'].astype('d')@d['b'].astype('d')
 else:
  n,c,h,w,kh,kw,sh,sw,ph,pw,oh,ow=dims(s);x=rng.normal(0,.25,s['x']).astype('f')
  if s['id']=='negative_pool':x=-np.abs(x);x[...,2:4,2:4]=-1 # tied negative maxima and zero padding
  d['x']=layout(x,np,case['layout'] if op!='col2im' else 'contiguous')
  if op=='col2im':
   col=rng.normal(0,.25,(n,c,kh,kw,oh,ow)).astype('f');d['col']=layout(col,np,case['layout']);ref=fold_ref(d['col'],s)
  elif op=='im2col':ref=matrix(d['x'].astype('d'),s)
  elif op in ('conv','dweight'):
   d['w']=rng.normal(0,.25,(s['oc'],c,kh,kw)).astype('f');d['b']=rng.normal(0,.25,(s['oc'],)).astype('f');d['gy']=rng.normal(0,.25,(n,s['oc'],oh,ow)).astype('f')
   ref=cpu_conv(d['x'].astype('d'),d['w'].astype('d'),d['b'].astype('d'),s) if op=='conv' else cpu_dw(d['x'].astype('d'),d['gy'].astype('d'),s)
  else:
   z=window(d['x'],s).reshape(n,c,kh*kw,oh,ow);d['index']=z.argmax(2);d['gy']=rng.normal(0,.25,(n,c,oh,ow)).astype('f')
   if op=='pool_forward':ref=z.max(2).astype('d')
   else:
    z=np.zeros(z.shape,'d');np.put_along_axis(z,d['index'][:,:,None],d['gy'][:,:,None].astype('d'),axis=2);ref=fold_ref(z.reshape(n,c,kh,kw,oh,ow),s)
 dev={}
 if gpu:
  for k,v in d.items():
   kind=case['layout'] if (k=='col' or (k=='x' and op!='col2im')) else 'contiguous';dev[k]=layout(cp.asarray(np.ascontiguousarray(v)),cp,kind)
 return d,dev,ref

def dwfunc(s):
 f=FC.Conv2DGradW.__new__(FC.Conv2DGradW);f.kernel_size=tuple(s['k']);f.stride=tuple(s['s']);f.pad=tuple(s['p']);return f

def poolgrad(s,d):
 f=FC.Pooling2DGrad.__new__(FC.Pooling2DGrad);f.kernel_size=tuple(s['k']);f.stride=tuple(s['s']);f.pad=tuple(s['p']);f.input_shape=tuple(s['x']);f.dtype=d['x'].dtype;f.indexes=d['index'];return f

def make_fn(case,mode,scope,d):
 op=case['op'];s=case.get('spec');cpu=mode=='numpy';xp=np if cpu else cp;full=scope=='function';native=mode in ('native','shared_old','register_old');note='production'
 if op=='gemm':
  a,b=d['a'],d['b'];out=d['out']
  if full:
   if cpu:fn=lambda:F.matmul(a,b)
   else:fn=lambda:F.matmul(a,b)
  else:fn=(lambda:rt.gemm(a,b,out=out)) if native else (lambda:xp.matmul(a,b,out=out))
 elif op=='im2col':
  x=d['x'];arr=(lambda x:matrix(x,s)) if cpu else ((lambda x:native_im(x,s)) if native else (lambda x:FC.im2col_array(x,s['k'],s['s'],s['p'],True)))
  if full:
   if native or cpu:fn=lambda:Adapter(arr)(x);note='diagnostic Function adapter'
   else:fn=lambda:FC.im2col(x,s['k'],s['s'],s['p'],True)
  else:fn=lambda:arr(x);note='diagnostic ABI adapter' if native else ('NumPy BLAS reference' if cpu else 'production')
 elif op=='col2im':
  x=d['col'];fn=(lambda:FC.col2im(x,s['x'],s['k'],s['s'],s['p'],False)) if full else (lambda:FC.col2im_array(x,s['x'],s['k'],s['s'],s['p'],False))
 elif op=='conv':
  x,w,b=d['x'],d['w'],d['b']
  if cpu:
   fn=(lambda:Adapter(lambda x,w,b:cpu_conv(x,w,b,s))(x,w,b)) if full else (lambda:cpu_conv(x,w,b,s));note='diagnostic NumPy BLAS adapter'
  elif full:fn=lambda:FC.conv2d(x,w,b,s['s'],s['p'])
  elif native:fn=lambda:rt.conv2d(x,w,b,s['s'],s['p'])
  else:fn=lambda:FC.conv2d_cpu_tensordot_array(x,w,b,s['s'],s['p'],xp=cp)
 elif op=='dweight':
  x,gy=d['x'],d['gy']
  if cpu:
   fn=(lambda:Adapter(lambda x,gy:cpu_dw(x,gy,s))(x,gy)) if full else (lambda:cpu_dw(x,gy,s));note='diagnostic NumPy BLAS adapter'
  elif full:
   parent=SimpleNamespace(inputs=(None,hy.Variable(d['w'])),stride=tuple(s['s']),pad=tuple(s['p']))
   fn=lambda:FC.Conv2DGradW(parent)(x,gy)
  elif native:fn=lambda:rt.packed_dweight(x,gy,s['k'],s['s'],s['p'])
  else:fn=lambda:cp.tensordot(gy,FC.im2col_array(x,s['k'],s['s'],s['p'],False),((0,2,3),(0,4,5)))
 elif op=='pool_forward':
  x=d['x'];fn=(lambda:FC.pooling(x,s['k'],s['s'],s['p'])) if full else (lambda:FC.Pooling(s['k'],s['s'],s['p']).forward(x))
 else:
  gy=d['gy'];parent=SimpleNamespace(kernel_size=tuple(s['k']),stride=tuple(s['s']),pad=tuple(s['p']),inputs=(hy.Variable(d['x']),),indexes=d['index'])
  fn=(lambda:FC.Pooling2DGrad(parent)(gy)) if full else (lambda:poolgrad(s,d).forward(gy))
 return fn,note

def accuracy(a,ref):
 a=host(a).astype('d');delta=np.abs(a-ref);floor=1e-8;atol=5e-5;rtol=2e-5
 return dict(max_abs=float(delta.max(initial=0)),relative_l2=float(np.linalg.norm(delta.ravel())/max(np.linalg.norm(ref.ravel()),1e-30)),mixed_fail_count=int(np.count_nonzero(delta>atol+rtol*np.abs(ref))),max_relative_floor_1e_8=float((delta/np.maximum(np.abs(ref),floor)).max(initial=0)),atol=atol,rtol=rtol,reference='independent CPU FP64',elements=ref.size)
