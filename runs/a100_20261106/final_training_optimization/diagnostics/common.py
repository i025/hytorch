from pathlib import Path
import sys,csv,json,contextlib,time,ast
sys.path.insert(0,str(Path.cwd()))
import torch
import cupy as cp
import numpy as np
from hytorch.backends.cuda import runtime as rt
from hytorch import functions_conv as FC
O=Path.cwd()/'runs/a100_20261106/final_training_optimization'
MODES=['B0','D','S','DS']
@contextlib.contextmanager
def config(mode):
 with rt.backend(matmul='cupy' if mode=='cupy' else 'shared',conv='cupy' if mode=='cupy' else 'im2col',pool='cupy' if mode=='cupy' else 'native',col2im='cupy' if mode=='cupy' else 'native',dweight='packed' if mode in ['D','DS'] else 'cupy'):yield

def write(path,rows):
 if not rows:return
 with Path(path).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def original(x,g,k,s,p):return cp.tensordot(g,FC.im2col_array(x,k,s,p,False),((0,2,3),(0,4,5)))
def metrics(a,b,atol,rtol):
 raw_a,raw_b=np.asarray(a),np.asarray(b)
 bitwise=float(np.mean(raw_a.view(np.uint32)==raw_b.view(np.uint32))) if raw_a.dtype==raw_b.dtype==np.float32 else ""
 a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64);d=np.abs(a-b);bad=d>atol+rtol*np.abs(b)
 return dict(max_abs=float(d.max(initial=0)),rel_l2=float(np.linalg.norm(d)/max(np.linalg.norm(b),1e-30)),failed=int(bad.sum()),bitwise_fraction=bitwise,exact_value_fraction=float(np.mean(a==b)),coordinates=json.dumps(np.argwhere(bad).tolist()))
specs=[('conv1',(4,3,66,200),24,(5,5),(2,2),(0,0)),('conv2',(4,24,31,98),36,(5,5),(2,2),(0,0)),('conv3',(4,36,14,47),48,(5,5),(2,2),(0,0)),('conv4',(4,48,5,22),64,(3,3),(1,1),(0,0)),('conv5',(4,64,3,20),64,(3,3),(1,1),(0,0)),('irregular',(2,3,13,17),7,(3,5),(2,3),(1,2)),('overlap',(3,5,19,23),11,(5,3),(1,1),(2,1)),('point',(1,7,9,11),13,(1,1),(1,1),(0,0)),('holes',(2,2,8,13),5,(1,3),(3,2),(1,2)),('tail',(5,3,7,8),9,(2,4),(2,1),(1,0))]
holdout=[('hold1',(2,4,21,29),13,(2,5),(3,2),(1,2)),('hold2',(1,9,12,19),17,(4,2),(2,3),(2,1)),('hold3',(3,2,15,11),6,(3,1),(1,2),(1,0))]
def arrays(spec,seed,layout='contiguous'):
 name,shape,oc,k,s,p=spec;n,c,h,w=shape;kh,kw=k;sh,sw=s;ph,pw=p;oh=(h+2*ph-kh)//sh+1;ow=(w+2*pw-kw)//sw+1;rng=np.random.default_rng(seed)
 x=rng.normal(size=shape).astype(np.float32);g=rng.normal(size=(n,oc,oh,ow)).astype(np.float32)
 def gpu(v):
  if layout=='negative':return cp.asarray(v[...,::-1].copy())[...,::-1]
  if layout=='slice':
   a=cp.empty((*v.shape[:-1],v.shape[-1]*2),cp.float32);a[...,::2]=cp.asarray(v);return a[...,::2]
  return cp.asarray(v)
 return gpu(x),gpu(g)
def validate(x,g,k,s,p,case,rows):
 a=cp.asnumpy(original(x,g,k,s,p));b=cp.asnumpy(rt.packed_dweight(x,g,k,s,p));xh,gh=cp.asnumpy(x),cp.asnumpy(g)
 col=FC.im2col_array(xh.astype(np.float64),k,s,p,True);gm=gh.astype(np.float64).transpose(1,0,2,3).reshape(g.shape[1],-1);r=(gm@col).reshape(a.shape)
 for label,u,v in [('A_R',a,r),('B_R',b,r),('B_A',b,a)]:
  for gate,atol,rtol in [('primary',2e-5,2e-4),('host_strict',1e-5,1e-4),('dtype_strict',2e-6,2e-5)]:rows.append(dict(case=case,comparison=label,gate=gate,**metrics(u,v,atol,rtol)))
 # Scalar longdouble dot, at maximum B/R absolute error, independent CPU expansion.
 idx=np.unravel_index(np.argmax(np.abs(b-r)),b.shape);oc,ic,ky,kx=idx;d=(ic*k[0]+ky)*k[1]+kx
 ld=np.sum(gm[oc].astype(np.longdouble)*col[:,d].astype(np.longdouble),dtype=np.longdouble)
 rows.append(dict(case=case,comparison='FP64_longdouble',gate='reference_check',max_abs=float(abs(ld-r[idx])),rel_l2='',failed=int(abs(ld-r[idx])>1e-9),bitwise_fraction='',coordinates=str(idx)))
 return not np.any(np.abs(b-r)>2e-5+2e-4*np.abs(r))
