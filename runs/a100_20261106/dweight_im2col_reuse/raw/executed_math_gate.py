import sys,time,csv,json
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import torch
import cupy as cp
import numpy as np
from hytorch import functions_conv as FC
from hytorch.backends.cuda import runtime as rt
O=Path('runs/a100_20261106/dweight_im2col_reuse');rows=[];failures=[]
specs=[('conv1',(4,3,66,200),24,(5,5),(2,2),(0,0)),('conv2',(4,24,31,98),36,(5,5),(2,2),(0,0)),('conv3',(4,36,14,47),48,(5,5),(2,2),(0,0)),('conv4',(4,48,5,22),64,(3,3),(1,1),(0,0)),('conv5',(4,64,3,20),64,(3,3),(1,1),(0,0)),('irregular',(2,3,13,17),7,(3,5),(2,3),(1,2)),('overlap',(3,5,19,23),11,(5,3),(1,1),(2,1)),('point',(1,7,9,11),13,(1,1),(1,1),(0,0)),('holes',(2,2,8,13),5,(1,3),(3,2),(1,2)),('tail',(5,3,7,8),9,(2,4),(2,1),(1,0))]
stream=cp.cuda.Stream(non_blocking=True);stream.use()
for ci,(name,shape,oc,kernel,stride,pad) in enumerate(specs):
 for layout in ['contiguous','negative','slice']:
  for seed in [42,123,501]:
   rng=np.random.default_rng(seed);host=rng.normal(size=shape).astype(np.float32);n,c,h,w=shape;kh,kw=kernel;sh,sw=stride;ph,pw=pad;oh=(h+2*ph-kh)//sh+1;ow=(w+2*pw-kw)//sw+1
   if layout=='contiguous':x=cp.asarray(host)
   elif layout=='negative':x=cp.asarray(host[...,::-1].copy())[...,::-1]
   else:
    storage=cp.empty((*shape[:-1],w*2),cp.float32);x=storage[...,::2];x[...] = cp.asarray(host)
   gh=rng.normal(size=(n,oc,oh,ow)).astype(np.float32);gy=cp.asarray(gh)
   # Actual native forward im2col with its existing packing convention.
   xc=cp.ascontiguousarray(x);col=cp.empty((n*oh*ow,c*kh*kw),cp.float32)
   rt._call('hy_im2col',xc.data.ptr,col.data.ptr,*rt._geometry(xc,kernel,stride,pad))
   six=FC.im2col_array(x,kernel,stride,pad,to_matrix=False)
   expected=six.transpose(0,4,5,1,2,3).reshape(col.shape)
   np.testing.assert_array_equal(cp.asnumpy(col),cp.asnumpy(expected))
   original=cp.tensordot(gy,six,((0,2,3),(0,4,5)))
   G=gy.transpose(1,0,2,3).reshape(oc,-1)
   candidate=(G@col).reshape(original.shape)
   ref=(G.astype(cp.float64)@col.astype(cp.float64)).reshape(original.shape)
   a,b,r=map(cp.asnumpy,[original,candidate,ref]);diff=np.abs(a.astype(np.float64)-b);dr=np.abs(b-r);ar=np.abs(a-r)
   row=dict(case=name,layout=layout,seed=seed,shape=shape,oc=oc,kernel=kernel,stride=stride,pad=pad,T=n*oh*ow,D=c*kh*kw,col_bytes=col.nbytes,col_layout_exact=True,max_abs_B_A=float(diff.max()),max_rel_B_A=float((diff/np.maximum(np.abs(a),1e-30)).max()),max_abs_A_FP64=float(ar.max()),max_abs_B_FP64=float(dr.max()),max_rel_B_FP64=float((dr/np.maximum(np.abs(r),1e-30)).max()),bitwise_equal=np.array_equal(a,b))
   for gate,atol,rtol in [('conv_host',1e-5,1e-4),('donkey',2e-5,2e-4),('small_dtype',2e-6,2e-5)]:
    for label,delta,reference in [('B_A',diff,a),('A_FP64',ar,r),('B_FP64',dr,r)]:
     bad=delta>(atol+rtol*np.abs(reference));row[f'{gate}_{label}_failed']=int(bad.sum())
     if gate=='donkey' and bad.any():
      idx=np.unravel_index(np.argmax(delta/(atol+rtol*np.abs(reference))),delta.shape);failures.append(dict(case=name,layout=layout,seed=seed,comparison=label,index=idx,A=float(a[idx]),B=float(b[idx]),reference64=float(r[idx]),ratio=float((delta/(atol+rtol*np.abs(reference)))[idx])))
   rows.append(row)
 print(name,'donkey failures',sum(r['donkey_B_A_failed'] for r in rows if r['case']==name),'maxabs',max(r['max_abs_B_A'] for r in rows if r['case']==name),flush=True)
with (O/'raw/math_gate.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(O/'raw/failure_examples.json').write_text(json.dumps(failures,indent=2,default=lambda v:int(v)))
print('DONE',len(rows),flush=True)
