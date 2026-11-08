import torch,cupy as cp,numpy as np,json,os
x=np.array([0.,-0.,1e-45,-1e-45,1e-38,-1e-38,1e-37,-1e-37],np.float32)
rows=[]
for lr in [0.,-0.,.001,1.,.5]:
 for gv in [0.,1e-45,1e-38]:
  p=cp.asarray(x);g=cp.full(x.shape,gv,cp.float32);delta=lr*g;y=p-delta
  rows.append(dict(lr=lr,g=gv,p=x.view(np.uint32).tolist(),delta=cp.asnumpy(delta).view(np.uint32).tolist(),result=cp.asnumpy(y).view(np.uint32).tolist(),cpu=(x-np.float32(lr)*np.full(x.shape,gv,np.float32)).view(np.uint32).tolist()))
print(json.dumps(rows,indent=2))
