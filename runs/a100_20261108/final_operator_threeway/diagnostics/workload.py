from common import *
from benchmarks.bench_cuda_training import SyntheticDataset,initialize,parameters,TinyNet
from donkeycar_hy.model import DonkeyNet
from hytorch.optimizers import SGD
records=[];gemms=[];cols=[];stack=[]
def meta(x):
 if isinstance(x,hy.Variable):x=x.data
 return dict(shape=list(x.shape),strides_bytes=list(x.strides),dtype=str(x.dtype)) if isinstance(x,(np.ndarray,cp.ndarray)) else None
orig=hy.core.Function.__call__;og=rt.gemm;odw=rt.packed_dweight;otd=cp.tensordot;oc=FC.col2im_array

def call(self,*inputs):
 name=type(self).__name__;stack.append(name)
 if name=='MatMul':
  a,b=[meta(v) for v in inputs];gemms.append(dict(M=a['shape'][0],N=b['shape'][1],K=a['shape'][1],path='MatMul_native' if self.native else 'MatMul_cupy_backward',a=a,b=b))
 try:out=orig(self,*inputs)
 finally:stack.pop()
 records.append(dict(function=name,inputs=[meta(v) for v in inputs],outputs=[meta(v) for v in out] if isinstance(out,list) else [meta(out)],stride=getattr(self,'stride',None),pad=getattr(self,'pad',None)))
 return out

def gem(a,b,*args,**kw):
 if not stack or stack[-1]!='MatMul':gemms.append(dict(M=a.shape[0],N=b.shape[1],K=a.shape[1],path='native_gemm:'+','.join(stack),a=meta(a),b=meta(b)))
 return og(a,b,*args,**kw)
def dw(x,g,k,s,p):
 gemms.append(dict(M=g.shape[1],N=x.shape[1]*k[0]*k[1],K=g.shape[0]*g.shape[2]*g.shape[3],path='packed_dweight_cublas',x=meta(x),gy=meta(g)))
 return odw(x,g,k,s,p)
def td(a,b,axes=2):
 if isinstance(axes,tuple) and axes==(0,1) and a.ndim==4 and b.ndim==4:gemms.append(dict(M=int(np.prod(a.shape[1:])),N=int(b.shape[0]*b.shape[2]*b.shape[3]),K=a.shape[0],path='dInput_tensordot_cublas',a=meta(a),b=meta(b)))
 return otd(a,b,axes)
def col(a,img,k,s,p,to_matrix=True):
 cols.append(dict(input=meta(a),img_shape=list(img),kernel=k,stride=s,pad=p,to_matrix=to_matrix));return oc(a,img,k,s,p,to_matrix)
model=DonkeyNet(dropout_rate=0);initialize(model,(3,66,200),42);model.to_gpu();ds=SyntheticDataset(4,(3,66,200),42)
hy.core.Function.__call__=call;rt.gemm=gem;rt.packed_dweight=dw;cp.tensordot=td;FC.col2im_array=col
with config('native'):
 x=cp.asarray(ds.x);t=cp.asarray(ds.t);loss=F.mean_square_error(model(x),t);model.cleargrads();loss.backward();SGD(.001,backend='legacy').setup(model).update();cp.cuda.get_current_stream().synchronize()
hy.core.Function.__call__=orig;rt.gemm=og;rt.packed_dweight=odw;cp.tensordot=otd;FC.col2im_array=oc
(O/'raw/workload.json').write_text(json.dumps(dict(functions=records,gemms=gemms,col2im=cols,loss=float(loss.data)),indent=2))
write('raw/real_gemm_calls.csv',gemms);write('raw/real_col2im_calls.csv',cols)
# Verify benchmark's real geometries against fresh function records.
conv=[v for v in records if v['function']=='Conv2d']
assert len(conv)==5
for row,s in zip(conv,specs()[:5]):
 assert row['inputs'][0]['shape']==list(s['x']) and row['inputs'][1]['shape']==[s['oc'],s['x'][1],*s['k']]
 assert list(row['stride'])==list(s['s']) and list(row['pad'])==list(s['p'])
for row in cols:
 match=next(s for s in specs()[:5] if list(s['x'])==row['img_shape']);n,c,h,w,kh,kw,sh,sw,ph,pw,oh,ow=dims(match)
 z=layout(np.empty((n,c,kh,kw,oh,ow),'f'),np,'training');assert list(z.strides)==row['input']['strides_bytes']
print('Fresh step: ',len(gemms),'GEMM calls;',len(cols),'col2im calls. All five col strides match training layout.')
(O/'raw/cases.json').write_text(json.dumps(cases(),indent=2));write('WORKLOAD.csv',[dict(c, spec=json.dumps(c.get('spec',{})),source='fresh workload validated' if c['origin']=='real' else c['origin']) for c in cases()])
