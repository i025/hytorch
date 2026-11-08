"""Opt-in Function container reduction must preserve the graph and every bit."""
import gc,weakref
import numpy as np
import pytest
import hytorch
from hytorch import Variable,Function
from hytorch.core import Config,using_config
from hytorch import functions as F
from hytorch.backends.cuda import runtime as rt
from test_col2im_cuda import cp

def bits(a,b):
 assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()

def test_default_nested_exception():
 assert Config.compact_function is False
 with using_config('compact_function',True):
  with pytest.raises(RuntimeError):
   with using_config('compact_function',False):raise RuntimeError()
  assert Config.compact_function is True
 assert Config.compact_function is False

@pytest.mark.parametrize('value',[np.array([-0.,0.,np.nan]),np.array(3.),np.float32(2),3.,3,None,(np.array(1),),(np.array(1),np.array(2)),[1,2]])
@pytest.mark.parametrize('grad',[False,True])
def test_output_contract(value,grad):
 class Fixed(Function):
  def forward(self,x):return value
 results=[]
 for mode in [False,True]:
  x=Variable(np.array(1.));fn=Fixed()
  with using_config('compact_function',mode),using_config('enable_backprop',grad):
   try:y=fn(x)
   except Exception as exc:results.append(type(exc));continue
  outputs=y if isinstance(y,list) else [y]
  assert fn.inputs==[x] and all(r() is o for r,o in zip(fn.outputs,outputs))
  assert all(o.creator is (fn if grad else None) for o in outputs)
  results.append([(o.data,o.generation) for o in outputs])
 if isinstance(results[0],type):assert results[0]==results[1]
 else:
  for (a,ga),(b,gb) in zip(*results):
   assert ga==gb
   if a is None:assert b is None
   else:bits(a,b)

@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('gpu',[False,True])
def test_conv_graph_mutation_layouts(cp,dtype,gpu):
 xp=cp if gpu else np
 def host(x):return cp.asnumpy(x) if gpu else x.copy()
 def run(mode,batch,layout):
  rng=np.random.default_rng(123)
  x=Variable(xp.asarray(rng.normal(0,.1,(batch,2,7,9)).astype(dtype)))
  if layout=='negative':x.data=x.data[:,:,:,::-1]
  if layout=='transpose':x.data=x.data.transpose(0,1,3,2)
  if layout=='zero':x.data=xp.broadcast_to(x.data[:1],(batch,*x.shape[1:]))
  w=Variable(xp.asarray(rng.normal(0,.1,(3,2,3,2)).astype(dtype)));b=Variable(xp.zeros(3,dtype=dtype))
  with using_config('compact_function',mode),rt.backend(conv='im2col',matmul='shared',pool='native',col2im='native',dweight='packed',metadata_cache=True):
   y=F.conv2d(x,w,b,stride=(2,1),pad=(1,0));loss=(y*y).sum();loss.backward(create_graph=True)
   out=[host(z.grad.data) for z in [x,w,b]];gx=x.grad;x.cleargrad();gx.sum().backward();out.append(host(x.grad.data))
   loss.backward();out += [host(z.grad.data) for z in [x,w,b]]
   # Mutate after graph creation, preserving baseline's existing semantics.
   if layout!='zero':x.data[...] *= dtype(.7)
   loss.backward();out += [host(z.grad.data) for z in [x,w,b]]
   with hytorch.no_grad():z=F.conv2d(x,w,b,stride=(2,1),pad=(1,0));assert z.creator is None
   out.append(host(z.data))
   from hytorch.optimizers import SGD
   class Target:
    def params(self):return iter([w,b])
   SGD(.001).setup(Target()).update();out += [host(w.data),host(b.data)]
  return out
 for batch in [1,3,2,1]:
  for layout in ['contiguous','negative','transpose','zero']:
   for a,b in zip(run(False,batch,layout),run(True,batch,layout)):bits(a,b)

@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('gpu',[False,True])
def test_pool_boundary_repeat_higher_grad(cp,dtype,gpu):
 xp=cp if gpu else np
 def run(mode,shape,k,s,p):
  xa=(np.arange(np.prod(shape)).reshape(shape)%17-20).astype(dtype)
  x=Variable(xp.asarray(xa)[:,:,:,::-1])
  with using_config('compact_function',mode),rt.backend(pool='native',col2im='native',metadata_cache=True):
   y=F.pooling(x,k,s,p);loss=(y*y).sum();loss.backward(create_graph=True);gx=x.grad
   out=[y.data.copy(),gx.data.copy()];x.cleargrad();gx.sum().backward();out.append(x.grad.data.copy());loss.backward();out.append(x.grad.data.copy())
  return [cp.asnumpy(a) if gpu else a for a in out]
 for shape,k,s,p in [((1,2,5,7),(3,2),(2,1),(1,1)),((3,1,7,9),2,2,0),((2,3,3,5),3,1,2)]:
  for a,b in zip(run(False,shape,k,s,p),run(True,shape,k,s,p)):bits(a,b)

def test_current_stream_lifetime_and_alias(cp):
 refs=[];stream=cp.cuda.Stream(non_blocking=True)
 with stream,using_config('compact_function',True),rt.backend(pool='native',metadata_cache=True):
  for n in [1,3,2,1]:
   x=Variable(cp.arange(n*63,dtype=cp.float32).reshape(n,1,7,9));refs.append(weakref.ref(x.data))
   y=F.pooling(x,2,2);y.sum().backward();assert x.grad is not None
  stream.synchronize();a=cp.ones((4,4),cp.float32)
  with pytest.raises(ValueError):rt.gemm(a,a,out=a)
  with pytest.raises(TypeError):rt.max_pool2d(a.astype(cp.float64),2)
  del x,y;gc.collect();assert all(r() is None for r in refs)
