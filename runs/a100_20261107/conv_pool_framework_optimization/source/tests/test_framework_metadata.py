"""Value-only validation cache: preserve numeric, graph and current-device contracts."""
import gc,weakref
import numpy as np
import pytest
import hytorch
from hytorch import Variable
from hytorch import functions as F
from hytorch.backends.cuda import runtime as rt
from test_col2im_cuda import cp

def test_metadata_default_nested_exception():
 assert not rt.current_config().get('metadata_cache',False)
 with rt.backend(metadata_cache=True):
  assert rt.current_config()['metadata_cache']
  with pytest.raises(RuntimeError):
   with rt.backend(metadata_cache=False):raise RuntimeError('restore')
  assert rt.current_config()['metadata_cache']
 assert not rt.current_config().get('metadata_cache',False)
 with pytest.raises(ValueError):
  with rt.backend(metadata_cache=1):pass

@pytest.mark.parametrize('shape',[(1,),(2,3),(0,3),(-1,4),(65536,65536),(2**31-1,),(2**31,),(),(np.int64(2),3),(2.,3),(True,3),[2,3]])
def test_shape_validation_equivalence(shape):
 def run(flag):
  with rt.backend(metadata_cache=flag):
   try:return ('value',rt._shape_ok(shape))
   except Exception as exc:return(type(exc),str(exc))
 assert run(False)==run(True)==run(True)

def test_value_cache_capacity_no_tensor_refs():
 rt._shape_ok_cached.cache_clear()
 with rt.backend(metadata_cache=True):
  rt._shape_ok((2,3));rt._shape_ok((2,3))
  assert rt._shape_ok_cached.cache_info().hits==1
  # Same equality with different scalar type must not hit integer entries.
  before=rt._shape_ok_cached.cache_info()
  rt._shape_ok((2.,3));rt._shape_ok((np.int64(2),3));rt._shape_ok([2,3])
  assert rt._shape_ok_cached.cache_info()==before
  for i in range(1,600):rt._shape_ok((i,3))
 assert rt._shape_ok_cached.cache_info().currsize==256

@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('gpu',[False,True])
def test_graph_repeats_changes_mutation_higher_grad(cp,dtype,gpu):
 xp=cp if gpu else np
 def host(z):return cp.asnumpy(z) if gpu else z
 def experiment(enabled,batch,noncontig):
  rng=np.random.default_rng(37);xa=rng.normal(0,.1,(batch,2,7,9)).astype(dtype);wa=rng.normal(0,.1,(3,2,3,2)).astype(dtype)
  x=Variable(xp.asarray(xa));w=Variable(xp.asarray(wa));b=Variable(xp.zeros(3,dtype=dtype))
  if noncontig:x.data=x.data[:,:,:,::-1]
  with rt.backend(matmul='shared',conv='im2col',pool='native',col2im='native',dweight='packed',metadata_cache=enabled):
   y=F.conv2d(x,w,b,stride=(2,1),pad=(1,0));loss=(y*y).sum()
   loss.backward(create_graph=True);gx=x.grad
   first=[host(v.grad.data).copy() for v in [x,w,b]]
   x.cleargrad();gx.sum().backward();second=host(x.grad.data).copy()
   # Original graph can be backwarded again; accumulate parameter gradients.
   loss.backward();accum=[host(v.grad.data).copy() for v in [x,w,b]]
   x.data[...] *=dtype(.5)
   with hytorch.no_grad():z=F.conv2d(x,w,b,stride=(2,1),pad=(1,0));assert z.creator is None
   mutated=host(z.data).copy()
   from hytorch.optimizers import SGD,WeightDecay
   class Target:
    def params(self):return iter([w,b])
   opt=SGD(.001).setup(Target());opt.add_hook(WeightDecay(.01));opt.update()
   updated=[host(v.data).copy() for v in [w,b]]
  return [*first,second,*accum,mutated,*updated]
 for batch in [1,3,2,1]:
  for nc in [False,True]:
   for a,b in zip(experiment(False,batch,nc),experiment(True,batch,nc)):np.testing.assert_array_equal(a,b)

def test_stream_address_dtype_and_lifetime(cp):
 rt._shape_ok_cached.cache_clear();stream=cp.cuda.Stream(non_blocking=True)
 with stream,rt.backend(metadata_cache=True):
  for i in range(4):
   a=cp.arange(35,dtype=cp.float32).reshape(5,7)+i;b=cp.ones((7,5),cp.float32);out=cp.empty((5,5),cp.float32)
   rt.gemm(a,b,out=out);expected=a@b;stream.synchronize();np.testing.assert_array_equal(cp.asnumpy(out),cp.asnumpy(expected))
  # Hot cache cannot admit a bad dtype or aliased output.
  with pytest.raises(TypeError):rt.gemm(a.astype(cp.float64),b)
  with pytest.raises(ValueError):rt.gemm(out,cp.ones((5,5),cp.float32),out=out)
  ref=weakref.ref(a);del a;gc.collect();assert ref() is None
 assert rt._shape_ok_cached.cache_info().hits>0
