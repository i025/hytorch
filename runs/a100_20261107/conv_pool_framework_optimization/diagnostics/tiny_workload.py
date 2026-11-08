from common import *
from benchmarks.bench_cuda_training import TinyNet,initialize
m=TinyNet();initialize(m,(3,16,16),42);m.to_gpu();calls=[];original=FC.Pooling.forward
def forward(self,x):
 calls.append(dict(shape=x.shape,dtype=str(x.dtype),strides=x.strides,kernel=self.kernel_size,stride=self.stride,pad=self.pad));return original(self,x)
FC.Pooling.forward=forward
with config('B0'):
 x=hytorch.Variable(cp.zeros((4,3,16,16),cp.float32));y=m(x);y.sum().backward();sync()
assert calls[0]['shape']==(4,4,16,16)
(O/'WORKLOAD_TINY_POOL.json').write_text(json.dumps(calls,indent=2))
