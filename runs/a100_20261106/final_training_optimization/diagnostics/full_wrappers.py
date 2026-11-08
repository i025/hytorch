from common import *
from hytorch import Variable,no_grad
from hytorch.optimizers import SGD
from benchmarks.bench_cuda_training import initialize,parameters
from donkeycar_hy.model import DonkeyNet
from types import SimpleNamespace
rows=[];sync=cp.cuda.get_current_stream().synchronize
for spec in specs+holdout:
 for layout in ['contiguous','negative','slice']:
  x,g=arrays(spec,2026,layout);k,s,p=spec[3:];conv=SimpleNamespace(inputs=[None,Variable(cp.empty((g.shape[1],x.shape[1],*k),cp.float32))],stride=s,pad=p);xv,gv=Variable(x),Variable(g)
  def call():return FC.Conv2DGradW(conv)(xv,gv)
  for mode in ['B0','D']:
   with config(mode),no_grad():
    for _ in range(10):call()
  for rep in range(15):
   for mode in (['B0','D'] if rep%2==0 else ['D','B0']):
    with config(mode),no_grad():
     sync();t=time.perf_counter_ns();y=call();cpu=(time.perf_counter_ns()-t)/1000;sync()
     t=time.perf_counter_ns();y=call();sync();wall=(time.perf_counter_ns()-t)/1000
    rows.append(dict(operator='dweight',case=spec[0],layout=layout,mode=mode,repeat=rep,cpu_submit_us=cpu,synchronized_us=wall,scope='full Conv2DGradW Function including fresh construction, validation, packing, allocation, binding and result Variable; backward no_grad'))
model=DonkeyNet(dropout_rate=0.);initialize(model,(3,66,200),42);model.to_gpu();ps=list(parameters(model).values())
for q in ps:q.grad=Variable(cp.full(q.shape,.1,cp.float32))
old=SGD(.001).setup(model);new=SGD(.001,'native').setup(model);production=rt.sgd_multi;rt.sgd_multi=rt._sgd_multi_candidate
try:
 for rep in range(31):
  for mode,opt in ([('legacy',old),('prototype_rejected',new)] if rep%2==0 else [('prototype_rejected',new),('legacy',old)]):
   sync();t=time.perf_counter_ns();opt.update();cpu=(time.perf_counter_ns()-t)/1000;sync();t=time.perf_counter_ns();opt.update();sync();wall=(time.perf_counter_ns()-t)/1000
   rows.append(dict(operator='sgd',case='DonkeyNet',layout='contiguous',mode=mode,repeat=rep,cpu_submit_us=cpu,synchronized_us=wall,scope='full Optimizer.update incl collection/hooks/filter/alias/descriptor; prototype REJECTED'))
finally:rt.sgd_multi=production
write(O/'FULL_WRAPPERS.csv',rows)
