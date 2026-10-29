"""Small separate profiler capture; never included in benchmark timings."""
import argparse
import cupy as cp
import torch
from common import setup, shared_call, inputs

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--library',required=True)
p.add_argument('--method',choices=('all','shared32','block32_tree'),default='all')
p.add_argument('--warmup',type=int,default=5)
p.add_argument('--iterations',type=int,default=3)
args=p.parse_args()
kernel=setup(args.library)
a,b=inputs(4096,42)
ca,cb=cp.asarray(a),cp.asarray(b);out=cp.empty((4096,4096),cp.float32)
ta,tb=torch.from_numpy(a).cuda(0),torch.from_numpy(b).cuda(0)
ty=torch.empty_like(ta)
calls={'shared32':shared_call(ca,cb,out),'block32_tree':kernel.prepared(ca,cb,out),
       'cupy':lambda:cp.matmul(ca,cb,out=out),'torch':lambda:torch.matmul(ta,tb,out=ty)}
if args.method!='all':calls={args.method:calls[args.method]}
for fn in calls.values():
    for _ in range(args.warmup):fn()
cp.cuda.get_current_stream().synchronize()
cp.cuda.profiler.start()
for name,fn in calls.items():
    cp.cuda.nvtx.RangePush(name)
    for _ in range(args.iterations):fn()
    cp.cuda.get_current_stream().synchronize()
    cp.cuda.nvtx.RangePop()
cp.cuda.profiler.stop()
print('Profiler workload completed')
