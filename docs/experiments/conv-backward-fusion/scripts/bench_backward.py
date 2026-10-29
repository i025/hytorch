"""Full Conv2d backward and dInput-only measurements on an allocated CUDA GPU.

CUDA Event elapsed time includes gaps between submitted kernels. Host submit,
end-event wait, and synchronized wall time are separate columns. Forward and
H2D initialization are outside the measured backward operation for all methods.
"""
import argparse
from contextlib import nullcontext
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
SHAPES = {
    'conv2': ((4,24,31,98),(36,24,5,5),2),
    'conv3': ((4,36,14,47),(48,36,5,5),2),
    'conv4': ((4,48,5,22),(64,48,3,3),1),
    'conv5': ((4,64,3,20),(64,64,3,3),1),
}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=['original','fused','torch'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--warmup',type=int,default=5)
    p.add_argument('--iterations',type=int,default=30)
    p.add_argument('--nvtx',action='store_true')
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    import cupy as cp
    import torch
    import hytorch
    from hytorch import Variable
    import hytorch.functions_conv as FC
    from hytorch.backends.cuda import backend,dispatch_stats
    from hytorch.backends.cuda.runtime import conv2d_dinput
    cp.cuda.Device(0).use()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    is_torch=a.method=='torch'
    (a.output/'config.json').write_text(json.dumps(dict(method=a.method,warmup=a.warmup,
        iterations=a.iterations,nvtx=a.nvtx,seed=42,shapes=SHAPES,dtype='float32',tf32=False,
        full_scope='dInput+dWeight+dBias; prepared NCHW input/weight/gy; no forward/H2D',
        event_scope='stream event span; includes GPU work and host-submission gaps',
        memory_scope='separate post-warmup call: CuPy memory-hook live requested peak, or torch allocator peak; includes outputs; not nvidia-smi'),indent=2)+'\n')
    rows,memory,checks=[],[],[]
    for layer,(shape,wshape,stride) in SHAPES.items():
        rng=np.random.default_rng(42)
        gyshape=(shape[0],wshape[0],(shape[2]-wshape[2])//stride+1,(shape[3]-wshape[3])//stride+1)
        data=dict(x=rng.normal(size=shape).astype(np.float32)*.1,
                  w=rng.normal(size=wshape).astype(np.float32)*.1,
                  gy=rng.normal(size=gyshape).astype(np.float32)*.1)
        np.savez(a.output/(layer+'-inputs.npz'),**data)
        if is_torch:
            x,w,gy=[torch.tensor(data[k],device='cuda') for k in ('x','w','gy')]
            event=lambda:torch.cuda.Event(enable_timing=True)
            elapsed=lambda s,e:s.elapsed_time(e)
            host=lambda v:v.detach().cpu().numpy()
        else:
            x,w,gy=[cp.asarray(data[k]) for k in ('x','w','gy')]
            event=cp.cuda.Event
            elapsed=cp.cuda.get_elapsed_time
            host=cp.asnumpy
            vx,vw=Variable(x),Variable(w)
            y=FC.conv2d(vx,vw,cp.zeros(wshape[0],cp.float32),stride=stride)
            operation=y.creator
        for scope in ('dinput','full'):
            if is_torch:
                def call():
                    return torch.ops.aten.convolution_backward(gy,x,w,[wshape[0]],
                        [stride,stride],[0,0],[1,1],False,[0,0],1,
                        [True,scope=='full',scope=='full'])
                context=nullcontext()
            else:
                def call():
                    if scope=='full':
                        gradients=operation.backward(Variable(gy))
                        return tuple(g.data for g in gradients)
                    if a.method=='fused':
                        return (conv2d_dinput(gy,w,shape,stride),)
                    return (FC.deconv2d(gy,w,stride=stride,outsize=shape[2:]).data,)
                context=backend(conv_backward='fused' if a.method=='fused' else 'original')
            with context,hytorch.no_grad():
                for _ in range(a.warmup):
                    result=call()
                cp.cuda.runtime.deviceSynchronize()
                del result
                dispatch_stats(reset=True)
                for step in range(a.iterations):
                    start,end=event(),event()
                    if a.nvtx:cp.cuda.nvtx.RangePush('backward')
                    if a.nvtx:cp.cuda.nvtx.RangePush(f'OP/micro:{scope}|layer={layer}|step={step}')
                    wall=time.perf_counter()
                    start.record()
                    begin=time.perf_counter()
                    result=call()
                    submit=time.perf_counter()-begin
                    end.record()
                    begin=time.perf_counter()
                    end.synchronize()
                    wait=time.perf_counter()-begin
                    total=time.perf_counter()-wall
                    if a.nvtx:cp.cuda.nvtx.RangePop();cp.cuda.nvtx.RangePop()
                    rows.append(dict(method=a.method,layer=layer,scope=scope,step=step,
                        event_us=float(elapsed(start,end))*1000,host_submit_us=submit*1e6,
                        end_event_wait_us=wait*1e6,sync_wall_us=total*1e6))
                    del result
                # Separate allocation measurement, never mixed into timing above.
                cp.cuda.runtime.deviceSynchronize()
                if is_torch:
                    before=torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                    result=call()
                    torch.cuda.synchronize()
                    mem=dict(peak_live_bytes=torch.cuda.max_memory_allocated()-before,
                             retained_bytes=torch.cuda.memory_allocated()-before,
                             malloc_calls=None,requested_bytes=None)
                else:
                    class Memory(cp.cuda.memory_hook.MemoryHook):
                        name='Stage4DLiveRequests'
                        def __init__(self):
                            self.live={};self.peak=0;self.calls=0;self.total=0
                        def malloc_postprocess(self,**kw):
                            self.live[kw['mem_ptr']]=kw['mem_size']
                            self.peak=max(self.peak,sum(self.live.values()))
                            self.calls+=1;self.total+=kw['size']
                        def free_postprocess(self,**kw):
                            self.live.pop(kw['mem_ptr'],None)
                    with Memory() as hook:
                        result=call()
                        cp.cuda.get_current_stream().synchronize()
                    mem=dict(peak_live_bytes=hook.peak,retained_bytes=sum(hook.live.values()),
                             malloc_calls=hook.calls,requested_bytes=hook.total)
                values={label:host(v) for label,v in zip(('dInput','dWeight','dBias'),result) if v is not None}
                np.savez(a.output/(layer+'-'+scope+'-outputs.npz'),**values)
                output_bytes=sum(v.nbytes for v in values.values())
                memory.append(dict(method=a.method,layer=layer,scope=scope,output_bytes=output_bytes,
                    temporary_peak_over_retained=mem['peak_live_bytes']-mem['retained_bytes'],**mem))
                checks.append(dict(layer=layer,scope=scope,dispatch=dispatch_stats(),
                                   finite=all(bool(np.isfinite(v).all()) for v in values.values())))
                del result
    for name,records in [('latencies.csv',rows),('memory.csv',memory)]:
        with (a.output/name).open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    assert all(c['finite'] for c in checks)
    (a.output/'summary.json').write_text(json.dumps(dict(status='COMPLETED',checks=checks),indent=2)+'\n')
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
