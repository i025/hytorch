"""Prepared dWeight benchmark: Event spans, complete wall and memory requests."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from cases import cases
from hytorch.utils import pair


def append(path,rows):
    exists=path.exists()
    with path.open('a',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        if not exists:writer.writeheader()
        writer.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=['original','fused','torch'],required=True)
    p.add_argument('--repeat',type=int,required=True);p.add_argument('--nvtx',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    import cupy as cp
    import torch
    import hytorch.functions_conv as FC
    from hytorch.backends.cuda import runtime
    cp.cuda.Device(0).use();torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    config=dict(method=a.method,repeat=a.repeat,nvtx=a.nvtx,cases=cases(),warmup=5,iterations=30,
        cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],event_scope='Stream span including host gaps; pure durations from separate CUPTI traces',
        memory_scope='Untimed pool requests / Torch allocator deltas; whole-card samples recorded by window guard')
    (a.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    for case in cases():
        shape,wshape=case['shape'],case['weight_shape'];n,ci,h,w=shape;co,_,kh,kw=wshape
        sh,sw=pair(case['stride']);ph,pw=pair(case['pad']);oh,ow=(h+2*ph-kh)//sh+1,(w+2*pw-kw)//sw+1
        rng=np.random.default_rng(42);x=rng.normal(size=shape).astype(np.float32)*.1
        gy=rng.normal(size=(n,co,oh,ow)).astype(np.float32)*.1
        np.savez(a.output/(case['case']+'-inputs.npz'),x=x,gy=gy)
        cx,cgy=cp.asarray(x),cp.asarray(gy);tx,tgy=torch.tensor(x,device='cuda'),torch.tensor(gy,device='cuda')
        def original():
            col=FC.im2col_array(cx,(kh,kw),(sh,sw),(ph,pw),to_matrix=False)
            return cp.tensordot(cgy,col,((0,2,3),(0,4,5)))
        def reference():return torch.nn.grad.conv2d_weight(tx,wshape,tgy,stride=(sh,sw),padding=(ph,pw))
        call=dict(original=original,fused=lambda:runtime.conv2d_dweight(cx,cgy,(kh,kw),(sh,sw),(ph,pw)),torch=reference)[a.method]
        baseline=cp.asnumpy(original())
        with torch.backends.cudnn.flags(enabled=False):expected=reference().cpu().numpy()
        for _ in range(5):output=call()
        cp.cuda.runtime.deviceSynchronize();del output;latencies=[]
        for step in range(30):
            start,end=cp.cuda.Event(),cp.cuda.Event()
            if a.nvtx:cp.cuda.nvtx.RangePush('OP/micro:dweight|case='+case['case']+'|step='+str(step))
            begin=time.perf_counter();start.record();submit=time.perf_counter();output=call()
            submit=time.perf_counter()-submit;end.record();wait=time.perf_counter();end.synchronize();wait=time.perf_counter()-wait
            wall=time.perf_counter()-begin
            if a.nvtx:cp.cuda.nvtx.RangePop()
            latencies.append(dict(case=case['case'],method=a.method,repeat=a.repeat,step=step,
                event_us=cp.cuda.get_elapsed_time(start,end)*1000,host_submit_us=submit*1e6,end_wait_us=wait*1e6,wall_us=wall*1e6))
            del output
        append(a.output/'latencies.csv',latencies)
        if a.method=='torch':
            before=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();output=call();torch.cuda.synchronize()
            peak=torch.cuda.max_memory_allocated()-before;retained=torch.cuda.memory_allocated()-before
            requests=requested=None;actual=output.cpu().numpy()
        else:
            class Requests(cp.cuda.memory_hook.MemoryHook):
                name='Stage4GRequests'
                def __init__(self):self.live={};self.peak=0;self.calls=0;self.total=0
                def malloc_postprocess(self,**kw):
                    self.live[kw['mem_ptr']]=kw['mem_size'];self.peak=max(self.peak,sum(self.live.values()));self.calls+=1;self.total+=kw['size']
                def free_postprocess(self,**kw):self.live.pop(kw['mem_ptr'],None)
            with Requests() as hook:output=call();cp.cuda.get_current_stream().synchronize()
            peak,retained,requests,requested=hook.peak,sum(hook.live.values()),hook.calls,hook.total;actual=cp.asnumpy(output)
        append(a.output/'memory.csv',[dict(case=case['case'],method=a.method,repeat=a.repeat,peak_live_bytes=peak,
            retained_bytes=retained,temporary_peak_bytes=peak-retained,malloc_requests=requests,requested_bytes=requested,
            im2col_bytes=4*n*ci*kh*kw*oh*ow,native_partial_bytes=4*((n*oh*ow+127)//128)*co*ci*kh*kw,
            cupy_pool_reserved_bytes=cp.get_default_memory_pool().total_bytes(),torch_reserved_bytes=torch.cuda.memory_reserved())])
        np.savez(a.output/(case['case']+'-outputs.npz'),actual=actual,original=baseline,torch_native=expected)
        errors=[]
        for name,ref in [('original',baseline),('torch_native',expected)]:
            delta=actual.astype(np.float64)-ref;norm=np.linalg.norm(ref.astype(np.float64))
            errors.append(dict(case=case['case'],method=a.method,reference=name,repeat=a.repeat,max_abs=float(abs(delta).max()),
                relative_l2=float(np.linalg.norm(delta)/norm),mismatches=int((abs(delta)>1e-5+1e-4*abs(ref)).sum()),finite=bool(np.isfinite(actual).all())))
        append(a.output/'errors.csv',errors)
        for ref in (baseline,expected):np.testing.assert_allclose(actual,ref,atol=1e-5,rtol=1e-4)
        del output;print(case['case']+' '+a.method+' complete',flush=True)
    (a.output/'summary.json').write_text(json.dumps(dict(status='COMPLETED',cases=len(cases())),indent=2)+'\n')


if __name__=='__main__':main()
