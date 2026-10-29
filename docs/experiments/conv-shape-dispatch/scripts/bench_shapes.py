"""Matched dInput calls: stream Event span, synchronized wall and allocations."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from cases import cases


def write(path,rows):
    with path.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split',choices=['calibration','held-out','validation-2','validation-final'],required=True)
    p.add_argument('--method',choices=['original','fused','torch','auto'],required=True)
    p.add_argument('--repeat',type=int,default=1)
    p.add_argument('--warmup',type=int,default=5)
    p.add_argument('--iterations',type=int,default=30)
    p.add_argument('--nvtx',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    import cupy as cp
    import torch
    import hytorch
    import hytorch.functions_conv as FC
    from hytorch.backends.cuda.runtime import conv2d_dinput,conv_dinput_dispatch,select_conv_dinput
    cp.cuda.Device(0).use();torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    config=dict(vars(a));config['output']=str(a.output);config['cases']=cases(a.split)
    config.update(dtype='float32',tf32=False,event_scope='Stream span, includes host submission gaps',
        wall_scope='Prepared arrays, complete dInput call and end-event synchronization; no H2D',
        memory_scope='Separate non-timed call; CuPy pool live requests / Torch allocator peak')
    (a.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    rows,errors,memory=[],[],[]
    for case in cases(a.split):
        shape,wshape=case['shape'],case['weight_shape'];s,pad=case['stride'],case['pad']
        n,ci,h,w=shape;co,_,kh,kw=wshape
        oh,ow=(h+2*pad-kh)//s+1,(w+2*pad-kw)//s+1
        rng=np.random.default_rng(42+int(case['case'][1:]))
        weight=rng.normal(size=wshape).astype(np.float32)*.1
        dy=rng.normal(size=(n,co,oh,ow)).astype(np.float32)*.1
        cw,cgy=cp.asarray(weight),cp.asarray(dy)
        cx=cp.empty(shape,dtype=cp.float32) if a.method=='auto' else None
        tw,tgy=torch.tensor(weight,device='cuda'),torch.tensor(dy,device='cuda')
        tx=torch.empty(shape,device='cuda')
        def original():return FC.deconv2d(cgy,cw,stride=s,pad=pad,outsize=shape[2:]).data
        def fused():return conv2d_dinput(cgy,cw,shape,s,pad)
        def reference():
            return torch.ops.aten.convolution_backward(tgy,tx,tw,[co],[s,s],[pad,pad],
                [1,1],False,[0,0],1,[True,False,False])[0]
        mode,reason=select_conv_dinput(shape,wshape,s,pad) if a.method=='auto' else (a.method,'explicit')
        def auto():
            selected,_=conv_dinput_dispatch(cx,cw,cgy,s,pad)
            return fused() if selected=='fused' else original()
        call={'original':original,'fused':fused,'torch':reference,'auto':auto}[a.method]
        expected=reference().cpu().numpy()
        with hytorch.no_grad():
            baseline=cp.asnumpy(original())
            for _ in range(a.warmup):output=call()
            cp.cuda.runtime.deviceSynchronize();del output
            for step in range(a.iterations):
                start,end=cp.cuda.Event(),cp.cuda.Event()
                if a.nvtx:
                    cp.cuda.nvtx.RangePush('backward')
                    cp.cuda.nvtx.RangePush('OP/shape:'+case['case']+'|method='+a.method+'|step='+str(step))
                begin=time.perf_counter();start.record();submit=time.perf_counter()
                output=call();submit=time.perf_counter()-submit;end.record()
                wait=time.perf_counter();end.synchronize();wait=time.perf_counter()-wait
                wall=time.perf_counter()-begin
                if a.nvtx:cp.cuda.nvtx.RangePop();cp.cuda.nvtx.RangePop()
                rows.append(dict(case=case['case'],split=a.split,method=a.method,selected=mode,reason=reason,
                    repeat=a.repeat,step=step,work=case['work'],event_us=cp.cuda.get_elapsed_time(start,end)*1000,
                    host_submit_us=submit*1e6,end_wait_us=wait*1e6,wall_us=wall*1e6))
                del output
            cp.cuda.runtime.deviceSynchronize()
            if mode=='torch':
                before=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
                output=call();torch.cuda.synchronize()
                peak=torch.cuda.max_memory_allocated()-before
                retained=torch.cuda.memory_allocated()-before;calls=None;requested=None
                actual=output.detach().cpu().numpy()
            else:
                class Requests(cp.cuda.memory_hook.MemoryHook):
                    name='Stage4EShapeRequests'
                    def __init__(self):self.live={};self.peak=0;self.calls=0;self.total=0
                    def malloc_postprocess(self,**kw):
                        self.live[kw['mem_ptr']]=kw['mem_size'];self.peak=max(self.peak,sum(self.live.values()))
                        self.calls+=1;self.total+=kw['size']
                    def free_postprocess(self,**kw):self.live.pop(kw['mem_ptr'],None)
                with Requests() as hook:output=call();cp.cuda.get_current_stream().synchronize()
                peak,retained,calls,requested=hook.peak,sum(hook.live.values()),hook.calls,hook.total
                actual=cp.asnumpy(output)
            memory.append(dict(case=case['case'],method=a.method,selected=mode,repeat=a.repeat,
                peak_live_bytes=peak,retained_bytes=retained,temporary_peak_bytes=peak-retained,
                malloc_requests=calls,requested_bytes=requested))
            for name,ref in [('original',baseline),('torch',expected)]:
                delta=actual.astype(np.float64)-ref
                row=dict(case=case['case'],method=a.method,reference=name,repeat=a.repeat,
                    max_abs=float(np.abs(delta).max()),relative_l2=float(np.linalg.norm(delta)/np.linalg.norm(ref.astype(np.float64))),
                    mismatches=int((np.abs(delta)>1e-5+1e-4*np.abs(ref)).sum()),finite=bool(np.isfinite(actual).all()))
                errors.append(row)
                if row['mismatches'] or not row['finite']:
                    np.savez(a.output/(case['case']+'-failure.npz'),weight=weight,dy=dy,actual=actual,expected=ref)
                    write(a.output/'errors.csv',errors)
                    np.testing.assert_allclose(actual,ref,atol=1e-5,rtol=1e-4)
            np.savez(a.output/(case['case']+'-arrays.npz'),weight=weight,dy=dy,actual=actual,original=baseline,torch=expected)
            del output
        print(case['case']+' '+a.method+' '+mode+' complete',flush=True)
    for name,values in [('latencies',rows),('errors',errors),('memory',memory)]:write(a.output/(name+'.csv'),values)
    (a.output/'summary.json').write_text(json.dumps(dict(status='COMPLETED',cases=len(cases(a.split)),
        numerical_comparisons=len(errors),max_abs=max(r['max_abs'] for r in errors)),indent=2)+'\n')


if __name__=='__main__':main()
