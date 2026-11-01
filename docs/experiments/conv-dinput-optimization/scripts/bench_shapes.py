"""Four matched dInput implementations; Event spans and wall are separate."""
import argparse
import csv
import ctypes as ct
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from cases import cases
from hytorch.utils import pair


def write(path,rows):
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split',choices=['history','fresh'],required=True)
    p.add_argument('--method',choices=['original','fused','fused_opt','torch'],required=True)
    p.add_argument('--scope',choices=['full','kernel'],default='full')
    p.add_argument('--repeat',type=int,default=1);p.add_argument('--warmup',type=int,default=5)
    p.add_argument('--iterations',type=int,default=30);p.add_argument('--nvtx',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.scope=='kernel' and a.method not in ('fused','fused_opt'):p.error('Preallocated kernel scope is native gather only')
    a.output.mkdir(parents=True,exist_ok=False)
    import cupy as cp
    import torch
    import hytorch
    import hytorch.functions_conv as FC
    from hytorch.backends.cuda import runtime
    cp.cuda.Device(0).use();torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    config=dict(vars(a),cases=cases(a.split));config['output']=str(a.output)
    config.update(event_scope='Stream span including host launch gaps, NOT pure kernel execution',
        kernel_time_scope='Nsight CUPTI duration in separate trace',dtype='float32',tf32=False,
        memory_scope='Independent untimed call; CuPy pool requests / Torch allocator deltas')
    (a.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    rows,errors,memory=[],[],[]
    with hytorch.no_grad():
        for case in cases(a.split):
            shape,wshape=case['shape'],case['weight_shape'];n,ci,h,w=shape;co,_,kh,kw=wshape
            sh,sw=pair(case['stride']);ph,pw=pair(case['pad']);oh,ow=(h+2*ph-kh)//sh+1,(w+2*pw-kw)//sw+1
            rng=np.random.default_rng(42+int(case['case'][1:]));weight=rng.normal(size=wshape).astype(np.float32)*.1
            dy=rng.normal(size=(n,co,oh,ow)).astype(np.float32)*.1
            np.savez(a.output/(case['case']+'-inputs.npz'),weight=weight,dy=dy)
            cw,cgy=cp.asarray(weight),cp.asarray(dy)
            tw,tgy=torch.tensor(weight,device='cuda'),torch.tensor(dy,device='cuda');tx=torch.empty(shape,device='cuda')
            def original():return FC.deconv2d(cgy,cw,stride=(sh,sw),pad=(ph,pw),outsize=(h,w)).data
            def reference():
                return torch.ops.aten.convolution_backward(tgy,tx,tw,[co],[sh,sw],[ph,pw],
                    [1,1],False,[0,0],1,[True,False,False])[0]
            functions=dict(original=original,fused=lambda:runtime.conv2d_dinput(cgy,cw,shape,(sh,sw),(ph,pw)),
                fused_opt=lambda:runtime.conv2d_dinput_opt(cgy,cw,shape,(sh,sw),(ph,pw)),torch=reference)
            call=functions[a.method]
            if a.scope=='kernel':
                gx=cp.empty(shape,dtype=cp.float32)
                name='hy_conv_dinput_opt' if a.method=='fused_opt' else 'hy_conv_dinput'
                fn=getattr(runtime.load_library(),name)
                fn.argtypes=[ct.c_void_p]*3+[ct.c_int]*13+[ct.c_void_p];fn.restype=ct.c_int
                args=(cgy.data.ptr,cw.data.ptr,gx.data.ptr,n,ci,h,w,co,kh,kw,sh,sw,ph,pw,oh,ow,cp.cuda.get_current_stream().ptr)
                def call():
                    status=fn(*args)
                    if status:raise RuntimeError(runtime.load_library().hy_cuda_error(status).decode())
                    return gx
            old=cp.asnumpy(original())
            # Use a uniform native FP32 correctness reference; timed torch call
            # above remains standard cuDNN, and both values are retained.
            with torch.backends.cudnn.flags(enabled=False):expected=reference().cpu().numpy()
            for _ in range(a.warmup):output=call()
            cp.cuda.runtime.deviceSynchronize();del output
            for step in range(a.iterations):
                start,end=cp.cuda.Event(),cp.cuda.Event()
                if a.nvtx:
                    cp.cuda.nvtx.RangePush('backward');cp.cuda.nvtx.RangePush('OP/micro:'+a.scope+'|case='+case['case']+'|step='+str(step))
                begin=time.perf_counter();start.record();submit=time.perf_counter();output=call()
                submit=time.perf_counter()-submit;end.record();wait=time.perf_counter();end.synchronize();wait=time.perf_counter()-wait
                wall=time.perf_counter()-begin
                if a.nvtx:cp.cuda.nvtx.RangePop();cp.cuda.nvtx.RangePop()
                rows.append(dict(case=case['case'],split=a.split,method=a.method,scope=a.scope,repeat=a.repeat,step=step,
                    event_us=cp.cuda.get_elapsed_time(start,end)*1000,host_submit_us=submit*1e6,end_wait_us=wait*1e6,wall_us=wall*1e6))
                del output
            if a.method=='torch':
                before=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();output=call();torch.cuda.synchronize()
                peak=torch.cuda.max_memory_allocated()-before;retained=torch.cuda.memory_allocated()-before
                calls=requested=None;actual=output.cpu().numpy()
            else:
                class Requests(cp.cuda.memory_hook.MemoryHook):
                    name='Stage4FRequests'
                    def __init__(self):self.live={};self.peak=0;self.calls=0;self.total=0
                    def malloc_postprocess(self,**kw):
                        self.live[kw['mem_ptr']]=kw['mem_size'];self.peak=max(self.peak,sum(self.live.values()));self.calls+=1;self.total+=kw['size']
                    def free_postprocess(self,**kw):self.live.pop(kw['mem_ptr'],None)
                with Requests() as hook:output=call();cp.cuda.get_current_stream().synchronize()
                peak,retained,calls,requested=hook.peak,sum(hook.live.values()),hook.calls,hook.total;actual=cp.asnumpy(output)
            memory.append(dict(case=case['case'],method=a.method,scope=a.scope,repeat=a.repeat,peak_live_bytes=peak,
                retained_bytes=retained,temporary_peak_bytes=peak-retained,malloc_requests=calls,requested_bytes=requested))
            np.savez(a.output/(case['case']+'-outputs.npz'),actual=actual,original=old,torch_native=expected)
            for name,ref in [('original',old),('torch_native',expected)]:
                delta=actual.astype(np.float64)-ref;norm=np.linalg.norm(ref.astype(np.float64))
                errors.append(dict(case=case['case'],method=a.method,reference=name,scope=a.scope,repeat=a.repeat,
                    max_abs=float(abs(delta).max()),relative_l2=float(np.linalg.norm(delta)/norm),
                    mismatches=int((abs(delta)>1e-5+1e-4*abs(ref)).sum()),finite=bool(np.isfinite(actual).all())))
                if errors[-1]['mismatches'] or not errors[-1]['finite']:
                    write(a.output/'errors.csv',errors);np.testing.assert_allclose(actual,ref,atol=1e-5,rtol=1e-4)
            del output
            print(case['case']+' '+a.method+' '+a.scope+' complete',flush=True)
    for name,values in [('latencies',rows),('errors',errors),('memory',memory)]:write(a.output/(name+'.csv'),values)
    (a.output/'summary.json').write_text(json.dumps(dict(status='COMPLETED',cases=len(cases(a.split))),indent=2)+'\n')


if __name__=='__main__':main()
