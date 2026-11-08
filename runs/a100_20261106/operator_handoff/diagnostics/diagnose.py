"""Temporary observation-only hooks. Run from repository root; no production edits."""
import argparse,contextlib,csv,functools,inspect,json,math,os,sys,time
from pathlib import Path
import numpy as np
ROOT=Path.cwd();sys.path.insert(0,str(ROOT))
import cupy as cp
import hytorch
from hytorch import core,functions as F,functions_conv as FC
from hytorch.backends.cuda import runtime as rt
from hytorch.optimizers import SGD
from hytorch.dataloaders import DataLoader
from benchmarks.bench_cuda_training import SyntheticDataset,initialize,parameters,snapshot
from donkeycar_hy.model import DonkeyNet

p=argparse.ArgumentParser();p.add_argument('--mode',choices=['workload','trace','plain'],required=True)
p.add_argument('--backend',choices=['native','cupy'],default='native');p.add_argument('--steps',type=int,default=5)
p.add_argument('--output',required=True);p.add_argument('--events',action='store_true');a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
cp.cuda.Device(0).use()
state={'active':False,'phase':'setup','step':0,'stack':[],'records':[],'memory':[],'events':[]}
param_names={}

def array_info(v):
    if isinstance(v,core.Variable):v=v.data
    if isinstance(v,(np.ndarray,cp.ndarray)):
        return dict(shape=list(v.shape),dtype=str(v.dtype),strides_bytes=list(v.strides),
                    c_contiguous=bool(v.flags.c_contiguous),f_contiguous=bool(v.flags.f_contiguous),
                    device='CUDA' if isinstance(v,cp.ndarray) else 'CPU',nbytes=int(v.nbytes),
                    ptr=int(v.data.ptr) if isinstance(v,cp.ndarray) else int(v.ctypes.data),
                    layout='NCHW if convolution image; see operator' if v.ndim==4 else 'strided tensor')
    if isinstance(v,(list,tuple)):return [array_info(x) for x in v]
    if v is None or isinstance(v,(str,bool,int,float)):return v
    return type(v).__name__

def label_layer(args,is_method):
    candidates=list(args[1:] if is_method else args)
    if is_method and args and hasattr(args[0],'inputs'):candidates+=list(args[0].inputs)
    for x in candidates:
        if isinstance(x,core.Variable):x=x.data
        if isinstance(x,cp.ndarray) and x.data.ptr in param_names:return param_names[x.data.ptr].split('/')[0]
    return state['stack'][-1]['layer'] if state['stack'] else ''

def wrap(owner,name,label,is_method=False):
    original=getattr(owner,name)
    @functools.wraps(original)
    def observed(*args,**kwargs):
        if not state['active']:return original(*args,**kwargs)
        vals=args[1:] if is_method else args
        layer=label_layer(args,is_method)
        rec=dict(id=len(state['records']),parent_id=state['stack'][-1]['id'] if state['stack'] else None,
                 step=state['step'],phase=state['phase'],operator=label,layer=layer,
                 input=array_info(vals),kwargs={k:array_info(v) for k,v in kwargs.items()},
                 stream_ptr=int(cp.cuda.get_current_stream().ptr),source=inspect.getsourcefile(original),
                 line=getattr(getattr(original,'__code__',None),'co_firstlineno',None))
        if is_method:
            rec['attrs']={k:array_info(getattr(args[0],k)) for k in ('stride','pad','kernel_size','input_shape','native','axis','keepdims') if hasattr(args[0],k)}
            if hasattr(args[0],'inputs') and name=='backward':
                rec['original_inputs']=array_info(args[0].inputs)
                rec['input_has_creator']=[v.creator is not None for v in args[0].inputs]
        if label=='native.gemm':
            x,w=vals[:2];m,k=x.shape;n=w.shape[1]
            variant=kwargs.get('variant',vals[2] if len(vals)>2 else 'shared')
            rec['gemm']=dict(M=m,N=n,K=k,flops=2*m*n*k,variant=variant,
                            kernel=('gemm_register' if min(m,n,k)>=64 else 'gemm_shared') if variant=='shared' else 'gemm_'+variant,
                            logical_transpose='inputs as observed; runtime packs to row-major')
        if label=='cupy.tensordot':
            x,w=vals[:2];axes=kwargs.get('axes',vals[2] if len(vals)>2 else 2)
            if isinstance(axes,int):ax=list(range(x.ndim-axes,x.ndim));bx=list(range(axes))
            else:ax,bx=axes;ax=[ax] if isinstance(ax,int) else list(ax);bx=[bx] if isinstance(bx,int) else list(bx)
            ax=[i%x.ndim for i in ax];bx=[i%w.ndim for i in bx]
            k=math.prod(x.shape[i] for i in ax);m=x.size//k;n=w.size//k
            rec['gemm']=dict(M=m,N=n,K=k,flops=2*m*n*k,axes=[ax,bx],kernel='CuPy tensordot_core -> cuBLAS; see correlated kernel',logical_transpose='contracted axes moved to head; not literal cuBLAS transa/transb')
        if label=='MatMul.forward':
            x,w=vals[:2];m,k=x.shape;n=w.shape[1]
            rec['gemm']=dict(M=m,N=n,K=k,flops=2*m*n*k,kernel=('native dispatch (see child native.gemm)' if getattr(args[0],'native',False) and a.backend=='native' else 'CuPy ndarray.dot'),logical_transpose='see input strides')
        state['records'].append(rec);state['stack'].append(rec)
        cp.cuda.nvtx.RangePush('op_%d'%rec['id'])
        ev=(cp.cuda.Event(),cp.cuda.Event()) if a.events else None
        if ev:ev[0].record()
        t=time.perf_counter_ns()
        try:
            value=original(*args,**kwargs)
            rec['cpu_body_ns']=time.perf_counter_ns()-t
            if ev:ev[1].record();state['events'].append((rec,ev))
            rec['output']=array_info(value)
            return value
        finally:
            cp.cuda.nvtx.RangePop();state['stack'].pop()
    setattr(owner,name,observed)

@contextlib.contextmanager
def phase(name):
    old=state['phase'];state['phase']=name
    cp.cuda.nvtx.RangePush('phase:'+name)
    try:yield
    finally:cp.cuda.nvtx.RangePop();state['phase']=old

class MemoryObserver(cp.cuda.memory_hook.MemoryHook):
    name='HytorchObservationOnly'
    def note(self,kind,kw):
        if state['active']:state['memory'].append(dict(kind=kind,step=state['step'],phase=state['phase'],parent_id=state['stack'][-1]['id'] if state['stack'] else None,**{k:int(v) for k,v in kw.items()}))
    def malloc_postprocess(self,**kw):self.note('pool_request',kw)
    def alloc_postprocess(self,**kw):self.note('device_alloc',kw)
    def free_postprocess(self,**kw):self.note('pool_free',kw)

model=DonkeyNet(dropout_rate=0.);initial=initialize(model,(3,66,200),42);model.to_gpu();opt=SGD(.001).setup(model)
ds=SyntheticDataset(64,(3,66,200),42)
backend=rt.backend(matmul='shared',conv='im2col',pool='native') if a.backend=='native' else rt.backend()
def restore():
    for k,v in parameters(model).items():v.data=cp.asarray(initial[k].copy())
    model.cleargrads();param_names.clear();param_names.update({v.data.data.ptr:k for k,v in parameters(model).items()})
def step(batch):
    with phase('transfer'):x,t=[cp.asarray(v) for v in batch]
    with phase('forward'):loss=F.mean_square_error(model(x),t)
    with phase('backward'):model.cleargrads();loss.backward()
    with phase('update'):opt.update()
    return loss
with backend:
    batch=next(iter(DataLoader(ds,4,shuffle=False)))
    for _ in range(10):step(batch)
    cp.cuda.get_current_stream().synchronize();restore()
    if a.mode=='workload':
        loss=step(batch);cp.cuda.get_current_stream().synchronize()
        reference=snapshot(model);ref_grads={k:cp.asnumpy(v.grad.data) for k,v in parameters(model).items()}
        ref_loss=float(loss.data.item());restore()
    if a.mode!='plain':
        seen=set()
        for module in (core,F,FC):
            for cls in vars(module).values():
                if inspect.isclass(cls) and issubclass(cls,core.Function) and cls not in seen:
                    seen.add(cls)
                    for method in ('forward','backward'):
                        if method in cls.__dict__:wrap(cls,method,cls.__name__+'.'+method,True)
        for name in ('im2col_array','col2im_array','_im2col_gpu','_col2im_gpu'):wrap(FC,name,name)
        for name in ('gemm','conv2d','max_pool2d','_call'):wrap(rt,name,'native.'+name)
        for name in ('tensordot','ascontiguousarray'):wrap(cp,name,'cupy.'+name)
        wrap(SGD,'update_one','SGD.update_one',True)
    state['records']=[];state['memory']=[];state['events']=[]
    measured_steps=1 if a.mode=='workload' else a.steps
    cp.cuda.runtime.profilerStart()
    start=time.perf_counter();state['active']=a.mode!='plain'
    with MemoryObserver() if a.mode=='workload' else contextlib.nullcontext():
        iterator=iter(DataLoader(ds,4,shuffle=False))
        for i in range(measured_steps):
            state['step']=i;cp.cuda.nvtx.RangePush('step:%d'%i)
            with phase('data'):batch=next(iterator)
            loss=step(batch)
            with phase('step_sync'):cp.cuda.get_current_stream().synchronize()
            with phase('loss_read'):value=float(loss.data.item())
            cp.cuda.nvtx.RangePop()
    cp.cuda.get_current_stream().synchronize();wall=time.perf_counter()-start;state['active']=False
    cp.cuda.runtime.profilerStop()
    if a.events:
        for rec,(start_ev,end_ev) in state['events']:rec['event_inclusive_ms']=cp.cuda.get_elapsed_time(start_ev,end_ev)
    checks={}
    if a.mode=='workload':
        actual=snapshot(model)
        checks['weights_bitwise_equal']=all(np.array_equal(actual[k],reference[k]) for k in reference)
        checks['gradients_bitwise_equal']=all(np.array_equal(cp.asnumpy(v.grad.data),ref_grads[k]) for k,v in parameters(model).items())
        checks['loss_equal']=value==ref_loss
        if not all(checks.values()):raise AssertionError(checks)
    (out/'operators.json').write_text(json.dumps(state['records'],indent=2))
    (out/'memory.json').write_text(json.dumps(state['memory'],indent=2))
    summary=dict(args=vars(a),steps=measured_steps,wall_s=wall,loss=value,checks=checks,
                 dispatch_counts=rt.dispatch_stats(),gpu=str(cp.cuda.runtime.getDeviceProperties(0)['name']),pci=cp.cuda.Device(0).pci_bus_id,
                 cupy=cp.__version__,numpy=np.__version__,driver=cp.cuda.runtime.driverGetVersion(),runtime=cp.cuda.runtime.runtimeGetVersion(),
                 memory_pool_used_bytes=cp.get_default_memory_pool().used_bytes(),memory_pool_total_bytes=cp.get_default_memory_pool().total_bytes(),
                 instrumented=a.mode!='plain',note='NVTX diagnostic CPU overhead; use plain/original benchmark for end-to-end timing')
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
