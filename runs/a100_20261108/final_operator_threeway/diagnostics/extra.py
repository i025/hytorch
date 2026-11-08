from common import *
import ctypes
blas_path=next(Path(np.__file__).parent.parent.glob("numpy.libs/*openblas*"))
blas_lib=ctypes.CDLL(str(blas_path))
blas_threads=blas_lib.scipy_openblas_get_num_threads64_
blas_threads.restype=ctypes.c_int
def threadpool_info():return dict(path=str(blas_path),actual_threads=blas_threads(),config=np.__config__.CONFIG["Build Dependencies"]["blas"])
import platform,subprocess
sync=cp.cuda.get_current_stream().synchronize
(O/'raw/environment.json').write_text(json.dumps(dict(python=sys.version,numpy=np.__version__,cupy=cp.__version__,torch=torch.__version__,blas=threadpool_info(),cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),device=str(cp.cuda.runtime.getDeviceProperties(0)),physical_gpu=1,cuda_visible_devices=os.getenv('CUDA_VISIBLE_DEVICES'),tf32_override=os.environ['NVIDIA_TF32_OVERRIDE'],torch_tf32_matmul=torch.backends.cuda.matmul.allow_tf32,torch_tf32_cudnn=torch.backends.cudnn.allow_tf32,seed=SEED,cpu_threads=4,metadata_cache=True,compact_function=True,stream=cp.cuda.get_current_stream().ptr),indent=2))
rows=[];idx=[]
for s in specs():
 c=dict(op='im2col',id=s['id'],spec=s,layout='contiguous');d,g,ref=prepare(c);x=g['x'];col=FC.im2col_array(x,s['k'],s['s'],s['p'],False);n,cc,h,w,kh,kw,sh,sw,ph,pw,oh,ow=dims(s)
 funcs={'cupy_sixdim':lambda:FC.im2col_array(x,s['k'],s['s'],s['p'],False),'cupy_layout_only':lambda:col.transpose(0,4,5,1,2,3).reshape(n*oh*ow,-1),'cupy_complete':lambda:FC.im2col_array(x,s['k'],s['s'],s['p'],True),'native_complete':lambda:native_im(x,s)}
 for fn in funcs.values():
  for _ in range(5):fn()
 for rep in range(16):
  modes=list(funcs);order=modes[rep%4:]+modes[:rep%4]
  for mode in order:
   with config('native' if mode.startswith('native') else 'cupy'):
    sync();a,b=cp.cuda.Event(),cp.cuda.Event();t=time.perf_counter_ns()
    for _ in range(5):result=funcs[mode]()
    submit=(time.perf_counter_ns()-t)/5000;sync();wall=(time.perf_counter_ns()-t)/5000;a.record()
    for _ in range(5):result=funcs[mode]()
    b.record();b.synchronize();rows.append(dict(case=s['id'],mode=mode,repeat=rep,sync_us=wall,submit_us=submit,event_us=cp.cuda.get_elapsed_time(a,b)*200,loops=5))
for c in [c for c in cases() if c['op']=='pool_forward']:
 d,g,ref=prepare(c);s=c['spec'];y,ix=rt.max_pool2d(g['x'],s['k'],s['s'],s['p']);e=accuracy(y,ref);ixerror=int(np.count_nonzero(cp.asnumpy(ix)!=d['index']));assert ixerror==0 and e['mixed_fail_count']==0
 idx.append(dict(case=c['id'],index_mismatch=ixerror,**e))
write('raw/IM2COL_LAYOUT_SAMPLES.csv',rows);write('raw/POOL_INDEX_CHECKS.csv',idx)
write('WORKLOAD.csv',[dict(c,spec=json.dumps(c.get('spec',{})),source='fresh workload validated' if c['origin']=='real' else c['origin']) for c in cases()])
print('layout timing and independent pool indices PASS')
