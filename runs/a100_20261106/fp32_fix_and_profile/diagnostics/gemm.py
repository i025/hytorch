"""Real observed GEMM dimensions, contiguous FP32 operands, no dispatch edits."""
import sys,os,json,csv,statistics,ctypes,hashlib,collections,argparse
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import numpy as np
import cupy as cp
from hytorch.backends.cuda import runtime as rt
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--workload',required=True);a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
records=json.loads(Path(a.workload).read_text());groups={}
for r in records:
    if 'gemm' not in r:continue
    if r['operator']=='MatMul.forward' and r.get('attrs',{}).get('native'):continue
    g=r['gemm'];key=(g['M'],g['N'],g['K'])
    z=groups.setdefault(key,dict(calls=0,origins=[]));z['calls']+=1;z['origins'].append(dict(layer=r['layer'],phase=r['phase'],operator=r['operator'],inputs=r['input'],gemm=g))
libs={};hashes={}
for mode,name in [('original','baseline'),('current','optimized')]:
    path=Path(f'runs/a100_20261106/libhytorch_cuda_{name}.so').resolve()
    hashes[mode]=hashlib.sha256(path.read_bytes()).hexdigest();os.environ['HYTORCH_CUDA_LIBRARY']=str(path);rt._library=None;libs[mode]=rt.load_library()
# Explicit default FP32 cuBLAS math; environment NVIDIA_TF32_OVERRIDE=0 is also set.
from cupy_backends.cuda.libs import cublas
handle=cp.cuda.device.get_cublas_handle();cublas.setMathMode(handle,cublas.CUBLAS_DEFAULT_MATH)
rows=[];details=[]
cp.cuda.runtime.profilerStart()
for idx,((m,n,k),info) in enumerate(groups.items()):
    rng=np.random.default_rng(4200+idx);aa=rng.standard_normal((m,k),dtype=np.float32);bb=rng.standard_normal((k,n),dtype=np.float32)
    da,db=cp.asarray(aa),cp.asarray(bb);outputs={mode:cp.empty((m,n),dtype=cp.float32) for mode in ['original','current','cublas']}
    ref32=aa@bb;ref64=aa.astype(np.float64)@bb.astype(np.float64)
    def call(mode):
        if mode=='cublas':cp.dot(da,db,out=outputs[mode])
        else:
            rt._library=libs[mode];rt.gemm(da,db,'shared',16,out=outputs[mode])
    for mode in outputs:
        for _ in range(10):call(mode)
    cp.cuda.get_current_stream().synchronize()
    times={mode:[] for mode in outputs}
    for repeat in range(20):
        modes=list(outputs) if repeat%2==0 else list(outputs)[::-1]
        for mode in modes:
            cp.cuda.nvtx.RangePush(f'gemm:{idx}:{mode}:{repeat}')
            start,end=cp.cuda.Event(),cp.cuda.Event();start.record();call(mode);end.record();end.synchronize()
            times[mode].append(float(cp.cuda.get_elapsed_time(start,end))*1000)
            cp.cuda.nvtx.RangePop()
    host={mode:cp.asnumpy(v) for mode,v in outputs.items()}
    bitwise=np.array_equal(host['original'],host['current'])
    for mode in outputs:
        err=host[mode].astype(np.float64)-ref64
        grid=[(n+15)//16,(m+15)//16,1] if mode=='original' or (mode=='current' and min(m,n,k)<64) else ([(n+63)//64,(m+63)//64,1] if mode=='current' else [])
        row=dict(shape_id=idx,M=m,N=n,K=k,calls_per_training_step=info['calls'],origins=json.dumps(info['origins']),implementation=mode,tile=16,grid=json.dumps(grid),block='[16,16,1]' if mode!='cublas' else 'see CUPTI trace',block_count=int(np.prod(grid)) if grid else '',median_us=statistics.median(times[mode]),mean_us=statistics.mean(times[mode]),stdev_us=statistics.stdev(times[mode]),speedup_vs_original=statistics.median(times['original'])/statistics.median(times[mode]),fp64_max_abs=float(np.abs(err).max()),fp64_relative_l2=float(np.linalg.norm(err)/np.linalg.norm(ref64)),original_strict_gate=bool(np.allclose(host[mode],ref32,atol=1e-4,rtol=1e-4)),current_bitwise_original=bool(bitwise),repeats_us=json.dumps(times[mode]))
        rows.append(row)
    assert bitwise
    assert all(r['fp64_relative_l2']<=1e-5 for r in rows if r['shape_id']==idx)
    print(idx,m,n,k,{mode:round(statistics.median(times[mode]),3) for mode in times},flush=True)
cp.cuda.runtime.profilerStop()
with (out/'gemm.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(out/'method.json').write_text(json.dumps(dict(library_sha256=hashes,seed='4200+shape_id',warmup=10,repeats=20,tile=16,math_mode=cublas.getMathMode(handle),tf32_override=os.environ.get('NVIDIA_TF32_OVERRIDE'),scope='logical M/N/K from instrumentation; random identical FP32 row-major packed inputs and preallocated output; excludes original tensor packing/transposition; current dispatch unchanged, so small shapes retain original shared kernel',gate='unchanged original atol=rtol=1e-4 logged, plus existing revision bitwise and relative L2<=1e-5 check; no threshold relaxed'),indent=2))
