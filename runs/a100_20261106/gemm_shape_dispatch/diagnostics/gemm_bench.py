"""Unprofiled CUDA-event paired GEMMs; actual kernels/launch dimensions from graphs."""
import sys,os,argparse,json,csv,statistics,hashlib,re,math,ctypes as ct
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import numpy as np
import cupy as cp
from hytorch.backends.cuda import runtime as rt
from cupy_backends.cuda.libs import cublas
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--profile',action='store_true');args=p.parse_args()
O=Path(args.output);O.mkdir(parents=True,exist_ok=False)
root=Path('runs/a100_20261106/gemm_shape_dispatch')
paths={'old':root/'lib_before.so','new':root/'lib_after.so','forced_shared':Path('runs/a100_20261106/libhytorch_cuda_baseline.so')}
libs={};hashes={}
for mode,path in paths.items():
 os.environ['HYTORCH_CUDA_LIBRARY']=str(path.resolve());rt._library=None;libs[mode]=rt.load_library();hashes[mode]=hashlib.sha256(path.read_bytes()).hexdigest()
handle=cp.cuda.device.get_cublas_handle();cublas.setMathMode(handle,cublas.CUBLAS_DEFAULT_MATH)
# CuPy's wrapper rejects capture. Use the same loaded cuBLAS library/handle
# through the standard SGEMM ABI; no custom CUDA math or changed precision.
cublas_paths={line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if '/libcublas.so.' in line and line.split()[-1].startswith('/')}
assert len(cublas_paths)==1,cublas_paths
blas=ct.CDLL(next(iter(cublas_paths)))
blas.cublasSetStream_v2.argtypes=[ct.c_void_p,ct.c_void_p];blas.cublasSetStream_v2.restype=ct.c_int
blas.cublasSgemm_v2.argtypes=[ct.c_void_p]+[ct.c_int]*5+[ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_void_p,ct.c_int]
blas.cublasSgemm_v2.restype=ct.c_int
alpha,beta=ct.c_float(1),ct.c_float(0)
def blas_dot(a,b,out,m,n,k):
    assert blas.cublasSetStream_v2(handle,cp.cuda.get_current_stream().ptr)==0
    status=blas.cublasSgemm_v2(handle,0,0,n,m,k,ct.byref(alpha),b.data.ptr,n,a.data.ptr,k,ct.byref(beta),out.data.ptr,n)
    if status:raise RuntimeError(f'cublasSgemm failed: {status}')
work=json.loads(Path('runs/a100_20261106/fp32_fix_and_profile/raw/workload/operators.json').read_text());shapes={}
for r in work:
 if 'gemm' not in r or (r['operator']=='MatMul.forward' and r.get('attrs',{}).get('native')):continue
 g=r['gemm'];key=(g['M'],g['N'],g['K']);v=shapes.setdefault(key,dict(calls=0,origins=[],group='training'));v['calls']+=1;v['origins'].append(r['layer']+':'+r['phase'])
for key in [(1024,1024,1024),(2048,2048,2048),(4096,4096,4096),(63,8192,65),(64,8128,65),(64,8192,63),(64,8192,64),(64,8192,65),(64,8256,65),(65,4096,65),(8192,63,65),(8192,64,64),(512,1024,63),(512,1024,64),(512,1024,65),(511,1023,67),(449,897,71),(127,4033,129),(97,193,113),(65,65,65),(2049,255,257),(255,2049,257),(1025,513,255)]:
 shapes.setdefault(key,dict(calls=0,origins=[],group='square' if key[0]==key[1]==key[2] else 'general'))
rows=[];decisions=[];dots=O/'graphs';dots.mkdir();modes=['old','new','forced_shared','cublas'];
if args.profile:cp.cuda.runtime.profilerStart()
for si,((m,n,k),info) in enumerate(shapes.items()):
 rng=np.random.default_rng(9000+si);aa=rng.standard_normal((m,k),dtype=np.float32);bb=rng.standard_normal((k,n),dtype=np.float32)
 da,db=cp.asarray(aa),cp.asarray(bb);outs={mode:cp.empty((m,n),cp.float32) for mode in modes}
 if not args.profile:ref32=aa@bb;ref64=aa.astype(np.float64)@bb.astype(np.float64)
 for tile in [16,32]:
  def call(mode):
   if mode=='cublas':blas_dot(da,db,outs[mode],m,n,k)
   else:rt._library=libs[mode];rt.gemm(da,db,'shared',tile,out=outs[mode])
  for mode in modes:
   for _ in range(5):call(mode)
  cp.cuda.get_current_stream().synchronize()
  batch=1 if m*n*k>=1000000000 else (8 if m*n*k>=10000000 else 32)
  graphs={};stream=cp.cuda.Stream(non_blocking=True)
  for mode in modes:
   with stream:
    stream.begin_capture()
    for _ in range(batch):call(mode)
    graph=stream.end_capture();graphs[mode]=graph
   dot=graph.debug_dot_str(flags=1);(dots/f'{si}_t{tile}_{mode}.dot').write_text(dot)
   actual='gemm_register' if 'gemm_register' in dot else ('gemm_shared' if 'gemm_shared' in dot else 'cuBLAS:see_graph_and_profile')
   ctas=((m+63)//64)*((n+63)//64)
   expected='gemm_register' if mode in ['old','new'] and min(m,n,k)>=64 and (mode=='old' or ctas>=128) else 'gemm_shared'
   if mode!='cublas':assert actual==expected,(si,mode,actual,expected,dot)
   decisions.append(dict(shape_id=si,M=m,N=n,K=k,tile=tile,implementation=mode,register_candidate_ctas=ctas,threshold=128 if mode=='new' else '',expected_kernel=expected if mode!='cublas' else '',actual_graph_kernel=actual,graph_file=str((dots/f'{si}_t{tile}_{mode}.dot').relative_to(O)),graph_repetitions=batch))
  # Pre-upload graph executions outside the measured region.
  for g in graphs.values():g.upload();g.launch()
  cp.cuda.get_current_stream().synchronize()
  times={mode:[] for mode in modes};start,end=cp.cuda.Event(),cp.cuda.Event()
  repeats=2 if args.profile else 12
  for repeat in range(repeats):
   order=modes if repeat%2==0 else modes[::-1]
   for mode in order:
    cp.cuda.nvtx.RangePush(f'gemm:{si}:{tile}:{mode}:{repeat}')
    if args.profile:
     call(mode);cp.cuda.get_current_stream().synchronize()
    else:
     start.record();graphs[mode].launch();end.record();end.synchronize()
     times[mode].append(float(cp.cuda.get_elapsed_time(start,end))*1000/batch)
    cp.cuda.nvtx.RangePop()
  if not args.profile:
   hosts={mode:cp.asnumpy(outs[mode]) for mode in modes}
   assert np.array_equal(hosts['old'],hosts['new']),(m,n,k,tile,'new changes output')
   assert np.array_equal(hosts['forced_shared'],hosts['new']),(m,n,k,tile,'shared changes output')
   for mode in modes:
    d=hosts[mode].astype(np.float64)-ref64;rel=float(np.linalg.norm(d)/np.linalg.norm(ref64));assert rel<=1e-5
    rows.append(dict(shape_id=si,M=m,N=n,K=k,tile=tile,group=info['group'],calls_per_training_step=info['calls'],origins=json.dumps(info['origins']),implementation=mode,median_us=statistics.median(times[mode]),mean_us=statistics.mean(times[mode]),stdev_us=statistics.stdev(times[mode]),min_us=min(times[mode]),max_us=max(times[mode]),speedup_vs_old=statistics.median(times['old'])/statistics.median(times[mode]),speedup_vs_forced_shared=statistics.median(times['forced_shared'])/statistics.median(times[mode]),old_new_bitwise=True,shared_new_bitwise=True,fp64_max_abs=float(np.max(np.abs(d))),fp64_relative_l2=rel,original_strict_gate=bool(np.allclose(hosts[mode],ref32,atol=1e-4,rtol=1e-4)),repeats=12,graph_batch=batch,repeats_us=json.dumps(times[mode]),timing='unprofiled CUDA events / graph repeated same GEMM; excludes H2D/D2H and Python gaps between kernels'))
   print(m,n,k,tile,{mode:round(statistics.median(times[mode]),3) for mode in modes},flush=True)
 if not args.profile:
  with (O/'comparison.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
if args.profile:cp.cuda.runtime.profilerStop()
with (O/'decisions.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(decisions[0]));w.writeheader();w.writerows(decisions)
(O/'method.json').write_text(json.dumps(dict(library_sha256=hashes,shapes=len(shapes),warmup=5,graph_warmup=1,repeats=2 if args.profile else 12,seed='9000+shape_id',math_mode=cublas.getMathMode(handle),tf32_override=os.environ.get('NVIDIA_TF32_OVERRIDE'),cublas_api='direct cublasSgemm_v2 using CuPy-created handle; row-major C=A@B via column-major C^T=B^T@A^T',note='graph microbenchmark isolates kernels; direct launches separately captured by Nsight; graph debug files observe real kernel names; never modify production kernel math'),indent=2))
