from pathlib import Path
import os,sys,subprocess,datetime,json
O=Path.cwd()/'runs/a100_20261106/dweight_im2col_reuse';D=O/'diagnostics';commands=[]
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',CUDA_PATH='/usr/local/cuda',LD_LIBRARY_PATH='/usr/local/cuda/lib64',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',HYTORCH_REQUIRE_CUDA='1',NVIDIA_TF32_OVERRIDE='0',PYTHONHASHSEED='0');py=sys.executable
keys=['CUDA_VISIBLE_DEVICES','CUDA_PATH','LD_LIBRARY_PATH','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','HYTORCH_REQUIRE_CUDA','NVIDIA_TF32_OVERRIDE','PYTHONHASHSEED']
def run(name,cmd):
 print('START',name,flush=True);t=datetime.datetime.now(datetime.timezone.utc).isoformat()
 with (O/'raw'/f'{name}.log').open('w') as f:r=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
 commands.append(dict(name=name,argv=cmd,start_utc=t,end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),returncode=r.returncode));(O/'raw/commands.json').write_text(json.dumps(dict(env={k:env[k] for k in keys},commands=commands),indent=2));print('END',name,r.returncode,flush=True)
 if r.returncode:raise RuntimeError(name)
run('gpu_before',['nvidia-smi'])
assert subprocess.check_output(['nvidia-smi','--id=1','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()==''
pytest=[py,'-c','import torch,pytest,sys;sys.exit(pytest.main(sys.argv[1:]))']
try:
 run('baseline_gpu',pytest+['tests/test_cuda_device.py','tests/test_backward_dtype.py','tests/test_gemm_dispatch.py','tests/test_col2im_cuda.py','tests/test_col2im_binding.py','-k','not cpu','-q'])
 run('baseline_host',pytest+['tests/test_cuda_host.py','tests/test_conv2d_gemm_cpu.py','tests/test_gemm_blocked_packed.py','tests/test_col2im_host.py','tests/test_col2im_binding.py','-k','not cache_changes and not warm_cache and not legacy_col2im and not fast_abi','-q'])
 run('baseline_dtype_host',pytest+['tests/test_backward_dtype.py','-k','not cuda and not donkey','-q'])
 for tool in ['memcheck','racecheck']:
  run(tool,['/usr/local/cuda/bin/compute-sanitizer','--tool',tool,'--error-exitcode','99','--target-processes','all',py,str(D/'math_gate.py'),tool])
finally:run('gpu_after',['nvidia-smi'])
