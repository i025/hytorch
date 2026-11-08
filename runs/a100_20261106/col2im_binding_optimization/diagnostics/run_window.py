from pathlib import Path
import subprocess,sys,os,json,datetime,shutil
ROOT=Path.cwd();O=ROOT/'runs/a100_20261106/col2im_binding_optimization';D=O/'diagnostics';T=ROOT/'runs/a100_20261106/col2im_binding_traces';T.mkdir(exist_ok=False)
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',CUDA_PATH='/usr/local/cuda',LD_LIBRARY_PATH='/usr/local/cuda/lib64',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',PYTHONHASHSEED='0',HYTORCH_REQUIRE_CUDA='1',NVIDIA_TF32_OVERRIDE='0');env['PATH']='/usr/local/cuda/bin:'+env['PATH'];py=sys.executable;commands=[]
nsys='/opt/nvidia/nsight-systems/2025.1.3/bin/nsys'
def run(name,cmd):
 print('START',name,flush=True);start=datetime.datetime.now(datetime.timezone.utc).isoformat()
 with (O/'raw'/f'{name}.log').open('w') as f:r=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
 commands.append(dict(name=name,argv=cmd,start_utc=start,end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),returncode=r.returncode));(O/'raw/window_commands.json').write_text(json.dumps(dict(environment={k:env[k] for k in ['CUDA_VISIBLE_DEVICES','CUDA_PATH','LD_LIBRARY_PATH','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','PYTHONHASHSEED','HYTORCH_REQUIRE_CUDA','NVIDIA_TF32_OVERRIDE']},commands=commands),indent=2));print('END',name,r.returncode,flush=True)
 if r.returncode:raise RuntimeError(name+' failed')
def profile(name,cmd):
 run(name,[nsys,'profile','--sample=none','--cpuctxsw=none','--trace=cuda,nvtx,osrt','--capture-range=cudaProfilerApi','--capture-range-end=stop','--output',str(T/name)]+cmd)
 run(name+'_export',[nsys,'export','--type=sqlite','--output',str(T/(name+'.sqlite')),str(T/(name+'.nsys-rep'))])
run('gpu_before',['nvidia-smi'])
s=subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True);assert [int(x.strip()) for x in s.strip().split(',')]==[0,0],s
shutil.copyfile(ROOT/'hytorch/backends/cuda/build/libhytorch_cuda.so',O/'lib_native_fast.so');shutil.copyfile(ROOT/'hytorch/backends/cuda/build/build.json',O/'raw/build.json')
pytest=[py,'-c','import torch,pytest,sys;sys.exit(pytest.main(sys.argv[1:]))']
try:
 run('gpu_tests',pytest+['tests/test_cuda_device.py','tests/test_backward_dtype.py','tests/test_gemm_dispatch.py','tests/test_col2im_cuda.py','tests/test_col2im_binding.py','-k','not cpu','-q'])
 run('host',pytest+['tests/test_cuda_host.py','tests/test_conv2d_gemm_cpu.py','tests/test_gemm_blocked_packed.py','tests/test_col2im_host.py','tests/test_col2im_binding.py','-k','not cache_changes and not warm_cache and not legacy_col2im and not fast_abi','-q'])
 run('dtype_host',pytest+['tests/test_backward_dtype.py','-k','not cuda and not donkey','-q'])
 for tool in ['memcheck','racecheck']:
  run(tool,['/usr/local/cuda/bin/compute-sanitizer','--tool',tool,'--error-exitcode','99','--target-processes','all']+pytest+['tests/test_col2im_cuda.py','tests/test_col2im_binding.py','-q'])
 run('cpu',[py,str(D/'cpu.py')])
 run('bench',[py,str(D/'bench.py')])
 run('training',[py,str(D/'train.py'),'--output',str(O/'raw/training')])
 profile('col2im',[py,str(D/'bench.py'),'--profile'])
 for mode in ['cupy','old','fast']:profile('train_'+mode,[py,str(D/'train.py'),'--trace',mode,'--output',str(O/'raw'/('train_'+mode))])
finally:run('gpu_after',['nvidia-smi'])
