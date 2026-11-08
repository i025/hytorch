"""Continue the checked baseline window: apply, build, verify and paired experiments."""
from pathlib import Path
import subprocess,sys,os,json,datetime,shutil
root=Path.cwd();O=root/'runs/a100_20261106/gemm_shape_dispatch';D=O/'diagnostics';T=root/'runs/a100_20261106/gemm_shape_dispatch_traces'
T.mkdir(exist_ok=True)
assert '56 passed' in (O/'raw/baseline_gpu_retry.log').read_text(),'Complete baseline before mutation'
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',CUDA_PATH='/usr/local/cuda',LD_LIBRARY_PATH='/usr/local/cuda/lib64',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',PYTHONHASHSEED='0',HYTORCH_REQUIRE_CUDA='1',NVIDIA_TF32_OVERRIDE='0',HYTORCH_DISPATCH_EVIDENCE=str(O/'raw/dispatch_tests'));env['PATH']='/usr/local/cuda/bin:'+env['PATH']
commands=[];py=sys.executable;nsys='/opt/nvidia/nsight-systems/2025.1.3/bin/nsys'
def run(name,cmd):
 print('START',name,flush=True);start=datetime.datetime.now(datetime.timezone.utc).isoformat()
 with (O/'raw'/f'{name}.log').open('w') as f:r=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
 commands.append(dict(name=name,argv=cmd,start_utc=start,end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),returncode=r.returncode));(O/'raw/benchmark_resume_commands.json').write_text(json.dumps(dict(environment={k:env[k] for k in ['CUDA_VISIBLE_DEVICES','CUDA_PATH','LD_LIBRARY_PATH','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','PYTHONHASHSEED','HYTORCH_REQUIRE_CUDA','NVIDIA_TF32_OVERRIDE']},commands=commands),indent=2));print('END',name,r.returncode,flush=True)
 if r.returncode:raise RuntimeError(name+' failed')
def profile(name,cmd):run(name,[nsys,'profile','--sample=none','--cpuctxsw=none','--trace=cuda,nvtx,osrt','--capture-range=cudaProfilerApi','--capture-range-end=stop','--output',str(T/name)]+cmd)
s=subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
assert [int(x.strip()) for x in s.strip().split(',')]==[0,0],s
try:
 run('gemm_paired_retry',[py,str(D/'gemm_bench.py'),'--output',str(O/'raw/gemm_paired_retry')])
 run('training_paired',[py,str(D/'train.py'),'--output',str(O/'raw/training_paired')])
 profile('gemm',[py,str(D/'gemm_bench.py'),'--profile','--output',str(O/'raw/gemm_profile')])
 for mode in ['old','new']:profile('train_'+mode,[py,str(D/'train.py'),'--trace',mode,'--output',str(O/'raw'/('train_'+mode))])
finally:run('gpu_final',['nvidia-smi'])
