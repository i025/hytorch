from pathlib import Path
import sys,subprocess,os,json,datetime
R=Path(__file__).resolve().parents[4];O=R/'runs/a100_20261107/framework_optimization';D=O/'diagnostics'
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',CUDA_PATH='/usr/local/cuda',LD_LIBRARY_PATH='/usr/local/cuda/lib64',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',HYTORCH_REQUIRE_CUDA='1',NVIDIA_TF32_OVERRIDE='0',PYTHONHASHSEED='0');env['PATH']='/usr/local/cuda/bin:'+env['PATH']
base_env=env.copy()
for task in sys.argv[1:]:
 env=base_env.copy()
 bits=task.split(':');kind=bits[0];mode=bits[1] if len(bits)>1 else 'B0'
 if kind=='framework':cmd=[sys.executable,str(D/'framework_bench.py')]
 elif kind in ('memcheck','racecheck'):
  cmd=['/usr/local/cuda/bin/compute-sanitizer','--tool',kind,'--error-exitcode','99',sys.executable,'-c','import torch,pytest,sys;sys.exit(pytest.main(sys.argv[1:]))','tests/test_framework_metadata.py','-q']
 elif kind=='tests':cmd=[sys.executable,'-c','import torch,pytest,sys;sys.exit(pytest.main(sys.argv[1:]))','tests','-q','-rs']
 elif kind=='operators':cmd=[sys.executable,str(D/'operators.py'),*bits[1:]]
 elif kind=='allocations':
  env['HYTORCH_EXPERIMENT_SOURCE']=str(O/'baseline/source')
  cmd=[sys.executable,str(D/'operators.py'),'--allocation-only']
 elif kind=='optrace15':
  path=O/'traces'/'operators_15'
  env['HYTORCH_EXPERIMENT_SOURCE']=str(O/'baseline/source')
  cmd=['/opt/nvidia/nsight-systems/2025.1.3/bin/nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--capture-range=cudaProfilerApi','--capture-range-end=stop','-o',str(path),sys.executable,str(D/'operators.py'),'--trace','--trace-repeats','15']
 elif kind=='optrace':
  path=O/'traces'/'operators_a'
  cmd=['/opt/nvidia/nsight-systems/2025.1.3/bin/nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--capture-range=cudaProfilerApi','--capture-range-end=stop','-o',str(path),sys.executable,str(D/'operators.py'),'--trace']
 elif kind=='trace':
  tag=bits[2] if len(bits)>2 else 'a';path=O/'traces'/f'train_{mode}_{tag}'
  cmd=['/opt/nvidia/nsight-systems/2025.1.3/bin/nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--capture-range=cudaProfilerApi','--capture-range-end=stop','-o',str(path),sys.executable,str(D/'training.py'),'--task','trace','--mode',mode]
 else:cmd=[sys.executable,str(D/'training.py'),'--task',kind,'--mode',mode]
 log=O/'raw'/('_'.join(bits)+'.log')
 if log.exists():raise RuntimeError(f'Refuse overwrite {log}')
 # Reproduction guard added after the late allocator diagnostic overlap.
 # This is a point-in-time check, never a GPU reservation or process termination.
 uuid=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
 active=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True)
 (O/'raw'/('resource_'+('_'.join(bits))+'.json')).write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),target_uuid=uuid,processes=active),indent=2))
 if any(line.split(',')[0].strip()==uuid for line in active.splitlines()):
  raise SystemExit('GPU1 is occupied. Refusing to start this task; existing processes untouched.')
 print('RUN',task,flush=True)
 with log.open('w') as f:r=subprocess.run(cmd,cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT)
 with (O/'raw/commands.jsonl').open('a') as f:f.write(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),argv=cmd,returncode=r.returncode,log=str(log)))+'\n')
 print(task,r.returncode,flush=True)
 if r.returncode:print(log.read_text()[-6000:]);sys.exit(r.returncode)
 if kind in ('trace','optrace','optrace15'):subprocess.run(['/opt/nvidia/nsight-systems/2025.1.3/bin/nsys','export','--type=sqlite','--output',str(path)+'.sqlite',str(path)+'.nsys-rep'],check=True,env=env)
