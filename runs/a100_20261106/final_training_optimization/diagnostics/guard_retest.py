from pathlib import Path
import sys,os,json,subprocess,datetime
O=Path.cwd()/'runs/a100_20261106/final_training_optimization';D=O/'diagnostics';T=O/'traces';T.mkdir(exist_ok=True);py=sys.executable
E=os.environ.copy();E.update(CUDA_VISIBLE_DEVICES='1',CUDA_PATH='/usr/local/cuda',LD_LIBRARY_PATH='/usr/local/cuda/lib64',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',HYTORCH_REQUIRE_CUDA='1',NVIDIA_TF32_OVERRIDE='0',PYTHONHASHSEED='0');E['PATH']='/usr/local/cuda/bin:'+E['PATH'];commands=[]
def run(name,args,allow=False):
 print('START',name,flush=True);start=datetime.datetime.now(datetime.timezone.utc).isoformat()
 with (O/'raw'/f'{name}.log').open('w') as f:r=subprocess.run(args,env=E,stdout=f,stderr=subprocess.STDOUT)
 commands.append(dict(name=name,argv=args,start=start,end=datetime.datetime.now(datetime.timezone.utc).isoformat(),code=r.returncode));(O/'raw/guard_commands.json').write_text(json.dumps(dict(env={k:E[k] for k in ['CUDA_VISIBLE_DEVICES','CUDA_PATH','LD_LIBRARY_PATH','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','HYTORCH_REQUIRE_CUDA','NVIDIA_TF32_OVERRIDE','PYTHONHASHSEED']},commands=commands),indent=2));print('END',name,r.returncode,flush=True)
 if r.returncode and not allow:raise RuntimeError(name)
def profile(name,args):
 nsys='/opt/nvidia/nsight-systems/2025.1.3/bin/nsys';run(name,[nsys,'profile','--sample=none','--cpuctxsw=none','--trace=cuda,nvtx,osrt','--capture-range=cudaProfilerApi','--capture-range-end=stop','--output',str(T/name)]+args);run(name+'_export',[nsys,'export','--type=sqlite','--output',str(T/(name+'.sqlite')),str(T/(name+'.nsys-rep'))])
pytest=[py,'-c','import torch,pytest,sys;sys.exit(pytest.main(sys.argv[1:]))']
run('gpu_guard_before',['nvidia-smi'])
assert subprocess.check_output(['nvidia-smi','--id=1','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()==''
try:
 run('guard_tests',pytest+['tests/test_final_optimization.py','-q'])
 for task in ['main','supplement','memory']:run(task+'_guard',[py,str(D/'train.py'),'--task',task])
 run('full_wrappers_guard',[py,str(D/'full_wrappers.py')])
 for rep in range(2):
  for mode in (['B0','D','S','DS'] if rep==0 else ['DS','S','D','B0']):profile(f'train_{mode}_{rep}',[py,str(D/'train.py'),'--task','trace','--mode',mode,'--tag',str(rep)])
 run('full_tests_guard',pytest+['tests','-q','-rs'])
finally:run('gpu_guard_after',['nvidia-smi'])
