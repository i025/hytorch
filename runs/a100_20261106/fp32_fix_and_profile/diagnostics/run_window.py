from pathlib import Path
import subprocess,sys,os,json,datetime,argparse
root=Path.cwd();p=argparse.ArgumentParser();p.add_argument('--output',default='runs/a100_20261106/fp32_fix_and_profile');p.add_argument('--traces',default='runs/a100_20261106/fp32_fix_and_profile_traces');args=p.parse_args();out=root/args.output;diag=Path(__file__).resolve().parent;traces=root/args.traces;traces.mkdir(parents=True,exist_ok=False);(out/'raw').mkdir(parents=True,exist_ok=True)
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',CUDA_PATH='/usr/local/cuda',LD_LIBRARY_PATH='/usr/local/cuda/lib64',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',PYTHONHASHSEED='0',HYTORCH_REQUIRE_CUDA='1',NVIDIA_TF32_OVERRIDE='0')
py=sys.executable;nsys='/opt/nvidia/nsight-systems/2025.1.3/bin/nsys';commands=[]
def run(name,cmd):
    print('START',name,flush=True);start=datetime.datetime.now(datetime.timezone.utc).isoformat()
    with (out/'raw'/f'{name}.log').open('w') as f:r=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
    commands.append(dict(name=name,argv=cmd,start_utc=start,end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),returncode=r.returncode))
    (out/'raw/window_commands.json').write_text(json.dumps(dict(environment={k:env[k] for k in ['CUDA_VISIBLE_DEVICES','CUDA_PATH','LD_LIBRARY_PATH','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','PYTHONHASHSEED','HYTORCH_REQUIRE_CUDA','NVIDIA_TF32_OVERRIDE']},commands=commands),indent=2))
    print('END',name,r.returncode,flush=True)
    if r.returncode:raise RuntimeError(f'{name} failed; inspect log')
def profile(name,cmd,extra=None):
    run(name,[nsys,'profile','--sample=none','--cpuctxsw=none','--trace=cuda,nvtx,osrt','--capture-range=cudaProfilerApi','--capture-range-end='+'stop','--output',str(traces/name)]+(extra or [])+cmd)
run('gpu_before',['nvidia-smi'])
status=subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
if [int(v.strip()) for v in status.strip().split(',')]!=[0,0]:raise SystemExit('GPU1 not idle')
try:
    run('dtype_gpu_tests',[py,'-m','pytest','tests/test_backward_dtype.py','-k','cuda','-q'])
    run('original_gpu_tests',[py,'-m','pytest','tests/test_cuda_device.py','-q'])
    for name,legacy in [('workload',False),('before_workload',True)]:
        run(name,[py,str(diag/'diagnose.py'),'--mode','workload','--output',str(out/'raw'/name)]+(['--legacy'] if legacy else []))
    run('training',[py,str(diag/'train.py'),'--output',str(out/'raw/training')])
    for name,mode,back in [('native_trace','trace','native'),('native_plain','plain','native'),('cupy_trace','trace','cupy')]:
        profile(name,[py,str(diag/'diagnose.py'),'--mode',mode,'--backend',back,'--steps','10','--output',str(out/'raw'/name)])
    for mode in ['before_native','after_native','after_cupy','torch']:
        profile('train_'+mode,[py,str(diag/'train.py'),'--trace',mode,'--output',str(out/'raw'/('train_'+mode))])
    profile('gemm',[py,str(diag/'gemm.py'),'--output',str(out/'raw/gemm'),'--workload',str(out/'raw/workload/operators.json')])
finally:
    run('gpu_after',['nvidia-smi'])
