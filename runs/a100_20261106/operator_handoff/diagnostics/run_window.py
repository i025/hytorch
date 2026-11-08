from pathlib import Path
import subprocess,sys,os,json,datetime,argparse
root=Path.cwd();script_dir=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--handoff',default='runs/a100_20261106/operator_handoff');p.add_argument('--traces',default='runs/a100_20261106/operator_handoff_traces');args=p.parse_args()
out=(root/args.handoff).resolve();traces=(root/args.traces).resolve();traces.mkdir(parents=True,exist_ok=False);(out/'raw').mkdir(parents=True,exist_ok=True)
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',CUDA_PATH='/usr/local/cuda',LD_LIBRARY_PATH='/usr/local/cuda/lib64',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',PYTHONHASHSEED='0')
nsys='/opt/nvidia/nsight-systems/2025.1.3/bin/nsys';py=sys.executable
commands=[]
def run(name,cmd):
 print('START',name,flush=True)
 with (out/'raw'/f'{name}.log').open('w') as f:r=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
 commands.append(dict(name=name,command=cmd,returncode=r.returncode,time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
 (out/'raw/window_commands.json').write_text(json.dumps(dict(environment={k:env[k] for k in ['CUDA_VISIBLE_DEVICES','CUDA_PATH','LD_LIBRARY_PATH','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','PYTHONHASHSEED']},commands=commands),indent=2))
 print('END',name,r.returncode,flush=True);return r.returncode
run('gpu_before',['nvidia-smi'])
status=subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
if [int(x.strip()) for x in status.strip().split(',')]!=[0,0]:raise SystemExit('GPU 1 is no longer idle')
diag=str(script_dir/'diagnose.py')
if run('workload',[py,diag,'--mode','workload','--output',str(out/'raw/workload')]):raise SystemExit('Diagnose workload failed')
for name,mode,backend in [('native_trace','trace','native'),('native_plain','plain','native'),('cupy_trace','trace','cupy')]:
 cmd=[nsys,'profile','--sample=none','--cpuctxsw=none','--trace=cuda,nvtx,osrt','--capture-range=cudaProfilerApi','--capture-range-end=stop','--output',str(traces/name),py,diag,'--mode',mode,'--backend',backend,'--steps','10','--output',str(out/'raw'/name)]
 if run(name,cmd):print('Trace failed: inspect log before retry',flush=True)
run('observer_overhead',[py,str(script_dir/'compare_overhead.py'),'--output',str(out/'raw/observer_overhead')])
run('gpu_after_initial',['nvidia-smi'])
