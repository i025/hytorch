import subprocess,sys,os,json
from pathlib import Path
root=Path(__file__).resolve().parent
results=[]
def run(name,args,env=None):
 command=[sys.executable,*args]
 print('START',name,flush=True)
 with (root/(name+'.log')).open('w') as f:
  result=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,env=env)
 results.append(dict(name=name,command=command,returncode=result.returncode,library=(env or os.environ).get('HYTORCH_CUDA_LIBRARY')))
 (root/'remaining_commands.json').write_text(json.dumps(results,indent=2))
 print('END',name,result.returncode,flush=True)
 if result.returncode: raise SystemExit(result.returncode)
run('ops_optimized',['benchmarks/bench_cuda_ops.py','--batches','1','4','16','--repeats','10','--output',str(root/'ops_optimized')])
run('smoke_optimized',['benchmarks/bench_cuda_training.py','--model','tiny','--samples','8','--batch','2','--matmul','shared','--conv','im2col','--pool','native','--output',str(root/'smoke_optimized')])
for repeat in range(3):
 groups=['native_baseline','native_optimized','cupy','torch']
 if repeat%2: groups.reverse()
 for group in groups:
  name=f'{group}_{repeat+1}'
  args=['benchmarks/bench_cuda_training.py','--model','donkey','--samples','256','--epochs','3','--batch','4','--warmup','10','--mode','throughput','--interval','0.2','--output',str(root/name)]
  env=os.environ.copy()
  if group.startswith('native'):
   args+=['--matmul','shared','--conv','im2col','--pool','native']
   env['HYTORCH_CUDA_LIBRARY']=str(root/'libhytorch_cuda_baseline.so') if group=='native_baseline' else str(Path('hytorch/backends/cuda/build/libhytorch_cuda.so').resolve())
  if group=='torch': args+=['--framework','torch']
  run(name,args,env)
