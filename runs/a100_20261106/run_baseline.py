import subprocess, sys, os, json
from pathlib import Path
root=Path(__file__).resolve().parent
jobs=[
 ('smoke_baseline',['benchmarks/bench_cuda_training.py','--device','cuda','--model','tiny','--samples','8','--batch','2','--matmul','shared','--conv','im2col','--pool','native']),
 ('gemm_baseline',['benchmarks/bench_cuda_gemm.py','--sizes','1024','2048','4096','--tiles','16','32','--repeats','10']),
 ('ops_baseline',['benchmarks/bench_cuda_ops.py','--batches','1','4','16','--repeats','10']),
 ('donkey_profile_baseline',['benchmarks/bench_cuda_training.py','--model','donkey','--samples','64','--batch','4','--matmul','shared','--conv','im2col','--pool','native','--warmup','5','--mode','phases'])]
results=[]
for name,args in jobs:
 command=[sys.executable,*args,'--output',str(root/name)]
 print('START',name,flush=True)
 with (root/(name+'.log')).open('w') as log:
  p=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
 results.append(dict(name=name,command=command,returncode=p.returncode))
 (root/'baseline_commands.json').write_text(json.dumps(results,indent=2))
 print('END',name,p.returncode,flush=True)
 if name=='smoke_baseline' and p.returncode: break
