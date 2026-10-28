"""One-device sequential experiments, checking exclusive idleness before each stage."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from wait_for_gpu import snapshot,idle
root=Path(__file__).resolve().parents[4]
uuid=sys.argv[1];py=sys.executable
monitor=root/'docs/experiments/a100-stage2a-gemm/scripts/run_monitored.py'
scripts=root/'docs/experiments/a100-gpu-window/scripts'
env=dict(os.environ,TORCH_HOME=str(root/'.venv/torch-cache'))
tasks=[
 ('torch_gemm',[py,str(scripts/'torch_gemm.py'),'--output','runs/torch_gemm_01']),
 ('gemm_diagnosis',[py,str(scripts/'diagnose_gemm.py'),'--output','runs/gemm_diagnosis_01']),
]
for model in ('tiny','donkey'):
 tasks.append((f'{model}_phases',[py,'scripts/run_cuda_comparison.py','--model',model,'--samples','64','--batch','4','--epochs','1','--repeats','1','--include-torch','--mode','phases','--no-fingerprints','--output',f'runs/synthetic_{model}_phases_01']))
tasks.append(('nsys',['nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','-o','runs/nsys_window_01',py,'benchmarks/bench_cuda_training.py','--model','donkey','--samples','64','--batch','4','--matmul','shared','--conv','im2col','--mode','throughput','--nvtx','--no-fingerprints','--output','runs/nsys_training_window_01']))
for tile in ('16','32'):
 tasks.append((f'ncu_{tile}',['ncu','--section','SpeedOfLight','--section','MemoryWorkloadAnalysis','--section','SchedulerStats','--section','Occupancy','--kernel-name','regex:gemm_shared','--launch-count','2','-o',f'runs/ncu_window_tile{tile}_01',py,'benchmarks/bench_cuda_gemm.py','--sizes','1024','--tiles',tile,'--repeats','3','--warmup','1','--output',f'runs/ncu_gemm_tile{tile}_01']))
tasks.append(('sanitizer',['compute-sanitizer','--tool','memcheck','--error-exitcode','1','--target-processes','all',py,'scripts/validate_cuda.py','--device','0','--output','runs/gpu_memcheck_window_01']))
for model in ('alexnet','vgg11','vgg16'):
 tasks.append((f'pretrained_{model}',[py,'scripts/convert_imagenet_weights.py','--model',model,'--output',f'runs/pretrained_{model}_01/weights.npz','--verify','--verify-device','cuda','--no-checksums']))
with (root/'runs/remaining_stages_01.jsonl').open('x') as journal:
 for name,command in tasks:
    time.sleep(5)
    current=next(r for r in snapshot() if r['uuid']==uuid)
    if not idle(current):
        journal.write(json.dumps(dict(stage=name,status='BLOCKED_DEVICE_BUSY',observed=current))+'\n');journal.flush()
        raise SystemExit('GPU reclaimed; stop starting further GPU work')
    started=time.time();print('START',name,flush=True)
    code=subprocess.run([py,str(monitor),'--gpu',uuid,'--record',f'runs/window_{name}_monitor_01','--',*command],cwd=root,env=env).returncode
    journal.write(json.dumps(dict(stage=name,returncode=code,started_at=started,finished_at=time.time(),command=command))+'\n');journal.flush()
    print('END',name,code,flush=True)
