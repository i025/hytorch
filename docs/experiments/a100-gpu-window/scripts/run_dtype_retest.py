"""Run and retain post-fix validation and matched training on one idle GPU."""
import json
from pathlib import Path
import subprocess
import sys
import time
from wait_for_gpu import snapshot,idle
root=Path(__file__).resolve().parents[4];uuid=sys.argv[1];py=sys.executable
monitor=root/'docs/experiments/a100-stage2a-gemm/scripts/run_monitored.py'
tasks=[('validation',[py,'scripts/validate_cuda.py','--device','0','--output','runs/validation_dtype_fix_01'])]
for mode,repeats in [('throughput','3'),('phases','1')]:
 for model in ('tiny','donkey'):
  tasks.append((f'{model}_{mode}',[py,'scripts/run_cuda_comparison.py','--model',model,'--samples','64','--batch','4','--epochs','1','--repeats',repeats,'--include-torch','--mode',mode,'--no-fingerprints','--output',f'runs/synthetic_{model}_{mode}_fixed_01']))
tasks.append(('nsys',['nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','-o','runs/nsys_dtype_fixed_01',py,'benchmarks/bench_cuda_training.py','--model','donkey','--samples','64','--batch','4','--matmul','shared','--conv','im2col','--mode','throughput','--nvtx','--no-fingerprints','--output','runs/nsys_training_fixed_01']))
journal=root/'runs/dtype_retest_stages_01.jsonl'
previous=[json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []
completed={r['stage'] for r in previous if r.get('returncode')==0}
with journal.open('a') as log:
 for name,command in tasks:
  if name in completed: continue
  time.sleep(5)
  current=next(r for r in snapshot() if r['uuid']==uuid)
  if not idle(current):
   log.write(json.dumps(dict(stage=name,status='BLOCKED_DEVICE_BUSY',observed=current))+'\n');log.flush()
   raise SystemExit('GPU reclaimed; remaining stages not started')
  start=time.time();print('START',name,flush=True)
  code=subprocess.run([py,str(monitor),'--gpu',uuid,'--record',f'runs/fixed_{name}_monitor_01','--',*command],cwd=root).returncode
  log.write(json.dumps(dict(stage=name,returncode=code,started_at=start,finished_at=time.time(),command=command))+'\n');log.flush()
  print('END',name,code,flush=True)
  if code: raise SystemExit(code)
