"""Run the priority GEMM pilot and full baseline immediately after idle detection."""
import json
from pathlib import Path
import subprocess
import sys
import time

root=Path(__file__).resolve().parents[4]
uuid=sys.argv[1]
monitor=root/'docs/experiments/a100-stage2a-gemm/scripts/run_monitored.py'
bench=root/'benchmarks/bench_cuda_gemm.py'
# New paths preserve the earlier refused launch and all preparation records.
for label,sizes in [('pilot',['128','256']),('formal',['1024','2048','4096'])]:
    output=root/('runs/gemm_stage2a_'+('pilot_03' if label=='pilot' else '01'))
    record=root/f'runs/gpu_window_gemm_{label}_monitor_03'
    command=[sys.executable,str(monitor),'--gpu',uuid,'--record',str(record),'--',
        sys.executable,str(bench),'--device','0','--sizes',*sizes,'--tiles','16','32',
        '--repeats','10','--warmup','5','--seed','42','--output',str(output)]
    result=subprocess.run(command,cwd=root)
    if result.returncode:
        raise SystemExit(result.returncode)
    report=json.loads((output/'summary.json').read_text())
    if not report['all_correct'] or report['gpu_status']!='EXECUTED':
        raise SystemExit('GEMM acceptance failed; original outputs retained')
    if label=='pilot':
        time.sleep(5)  # Let the exited pilot's device utilization sample settle.
