"""Run separate, matched phase/throughput repetitions without profiler injection."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--gpu-uuid', required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
configs = {
    'cupy': ['--framework', 'hytorch', '--matmul', 'cupy', '--conv', 'cupy', '--pool', 'cupy', '--tile', '16'],
    'combined': ['--framework', 'hytorch', '--matmul', 'shared', '--conv', 'im2col', '--pool', 'cupy', '--tile', '16'],
    'torch': ['--framework', 'torch'],
}
common = ['--device', 'cuda', '--gpu', '0', '--monitor-id', a.gpu_uuid,
          '--model', 'donkey', '--samples', '256', '--batch', '4', '--epochs', '4',
          '--warmup', '10', '--seed', '42', '--lr', '0.001', '--workers', '0',
          '--no-monitor', '--no-fingerprints', '--save-comparison-inputs']
records = []
for repeat in range(1, 4):
    names = list(configs)
    names = names[repeat-1:] + names[:repeat-1]
    for mode in ['throughput', 'phases']:
        for method in names:
            destination = a.output / f'{mode}-{method}-r{repeat}'
            command = [sys.executable, str(ROOT/'benchmarks/bench_cuda_training.py'),
                       *common, *configs[method], '--mode', mode, '--output', str(destination)]
            started = datetime.now(timezone.utc).isoformat()
            clock = time.monotonic()
            log = a.output / f'{destination.name}.log'
            with log.open('x') as stream:
                result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            record = dict(method=method, mode=mode, repeat=repeat, command=command,
                          started_at=started, finished_at=datetime.now(timezone.utc).isoformat(),
                          wall_s=time.monotonic()-clock, returncode=result.returncode,
                          output=str(destination), log=str(log))
            records.append(record)
            (a.output/'commands.json').write_text(json.dumps(records, indent=2))
            print(json.dumps(record), flush=True)
            if result.returncode:
                raise SystemExit(result.returncode)
