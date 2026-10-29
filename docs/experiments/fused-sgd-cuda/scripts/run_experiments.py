"""Sequential stage 4B experiments; invoke inside the existing single-GPU guard."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
SCRIPTS = Path(__file__).resolve().parent
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--library', type=Path, required=True)
p.add_argument('--gpu-uuid', required=True)
a = p.parse_args()
a.output = a.output.resolve()
a.output.mkdir(parents=True, exist_ok=False)
env = dict(os.environ, HYTORCH_CUDA_LIBRARY=str(a.library.resolve()),
           HYTORCH_REQUIRE_CUDA='1', HYTORCH_SGD_TEST_OUTPUT=str(a.output/'correctness'))
records = []


def run(name, command):
    record = dict(name=name, command=[str(v) for v in command],
                  started_at=datetime.now(timezone.utc).isoformat(),
                  log=str(a.output/(name+'.log')))
    records.append(record)
    (a.output/'commands.json').write_text(json.dumps(records, indent=2)+'\n')
    print('START '+name, flush=True)
    start = time.monotonic()
    with Path(record['log']).open('x') as stream:
        result = subprocess.run(record['command'], cwd=ROOT, env=env,
                                stdout=stream, stderr=subprocess.STDOUT)
    record.update(returncode=result.returncode, wall_s=time.monotonic()-start,
                  finished_at=datetime.now(timezone.utc).isoformat())
    (a.output/'commands.json').write_text(json.dumps(records, indent=2)+'\n')
    print(json.dumps(record), flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)


run('gpu-correctness', [sys.executable, '-m', 'pytest', 'tests/test_fused_sgd_gpu.py',
                       '-q', '-ra', '--junitxml='+str(a.output/'gpu-correctness.xml')])
methods = ['baseline', 'fused', 'torch']
configs = {
    'baseline': ['--framework', 'hytorch', '--matmul', 'shared', '--conv', 'im2col', '--pool', 'cupy', '--tile', '16'],
    'fused': ['--framework', 'hytorch', '--matmul', 'shared', '--conv', 'im2col', '--pool', 'cupy', '--tile', '16', '--fused-sgd'],
    'torch': ['--framework', 'torch'],
}
common = ['--device', 'cuda', '--gpu', '0', '--monitor-id', a.gpu_uuid,
          '--model', 'donkey', '--batch', '4', '--warmup', '10', '--seed', '42',
          '--lr', '0.001', '--workers', '0', '--no-monitor', '--no-fingerprints',
          '--save-comparison-inputs']
micro = [sys.executable, SCRIPTS/'bench_update.py', '--warmup', '5', '--updates', '100', '--repeats', '1']
for repeat in range(1, 4):
    ordered = methods[repeat-1:]+methods[:repeat-1]
    for method in ordered:
        name = f'update-{method}-r{repeat}'
        run(name, [*micro, '--method', method, '--output', a.output/name])
for repeat in range(1, 4):
    ordered = methods[repeat-1:]+methods[:repeat-1]
    for mode in ['throughput', 'phases']:
        for method in ordered:
            name = f'training-{mode}-{method}-r{repeat}'
            run(name, [sys.executable, ROOT/'benchmarks/bench_cuda_training.py',
                       *common, '--samples', '256', '--epochs', '4', *configs[method],
                       '--mode', mode, '--output', a.output/name])
for kind in ['update', 'training']:
    for method in methods:
        name = f'trace-{kind}-{method}'
        if kind == 'update':
            app = [*micro, '--method', method, '--nvtx', '--output', a.output/name]
        else:
            app = [sys.executable, SCRIPTS/'profile_training.py', *common, '--samples', '64',
                   '--epochs', '1', *configs[method], '--mode', 'throughput', '--nvtx',
                   '--output', a.output/name]
        run(name, ['nsys', 'profile', '--trace=cuda,nvtx', '--sample=none',
                   '--cpuctxsw=none', '-o', a.output/name, *app])
        run(name+'-export', ['nsys', 'export', '--type', 'sqlite',
                            '--output', a.output/(name+'.sqlite'), a.output/(name+'.nsys-rep')])
(a.output/'completion.json').write_text(json.dumps(dict(status='PASSED',
    completed_commands=len(records), finished_at=datetime.now(timezone.utc).isoformat()), indent=2)+'\n')
