"""Run the stage 4A supplement sequentially under the external single-GPU guard.

Validation is a prerequisite. Each child has a fresh output directory and full
stdout/stderr. No production computation or fingerprint code is changed.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
SCRIPTS = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    records = []

    def run(name, command):
        log = args.output / (name + '.log')
        record = dict(name=name, command=[str(v) for v in command],
                      started_at=datetime.now(timezone.utc).isoformat(), log=str(log))
        records.append(record)
        (args.output / 'commands.json').write_text(json.dumps(records, indent=2) + '\n')
        print('START ' + name, flush=True)
        start = time.monotonic()
        with log.open('x') as stream:
            result = subprocess.run(record['command'], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        record.update(returncode=result.returncode, wall_s=time.monotonic()-start,
                      finished_at=datetime.now(timezone.utc).isoformat())
        (args.output / 'commands.json').write_text(json.dumps(records, indent=2) + '\n')
        print(json.dumps(record), flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)

    run('validation', [sys.executable, SCRIPTS/'validate_gpu_instrumentation.py',
                      '--output', args.output/'validation', '--device', '0',
                      '--warmup', '10', '--steps', '32', '--repeats', '3', '--memory-stats', '--no-fingerprints'])
    run('measurements', [sys.executable, SCRIPTS/'run_measurements.py',
                        '--output', args.output/'measurements', '--gpu-uuid', args.gpu_uuid])
    configs = {
        'cupy': ['--framework', 'hytorch', '--matmul', 'cupy', '--conv', 'cupy', '--pool', 'cupy', '--tile', '16'],
        'combined': ['--framework', 'hytorch', '--matmul', 'shared', '--conv', 'im2col', '--pool', 'cupy', '--tile', '16'],
        'torch': ['--framework', 'torch'],
    }
    common = ['--device', 'cuda', '--gpu', '0', '--monitor-id', args.gpu_uuid,
              '--model', 'donkey', '--batch', '4', '--warmup', '10', '--seed', '42',
              '--lr', '0.001', '--workers', '0', '--no-monitor', '--no-fingerprints',
              '--save-comparison-inputs']
    for repeat in range(1, 4):
        names = list(configs)
        names = names[repeat-1:] + names[:repeat-1]
        for method in names:
            name = 'instrumented-' + method + '-r' + str(repeat)
            options = ['--operator-nvtx', '--nvtx']
            if method != 'torch':
                options.append('--memory-stats')
            run(name, [sys.executable, SCRIPTS/'profile_training.py', *options,
                       *common, *configs[method], '--samples', '256', '--epochs', '4',
                       '--mode', 'throughput', '--output', args.output/name])
    # Phase-only controls and detailed traces have identical data and step count.
    # Nsight timings remain separate from unprofiled throughput/phase results.
    for detail in (False, True):
        for method in configs:
            name = 'trace-' + ('detail-' if detail else 'phase-') + method
            prefix = args.output / name
            options = ['--nvtx']
            if detail:
                options.append('--operator-nvtx')
                if method != 'torch':
                    options.append('--memory-stats')
            run(name, ['nsys', 'profile', '--trace=cuda,nvtx', '--sample=none',
                       '--cpuctxsw=none', '-o', prefix,
                       sys.executable, SCRIPTS/'profile_training.py', *options,
                       *common, *configs[method], '--samples', '64', '--epochs', '1',
                       '--mode', 'throughput', '--output', args.output/(name+'-training')])
            run(name+'-export', ['nsys', 'export', '--type', 'sqlite', '--output',
                                str(prefix)+'.sqlite', str(prefix)+'.nsys-rep'])
    (args.output/'completion.json').write_text(json.dumps(dict(
        status='COMPLETED', finished_at=datetime.now(timezone.utc).isoformat(),
        child_commands=len(records)), indent=2)+'\n')


if __name__ == '__main__':
    main()
