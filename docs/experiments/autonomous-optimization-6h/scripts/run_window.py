"""One fixed UUID, CPU-only controller, sequential jobs, no device reservation/reset."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
spec = importlib.util.spec_from_file_location('prior_window', ROOT / 'docs/experiments/udacity-real-training/scripts/run_window.py')
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)


def now():
    return datetime.now(timezone.utc).isoformat()


def jobs(args):
    common = ['--prepared', str(args.prepared), '--no-fingerprints']
    plan = [dict(name='numerical-replay', method=None, command=[sys.executable, str(HERE / 'diagnose_replay.py'),
        *common, '--cache', str(args.cache), '--output', str(args.output / 'numerical-replay')])]
    for name, files in [('relu-tests', ['tests/test_relu_backward_gpu.py']),
                        ('gpu-regression', ['tests/test_conv_dweight_gpu.py', 'tests/test_conv_dinput_opt_gpu.py',
                            'tests/test_conv_shape_dispatch_gpu.py', 'tests/test_conv_backward_fusion_gpu.py',
                            'tests/test_fused_sgd_gpu.py', 'tests/test_conv_dinput_pruning.py',
                            'tests/test_cuda_device.py'])]:
        plan.append(dict(name=name, method=None, command=[sys.executable, '-m', 'pytest', *files,
            '-q', '-ra', '--junitxml=' + str(args.output / (name + '.xml'))]))
    plan.append(dict(name='prototype-correctness', method=None, command=[sys.executable, str(HERE / 'check_prototypes.py'),
        *common, '--output', str(args.output / 'prototype-correctness')]))
    plan.append(dict(name='long-j-correctness', method=None, command=[sys.executable, str(HERE / 'check_long_j.py'),
        *common, '--reference', str(args.output / 'numerical-replay'), '--output', str(args.output / 'long-j-correctness')]))
    plan.append(dict(name='relu-micro', method=None, command=[sys.executable, str(HERE / 'bench_relu.py'),
        '--output', str(args.output / 'relu-micro')]))
    methods = 'AFIPRJE'
    for task in ('compute', 'io', 'phases'):
        for repeat in range(1, 4):
            order = methods[repeat - 1:] + methods[:repeat - 1]
            for model in ('donkey', 'spatial'):
                for method in order:
                    name = f'{task}-{model}-{method}-r{repeat}'
                    plan.append(dict(name=name, method=method, command=[sys.executable, str(HERE / 'measure_training.py'),
                        *common, '--task', task, '--model', model, '--method', method,
                        '--output', str(args.output / name)]))
                if task == 'io':
                    name = f'io-{model}-Eprefetch-r{repeat}'
                    plan.append(dict(name=name, method='E', command=[sys.executable, str(HERE / 'measure_training.py'),
                        *common, '--task', task, '--model', model, '--method', 'E', '--workers', '2',
                        '--output', str(args.output / name)]))
    for model in ('donkey', 'spatial'):
        for method in 'IJ':
            name = f'trace-{model}-{method}'
            plan.append(dict(name=name, method=method, command=['nsys', 'profile', '--trace=cuda,nvtx',
                '--sample=none', '--cpuctxsw=none', '-o', str(args.output / name), sys.executable,
                str(HERE / 'measure_training.py'), *common, '--task', 'trace', '--model', model,
                '--method', method, '--steps', '32', '--output', str(args.output / name)]))
    name = 'trace-relu-micro'
    plan.append(dict(name=name, method=None, command=['nsys', 'profile', '--trace=cuda,nvtx',
        '--sample=none', '--cpuctxsw=none', '-o', str(args.output / name), sys.executable,
        str(HERE / 'bench_relu.py'), '--trace', '--output', str(args.output / name)]))
    return plan


class Monitor:
    def __init__(self, output, uuid):
        self.output, self.uuid = output, uuid
        self.active, self.job, self.reason = None, 'preflight', None
        self.stop = threading.Event()

    def run(self):
        with (self.output / 'resources.jsonl').open('x') as stream:
            while not self.stop.is_set():
                row = prior.sample(self.uuid)
                own = []
                if self.active is not None:
                    import psutil
                    own = [self.active.pid]
                    try:
                        own += [p.pid for p in psutil.Process(self.active.pid).children(recursive=True)]
                    except psutil.NoSuchProcess:
                        pass  # The short-lived active child exited between samples.
                unexpected = [p['pid'] for p in row['processes'] if p['pid'] not in own]
                row.update(job=self.job, own_pids=own, unexpected_gpu_pids=unexpected)
                if unexpected:
                    self.reason = 'New external GPU process; stop our work to avoid contention'
                if int(row['gpu']['memory.total']) - int(row['gpu']['memory.used']) < 1024:
                    self.reason = 'Less than 1GiB free memory'
                row['stop_reason'] = self.reason
                stream.write(json.dumps(row) + '\n')
                stream.flush()
                if self.reason:
                    if self.active is not None and self.active.poll() is None:
                        os.killpg(self.active.pid, signal.SIGTERM)
                    return
                self.stop.wait(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--uuid')
    parser.add_argument('--plan', type=Path)
    args = parser.parse_args()
    plan = jobs(args)
    if args.plan:
        with args.plan.open('x') as stream:
            json.dump(plan, stream, indent=2)
        print('Prepared', len(plan), 'sequential jobs; GPU NOT_RUN')
        return
    args.output.mkdir(parents=True, exist_ok=False)
    baseline = prior.sample(args.uuid)
    assert not baseline['processes'] and int(baseline['gpu']['memory.used']) < 500, 'An idle allocated device is required'
    assert int(baseline['gpu']['utilization.gpu']) < 5, 'Device activity is not idle'
    uuid = baseline['gpu']['uuid']
    state = dict(status='RUNNING', physical_gpu=int(baseline['gpu']['index']), uuid=uuid,
        git_version=subprocess.check_output(['git', 'log', '-1', '--format=%H'], cwd=ROOT, text=True).strip(),
        logical_gpu=0, resource_condition='OBSERVED_IDLE_GPU; no exclusive lease', started_at=now(),
        device_switches=0, gpu_reset_called=False, parent_creates_cuda_context=False)
    (args.output / 'before.json').write_text(json.dumps(baseline, indent=2) + '\n')
    (args.output / 'window-start.json').write_text(json.dumps(state, indent=2) + '\n')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, CUDA_DEVICE_ORDER='PCI_BUS_ID',
        HYTORCH_CUDA_LIBRARY=str(args.library.resolve()), NVIDIA_TF32_OVERRIDE='0',
        HYTORCH_RESOURCE_CONDITION=state['resource_condition'], HYTORCH_REQUIRE_CUDA='1',
        HYTORCH_PRUNING_DEVICE='cuda', HYTORCH_SHAPE_TEST_OUTPUT=str(args.output / 'test-arrays'),
        HYTORCH_PRUNING_TEST_OUTPUT=str(args.output / 'pruning-arrays'),
        HYTORCH_SGD_TEST_OUTPUT=str(args.output / 'sgd-arrays'),
        HYTORCH_CONV_TEST_OUTPUT=str(args.output / 'legacy-arrays'))
    monitor = Monitor(args.output, uuid)
    thread = threading.Thread(target=monitor.run)
    thread.start()
    records = []
    try:
        for job in plan:
            if monitor.reason:
                raise RuntimeError(monitor.reason)
            if datetime.now(timezone.utc) >= datetime(2026, 11, 4, 23, 6, 35, tzinfo=timezone.utc):
                state['status'] = 'BUDGET_STOP'
                break
            local = dict(env)
            if job['method'] == 'A':
                local['HYTORCH_CUDA_LIBRARY'] = str(args.output / 'absent_native_library.so')
            row = dict(job, started_at=now(), uuid=uuid, library=local['HYTORCH_CUDA_LIBRARY'])
            records.append(row)
            monitor.job = job['name']
            print('START', job['name'], flush=True)
            with (args.output / (job['name'] + '.log')).open('x') as stream:
                monitor.active = subprocess.Popen(job['command'], cwd=ROOT, env=local,
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                row['pid'] = monitor.active.pid
                (args.output / 'commands.json').write_text(json.dumps(records, indent=2) + '\n')
                code = monitor.active.wait()
            monitor.active = None
            row.update(returncode=code, finished_at=now())
            (args.output / 'commands.json').write_text(json.dumps(records, indent=2) + '\n')
            if code:
                raise subprocess.CalledProcessError(code, job['command'])
            if job['name'] == 'numerical-replay':
                print('NUMERICAL_REVIEW: CPU controller waiting for review-allow.json; fixed UUID, no CUDA context held', flush=True)
                while not (args.output / 'review-allow.json').exists():
                    if monitor.reason:
                        raise RuntimeError(monitor.reason)
                    time.sleep(2)
                review = json.loads((args.output / 'review-allow.json').read_text())
                assert review['proceed'], 'Numerical review did not authorize further experiments'
        else:
            state['status'] = 'COMPLETED'
    finally:
        if monitor.active is not None and monitor.active.poll() is None:
            os.killpg(monitor.active.pid, signal.SIGTERM)
            monitor.active.wait()
        monitor.stop.set()
        thread.join()
        state.update(finished_at=now(), completed_jobs=sum(r.get('returncode') == 0 for r in records),
            planned_jobs=len(plan), stop_reason=monitor.reason,
            release='All our GPU children exited; CPU controller has no CUDA context; no deviceReset')
        if state['status'] == 'RUNNING':
            state['status'] = 'FAILED'
        (args.output / 'after.json').write_text(json.dumps(prior.sample(uuid), indent=2) + '\n')
        (args.output / 'window.json').write_text(json.dumps(state, indent=2) + '\n')
        print(json.dumps(state), flush=True)


if __name__ == '__main__':
    main()
