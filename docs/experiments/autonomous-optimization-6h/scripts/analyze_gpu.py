"""CPU-only matched-run checks, statistics, CUPTI exports and attribution."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from common import ROOT, compare, write_csv, write_json


def read(path):
    return list(csv.DictReader(path.open()))


def js(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--analysis', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    args.analysis.mkdir(parents=True, exist_ok=False)
    state = js(args.raw / 'window.json')
    commands = js(args.raw / 'commands.json')
    raw, all_steps, all_losses, errors = [], [], [], []
    initial = {}
    for job in commands:
        folder = args.raw / job['name']
        if job.get('returncode') != 0 or not (folder / 'config.json').exists():
            continue
        config = js(folder / 'config.json')
        if 'task' not in config:
            continue
        summary = js(folder / 'summary.json')
        model, task = config['model'], config['task']
        method = 'Eprefetch' if config['method'] == 'E' and config['workers'] == 2 else config['method']
        repeat = int(job['name'].rsplit('-r', 1)[1]) if '-r' in job['name'] else 0
        weights = dict(np.load(folder / 'initial_weights.npz'))
        if model not in initial:
            initial[model] = weights
        for key in weights:
            np.testing.assert_array_equal(weights[key], initial[model][key])
        assert config['cuda_visible_devices'] == state['uuid']
        assert config['dtype'] == 'float32' and config['seed'] == 42 and config['dropout'] == 0 and not config['tf32']
        assert config['learning_rate'] == .001 and config['batch'] == 4
        counts = summary['dispatch_counts']
        if method == 'A':
            assert counts == {}, 'A must never schedule our native code'
        if method in ('I', 'P', 'R', 'J'):
            assert counts['conv_dweight_fused'] == (5 if model == 'donkey' else 3) * config['steps']
        if method in ('R', 'J'):
            assert counts['relu_backward_fused'] == (7 if model == 'donkey' else 3) * config['steps']
        steps, losses = lines(folder / 'steps.jsonl'), lines(folder / 'losses.jsonl')
        assert len(steps) == len(losses) == config['steps']
        assert all(np.isfinite(r['loss']) for r in losses)
        row = dict(run=job['name'], task=task, model=model, method=method, repeat=repeat,
            workers=config['workers'], steps=config['steps'], samples=summary['samples'],
            wall_s=summary['wall_s'], samples_per_sec=summary['samples_per_sec'],
            complete_step_ms=summary['wall_s'] / config['steps'] * 1000,
            measured_step_mean_ms=np.mean([s['step_s'] for s in steps]) * 1000,
            final_loss=summary['final_loss'], cupy_reserved_bytes=summary['cupy_pool_reserved_bytes'],
            torch_peak_bytes=summary['torch_peak_allocated_bytes'],
            physical_gpu=state['physical_gpu'], uuid=state['uuid'], dispatch=json.dumps(counts))
        if task == 'phases':
            row.update({p + '_ms': np.mean([s[p + '_s'] for s in steps]) * 1000
                        for p in ('data', 'transfer', 'forward', 'backward', 'update')})
        raw.append(row)
        all_steps.extend(dict(r, run=job['name'], task=task, model=model, method=method, repeat=repeat) for r in steps)
        all_losses.extend(dict(r, run=job['name'], task=task, model=model, method=method, repeat=repeat) for r in losses)
        if task in ('compute', 'io', 'phases'):
            for reference in ('A', 'I', 'E'):
                target = args.raw / f'{task}-{model}-{reference}-r{repeat}'
                if not (target / 'summary.json').exists():
                    continue
                actual, expected = dict(np.load(folder / 'final_weights.npz')), dict(np.load(target / 'final_weights.npz'))
                for key in actual:
                    errors.append(compare(actual[key], expected[key], 'weight/' + key, run=job['name'], reference=reference))
                errors.append(compare(np.array([r['loss'] for r in losses]),
                    np.array([r['loss'] for r in lines(target / 'losses.jsonl')]), 'loss', run=job['name'], reference=reference))

    # Heterogeneous phase columns are blank for non-phase records, never fabricated zeroes.
    fields = list(dict.fromkeys(k for r in raw for k in r))
    write_csv(args.output / 'training-raw.csv', [{k: r.get(k) for k in fields} for r in raw])
    write_csv(args.output / 'steps.csv', all_steps)
    write_csv(args.output / 'losses.csv', all_losses)
    write_csv(args.output / 'numerical-errors.csv', errors)
    stats, gains, paired = [], [], []
    groups = defaultdict(list)
    for row in raw:
        if row['task'] != 'trace':
            groups[(row['model'], row['task'], row['method'])].append(row)
    for (model, task, method), rows in groups.items():
        for metric in ('wall_s', 'samples_per_sec', 'complete_step_ms', *([p + '_ms' for p in ('data', 'transfer', 'forward', 'backward', 'update')] if task == 'phases' else [])):
            values = [r[metric] for r in rows]
            stats.append(dict(model=model, task=task, method=method, metric=metric, repeats=len(values),
                mean=float(np.mean(values)), sd=float(np.std(values, ddof=1)) if len(values) > 1 else None,
                min=min(values), max=max(values), values=json.dumps(values)))
    for model in ('donkey', 'spatial'):
        for task in ('compute', 'io'):
            for new, old in [('F', 'A'), ('I', 'F'), ('P', 'I'), ('R', 'I'), ('J', 'I'), ('J', 'A'), ('J', 'E'), ('J', 'Eprefetch')]:
                aa, bb = groups[(model, task, new)], groups[(model, task, old)]
                if len(aa) < 3 or len(bb) < 3:
                    continue
                ratio = np.mean([r['samples_per_sec'] for r in aa]) / np.mean([r['samples_per_sec'] for r in bb])
                speed = np.mean([r['wall_s'] for r in bb]) / np.mean([r['wall_s'] for r in aa])
                gains.append(dict(model=model, task=task, comparison=new + '/' + old,
                    throughput_ratio=ratio, throughput_gain_pct=(ratio - 1) * 100,
                    wall_speedup=speed, wall_reduction_pct=(1 - 1 / speed) * 100,
                    repeats_new=len(aa), repeats_old=len(bb),
                    all_new_above_old_range=min(r['samples_per_sec'] for r in aa) > max(r['samples_per_sec'] for r in bb)))
                for a in aa:
                    b = next(r for r in bb if r['repeat'] == a['repeat'])
                    paired.append(dict(model=model, task=task, comparison=new + '/' + old, repeat=a['repeat'],
                        throughput_ratio=a['samples_per_sec'] / b['samples_per_sec'],
                        wall_speedup=b['wall_s'] / a['wall_s']))
    write_csv(args.output / 'timing-summary.csv', stats)
    write_csv(args.output / 'speedups.csv', gains)
    write_csv(args.output / 'paired-speedups.csv', paired)

    traces, micro = [], []
    for report in sorted(args.raw.glob('*.nsys-rep')):
        name = report.stem
        database = args.analysis / (name + '.sqlite')
        parsed = args.analysis / name
        if name == 'trace-relu-micro':
            analyzer = ROOT / 'docs/experiments/conv-dweight-fusion/scripts/analyze_micro_trace.py'
            options = ['--expected-cases', '12']
        else:
            analyzer = ROOT / 'docs/experiments/donkeynet-backward-profiling/scripts/analyze_trace.py'
            options = ['--label', name]
        for stage, command in [('export', ['nsys', 'export', '--type', 'sqlite', '--output', str(database), str(report)]),
                               ('parse', [sys.executable, str(analyzer), '--input', str(database), '--output', str(parsed), *options])]:
            with (args.analysis / (name + '-' + stage + '.log')).open('x') as log:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            with (args.analysis / 'commands.jsonl').open('a') as log:
                log.write(json.dumps(dict(command=command, returncode=completed.returncode)) + '\n')
            completed.check_returncode()
        if name == 'trace-relu-micro':
            micro.extend(read(parsed / 'micro-kernels.csv'))
        else:
            assert js(parsed / 'profiling-summary.json')['unmatched_kernel_api_count'] == 0
            traces.extend(dict(r, run=name) for r in read(parsed / 'phase-summary.csv'))
    write_csv(args.output / 'trace-phases.csv', traces)
    write_csv(args.output / 'micro-kernel-times.csv', micro)
    write_json(args.output / 'summary.json', dict(status='COMPLETED', window_status=state['status'],
        measured_runs=len(raw), formal_runs=sum(r['task'] in ('compute', 'io') for r in raw),
        numerical_groups=len(errors), numerical_failed_groups=sum(r['mismatches'] > 0 for r in errors),
        maximum_error=max(r['max_abs'] for r in errors) if errors else None,
        traces=len(list(args.raw.glob('*.nsys-rep'))), physical_gpu=state['physical_gpu'], uuid=state['uuid'],
        confidence_scope='At least three independent process repeats; ranges and all pairwise repeats retained, no claimed cross-window stability'))


if __name__ == '__main__':
    main()
