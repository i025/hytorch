"""Archive historical training inputs to analysis, plus the rejected GPU launch."""
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[4]
DOC = ROOT/'docs/experiments/donkeynet-backward-profiling'
OLD = ROOT/'docs/experiments/a100-gpu-window/training-comparison'
records = []
for mode, repeats in [('throughput', 3), ('phases', 1)]:
    for method in ['cupy', 'combined', 'torch']:
        for repeat in range(1, repeats+1):
            name = f'{method}_r{repeat}'
            source = OLD/f'donkey-{mode}-fixed-01'/name
            destination = DOC/'results/historical-training'/mode/name
            destination.mkdir(parents=True, exist_ok=False)
            for filename in ['config.json', 'summary.json', 'steps.jsonl', 'losses.jsonl', 'hardware.jsonl']:
                shutil.copy2(source/filename, destination/filename)
            records.append(dict(mode=mode, method=method, repeat=repeat,
                                source=str(source.relative_to(ROOT)),
                                archive=str(destination.relative_to(ROOT))))
(DOC/'results/historical-training/sources.json').write_text(json.dumps(records, indent=2))
source = ROOT/'runs/donkeynet_4a_measurement_monitor_01'
destination = DOC/'logs/measurement-attempt-01'
destination.mkdir(parents=True, exist_ok=False)
snapshots = json.loads((source/'before.json').read_text())
public = [{**{k:v for k,v in row.items() if k!='process_ids'},
           'compute_process_count':len(row['process_ids'])} for row in snapshots]
(destination/'before.json').write_text(json.dumps(public, indent=2))
attempt = dict(status='NOT_RUN', reason='GPU 6 became occupied before the launch guard completed',
               guard_returncode=1, gpu_work_started=False,
               stdout='BLOCKED: selected GPU is not idle; no GPU work started\n',
               provenance='Return code and stdout copied from the actual execution result; no child launch.json exists because the guard rejected launch',
               command=['python','docs/experiments/donkeynet-backward-profiling/scripts/run_gpu.py',
                        '--gpu','GPU-d40fdd6f-33be-8ccc-fc58-f4daf39a4b82',
                        '--record','runs/donkeynet_4a_measurement_monitor_01','--','python',
                        'docs/experiments/donkeynet-backward-profiling/scripts/run_measurements.py',
                        '--gpu-uuid','GPU-d40fdd6f-33be-8ccc-fc58-f4daf39a4b82',
                        '--output','runs/donkeynet_4a_measurements_01'])
(destination/'result.json').write_text(json.dumps(attempt, indent=2))
(destination/'terminal.log').write_text(attempt['stdout'])
print(f'Archived {len(records)} historical training runs and the rejected launch; no GPU computation or checksums')
