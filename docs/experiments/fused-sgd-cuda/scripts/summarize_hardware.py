"""Summarize coarse outer-guard samples within full child-process windows."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--commands', type=Path, required=True)
p.add_argument('--monitor', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
commands = json.loads(a.commands.read_text())
samples = [json.loads(s) for s in a.monitor.read_text().splitlines()]
rows = []
for command in commands:
    start, end = (datetime.fromisoformat(command[k]) for k in ('started_at', 'finished_at'))
    selected = [s for s in samples if start <= datetime.fromisoformat(s['timestamp_utc']) <= end]
    active = [s for s in selected if s['owned_process_ids']]
    for scope, group in [('full_process_window', selected), ('samples_with_owned_gpu_client', active)]:
        row = dict(command=command['name'], scope=scope, samples=len(group),
                   external_client_samples=sum(bool(s['external_process_ids']) for s in group))
        for field in ['memory.used', 'utilization.gpu', 'clocks.sm', 'power.draw', 'temperature.gpu']:
            values = [float(s['gpu'][field]) for s in group]
            row[field+'_mean'] = sum(values)/len(values) if values else None
            row[field+'_min'] = min(values) if values else None
            row[field+'_max'] = max(values) if values else None
        rows.append(row)
with a.output.open('x', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print('Saved', len(rows), 'process-window summaries from', len(samples), 'samples; not phase-level hardware utilization.')
