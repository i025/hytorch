"""Wait up to two hours for one idle A100, then run the prepared guarded batch."""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[5]

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--seconds', type=float, default=7200)
p.add_argument('--interval', type=float, default=20)
p.add_argument('--library', type=Path, required=True)
p.add_argument('--gpu-uuid', help='Keep profiling on the device used for formal measurements')
p.add_argument('--trace-only', action='store_true')
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
started = datetime.now(timezone.utc)
deadline = time.monotonic() + a.seconds
config = dict(started_at=started.isoformat(), max_wait_seconds=a.seconds,
              interval_seconds=a.interval, library=str(a.library.resolve()),
              gpu_uuid=a.gpu_uuid, trace_only=a.trace_only,
              pid=os.getpid(), command=sys.argv)
(a.output/'config.json').write_text(json.dumps(config, indent=2)+'\n')
attempt = 0
with (a.output/'snapshots.jsonl').open('x') as snapshots:
    while time.monotonic() < deadline:
        fields = 'index,uuid,name,memory.used,utilization.gpu'
        raw = subprocess.check_output(['nvidia-smi', '--query-gpu='+fields,
            '--format=csv,noheader,nounits'], text=True)
        clients = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
            '--format=csv,noheader,nounits'], text=True)
        gpus = [dict(zip(fields.split(','), [v.strip() for v in r])) for r in csv.reader(raw.splitlines())]
        processes = [dict(zip(('uuid','pid'), [v.strip() for v in r])) for r in csv.reader(clients.splitlines())]
        occupied = {r['uuid'] for r in processes}
        idle = [g for g in gpus if 'A100' in g['name'] and (a.gpu_uuid is None or g['uuid']==a.gpu_uuid)
                and g['uuid'] not in occupied
                and float(g['memory.used']) == 0 and float(g['utilization.gpu']) == 0]
        row = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), gpus=gpus,
                   processes=processes, idle_candidates=[g['uuid'] for g in idle])
        snapshots.write(json.dumps(row)+'\n'); snapshots.flush()
        if idle:
            gpu = idle[0]
            attempt += 1
            out = a.output/('gpu-attempt-%02d' % attempt)
            guard = a.output/('guard-attempt-%02d' % attempt)
            command = [sys.executable, str(ROOT/'docs/experiments/donkeynet-backward-profiling/scripts/run_gpu.py'),
                '--gpu', gpu['uuid'], '--record', str(guard), '--', sys.executable,
                str(ROOT/'docs/experiments/conv-backward-fusion/scripts/run_experiments.py'),
                '--output', str(out), '--library', str(a.library.resolve()), '--gpu-uuid', gpu['uuid']]
            if a.trace_only:command+=['--only','traces']
            (a.output/('attempt-%02d.json' % attempt)).write_text(json.dumps(dict(
                physical_gpu=gpu['index'], uuid=gpu['uuid'], logical_gpu=0, command=command,
                selected_at=datetime.now(timezone.utc).isoformat()), indent=2)+'\n')
            print('IDLE GPU '+gpu['index']+' '+gpu['uuid']+'; starting guarded batch', flush=True)
            with (a.output/('attempt-%02d.log' % attempt)).open('x') as log:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if not (guard/'launch.json').exists():
                print('Device changed before guard launch; continuing wait', flush=True)
                continue
            status = 'COMPLETED' if result.returncode == 0 else 'INTERRUPTED' if result.returncode == 75 else 'FAILED'
            record = dict(status=status, returncode=result.returncode, raw_output=str(out.resolve()),
                guard=str(guard.resolve()), gpu=gpu, finished_at=datetime.now(timezone.utc).isoformat())
            (a.output/'completion.json').write_text(json.dumps(record, indent=2)+'\n')
            print(json.dumps(record), flush=True)
            raise SystemExit(result.returncode)
        time.sleep(min(a.interval, max(0, deadline-time.monotonic())))
(a.output/'completion.json').write_text(json.dumps(dict(status='NOT_RUN',
    reason='No idle A100 during the two-hour monitoring window',
    finished_at=datetime.now(timezone.utc).isoformat()), indent=2)+'\n')
print('NOT_RUN: two-hour monitoring window ended without an idle A100', flush=True)
