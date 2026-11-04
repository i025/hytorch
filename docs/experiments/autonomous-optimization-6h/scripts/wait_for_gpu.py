"""Observe idle cards without creating CUDA contexts, then start one fixed window."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def observe():
    fields = 'index,uuid,name,memory.total,memory.used,utilization.gpu'
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=' + fields,
                                   '--format=csv,noheader,nounits'], text=True)
    cards = [dict(zip(fields.split(','), [v.strip() for v in row])) for row in csv.reader(raw.splitlines())]
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory',
                                    '--format=csv,noheader,nounits'], text=True)
    occupied = {row[0].strip() for row in csv.reader(apps.splitlines()) if row}
    return dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), cards=cards,
                occupied_uuids=sorted(occupied), processes_raw=apps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--uuid', help='Keep the same physical device after an interrupted window')
    parser.add_argument('--stable-seconds', type=float, default=90)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    cutoff = datetime(2026, 11, 4, 22, 40, tzinfo=timezone.utc)
    # Leave enough of the six-hour budget to run the prepared plan and archive.
    idle_since = {}
    with args.log.open('x') as log:
        while datetime.now(timezone.utc) < cutoff:
            row = observe()
            log.write(json.dumps(row) + '\n')
            log.flush()
            idle = [c for c in row['cards'] if (args.uuid is None or c['uuid'] == args.uuid)
                    and c['uuid'] not in row['occupied_uuids']
                    and int(c['memory.used']) < 500 and int(c['utilization.gpu']) < 5]
            current = {c['uuid'] for c in idle}
            idle_since = {uuid: started for uuid, started in idle_since.items() if uuid in current}
            for card in idle:
                idle_since.setdefault(card['uuid'], time.monotonic())
            ready = [c for c in idle if time.monotonic() - idle_since[c['uuid']] >= args.stable_seconds]
            if ready:
                card = ready[0]
                print('SELECTED', card['index'], card['uuid'], row['timestamp_utc'], flush=True)
                return subprocess.call([*command, '--uuid', card['uuid']])
            print(row['timestamp_utc'], 'No sustained idle GPU; no CUDA context or memory held', flush=True)
            time.sleep(45)
    print('GPU NOT_RUN: no idle card before the budget cutoff', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
