"""Bounded read-only idle-GPU watch. No reservation and no GPU computation."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time


def snapshot():
    gpu_text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,utilization.gpu',
                                     '--format=csv,noheader,nounits'],text=True)
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid',
                                      '--format=csv,noheader,nounits'],text=True).splitlines()
    return [dict(index=int(row[0]),uuid=row[1].strip(),name=row[2].strip(),
                 memory_used_mib=int(row[3]),utilization_pct=int(row[4]),
                 compute_process_count=processes.count(row[1].strip()))
            for row in csv.reader(gpu_text.splitlines())]


def idle(row):
    return row['memory_used_mib']==0 and row['utilization_pct']==0 and row['compute_process_count']==0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deadline',required=True,help='UTC ISO-8601 end of monitoring window')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--on-idle', nargs=argparse.REMAINDER, help='Command to run immediately; {gpu_uuid} is replaced')
    args=parser.parse_args()
    deadline=datetime.fromisoformat(args.deadline).timestamp()
    args.output.mkdir(parents=True,exist_ok=False)
    with (args.output/'config.json').open('x') as stream:
        json.dump(dict(started_at=datetime.now(timezone.utc).isoformat(),deadline=args.deadline,
                       interval_s=30,confirmation_s=10),stream,indent=2)
    with (args.output/'observations.jsonl').open('x') as log:
        while time.time()<deadline:
            rows=snapshot()
            log.write(json.dumps(dict(timestamp=time.time(),gpus=rows))+'\n'); log.flush()
            candidates=[r for r in rows if idle(r)]
            if candidates:
                time.sleep(min(10,max(0,deadline-time.time())))
                confirmed=snapshot()
                log.write(json.dumps(dict(timestamp=time.time(),gpus=confirmed))+'\n'); log.flush()
                ready=[r for r in confirmed if idle(r) and r['uuid'] in {c['uuid'] for c in candidates}]
                if ready and time.time()<deadline:
                    result=dict(status='IDLE_FOUND',gpu=ready[0],observed_at=datetime.now(timezone.utc).isoformat())
                    break
            time.sleep(min(30,max(0,deadline-time.time())))
        else:
            result=dict(status='TIMEOUT_NO_IDLE_GPU',finished_at=datetime.now(timezone.utc).isoformat())
    with (args.output/'result.json').open('x') as stream: json.dump(result,stream,indent=2)
    print(json.dumps(result),flush=True)
    if result['status']=='IDLE_FOUND' and args.on_idle:
        command=[part.replace('{gpu_uuid}',result['gpu']['uuid']) for part in args.on_idle]
        return subprocess.run(command).returncode
    return 0


if __name__=='__main__':
    raise SystemExit(main())
