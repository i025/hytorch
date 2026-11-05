"""Future opt-in single-device controller. Never run during the CPU-only stage."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--uuid',required=True)
    p.add_argument('--run-gpu',action='store_true',required=True)
    a = p.parse_args()
    plan = json.loads(a.plan.read_text())
    out = Path(plan['output'])
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, **plan['environment'], CUDA_VISIBLE_DEVICES=a.uuid,
               CUDA_DEVICE_ORDER='PCI_BUS_ID', NVIDIA_TF32_OVERRIDE='0')
    events = []
    def write():
        (out/'window.json').write_text(json.dumps(dict(uuid=a.uuid,logical_device=0,tf32_override='0',events=events,
            status='STOPPED' if events and events[-1].get('returncode',0) else 'RECORDED'),indent=2)+'\n')
    for task in plan['commands']:
        snapshot = subprocess.check_output(['nvidia-smi','--id='+a.uuid,
            '--query-gpu=index,uuid,name,memory.used,memory.total','--format=csv,noheader'],text=True)
        processes = subprocess.check_output(['nvidia-smi','--id='+a.uuid,
            '--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
        event = dict(task=task['name'],started=datetime.now(timezone.utc).isoformat(),
            gpu=snapshot,processes=processes)
        events.append(event);write()
        # This controller has no CUDA context. A foreign compute process aborts, not preempts.
        if processes.strip():
            event.update(returncode=1,reason='Device acquired by another process; stop, do not switch')
            write()
            raise RuntimeError(event['reason'])
        with (out/(task['name']+'.log')).open('x') as f:
            result = subprocess.run(task['command'],cwd=plan['cwd'],env=env,stdout=f,stderr=subprocess.STDOUT)
        event.update(returncode=result.returncode,finished=datetime.now(timezone.utc).isoformat(),
            command=task['command'])
        write()
        if result.returncode:
            sys.exit(result.returncode)
    events.append(dict(finished=datetime.now(timezone.utc).isoformat(),
        reason='All sequential child processes exited; CUDA contexts released. No device switch.'))
    write()


if __name__ == '__main__':
    main()
