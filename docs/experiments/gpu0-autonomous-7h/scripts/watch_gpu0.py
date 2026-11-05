"""60-second GPU0 monitor and one-window dispatcher; no CUDA context or reservation."""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil


def snapshot(uuid):
    fields = 'index,uuid,name,memory.used,memory.free,memory.total,utilization.gpu,temperature.gpu,power.draw'
    result = subprocess.check_output(['nvidia-smi','--id='+uuid,'--query-gpu='+fields,
        '--format=csv,noheader,nounits'],text=True,timeout=8)
    gpu = next(csv.reader(result.splitlines(),skipinitialspace=True))
    processes = subprocess.check_output(['nvidia-smi','--id='+uuid,
        '--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
        '--format=csv,noheader,nounits'],text=True,timeout=8)
    apps = [dict(uuid=r[0],pid=int(r[1]),name=r[2],memory_mib=r[3])
        for r in csv.reader(processes.splitlines(),skipinitialspace=True)]
    assert int(gpu[0]) == 0 and gpu[1] == uuid and 'A100' in gpu[2]
    return dict(physical_index=int(gpu[0]),uuid=gpu[1],name=gpu[2],memory_used_mib=int(gpu[3]),
        memory_free_mib=int(gpu[4]),memory_total_mib=int(gpu[5]),utilization_pct=int(gpu[6]),
        temperature_c=gpu[7],power_w=gpu[8],processes=apps)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--schedule',type=Path,required=True)
    a=p.parse_args();config=json.loads(a.schedule.read_text())
    raw=Path(config['raw']);report=Path(config['report'])
    deadline=datetime.fromisoformat(config['deadline_utc']).timestamp()
    cutoff=datetime.fromisoformat(config['gpu_cutoff_utc']).timestamp()
    child=None;window_started=False;streak=0
    def state(**extra):
        value=dict(monitor_pid=os.getpid(),last_update_utc=datetime.now(timezone.utc).isoformat(),
            idle_streak=streak,window_started=window_started,child_pid=child.pid if child else None,
            deadline_utc=config['deadline_utc'],**extra)
        (raw/'monitor-state.json').write_text(json.dumps(value,indent=2)+'\n')
    with (report/'gpu-monitor.csv').open('x',newline='') as f:
        writer=None
        while time.time()<deadline:
            tick=time.monotonic();now=datetime.now(timezone.utc)
            gpu=snapshot(config['uuid'])
            own={os.getpid()}|{v.pid for v in psutil.Process().children(recursive=True)}
            foreign=[v for v in gpu['processes'] if v['pid'] not in own]
            idle=not gpu['processes'] and gpu['memory_used_mib']<=config['idle_memory_mib'] and gpu['utilization_pct']<=config['idle_utilization_pct']
            streak=streak+1 if idle else 0
            row=dict(utc=now.isoformat(),beijing=now.astimezone(timezone(timedelta(hours=8))).isoformat(),
                **{k:v for k,v in gpu.items() if k!='processes'},processes=json.dumps(gpu['processes']),
                foreign_processes=json.dumps(foreign),idle=idle,idle_streak=streak,window_started=window_started)
            if writer is None:
                writer=csv.DictWriter(f,fieldnames=list(row),lineterminator='\n');writer.writeheader()
            writer.writerow(row);f.flush()
            if child and child.poll() is not None:
                state(status='GPU_WINDOW_FINISHED',returncode=child.returncode)
                child=None
            if child and foreign:
                (raw/'resource-conflict.json').write_text(json.dumps(row,indent=2)+'\n')
            plan=raw/'gpu-plan.json'
            if not window_started and streak>=config['idle_samples'] and time.time()<cutoff and plan.exists():
                window_started=True
                with (raw/'window-controller.log').open('x') as log:
                    child=subprocess.Popen([sys.executable,str(Path(__file__).with_name('run_window.py')),
                        '--schedule',str(a.schedule.resolve()),'--plan',str(plan)],
                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                state(status='GPU_WINDOW_STARTED')
            else:
                state(status='GPU_RUNNING' if child else 'MONITORING')
            time.sleep(max(0,min(config['interval_s']-(time.monotonic()-tick),deadline-time.time())))
    state(status='DEADLINE_REACHED')


if __name__=='__main__':
    main()
