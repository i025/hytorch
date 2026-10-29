"""Wait for an idle A100, then launch one recorded experiment through the guard."""
import argparse
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[4]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--seconds',type=float,default=3600)
p.add_argument('--idle-memory-mib',type=float,default=0)
p.add_argument('command',nargs=argparse.REMAINDER)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
command=a.command[1:] if a.command[:1]==['--'] else a.command
(a.output/'config.json').write_text(json.dumps(dict(command=command,seconds=a.seconds,
    idle_memory_mib=a.idle_memory_mib,started_at=datetime.now(timezone.utc).isoformat()),indent=2)+'\n')
deadline=time.monotonic()+a.seconds;attempt=0
with (a.output/'snapshots.jsonl').open('x') as log:
    while time.monotonic()<deadline:
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        gpus=[dict(zip(('index','uuid','name','memory','utilization'),[s.strip() for s in r])) for r in csv.reader(raw.splitlines())]
        raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
        clients=[r for r in csv.reader(raw.splitlines())]
        occupied={r[0].strip() for r in clients}
        idle=[g for g in gpus if 'A100' in g['name'] and g['uuid'] not in occupied and float(g['memory'])<=a.idle_memory_mib and float(g['utilization'])==0]
        log.write(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(),gpus=gpus,clients=clients,idle=idle))+'\n');log.flush()
        if not idle:time.sleep(min(20,max(0,deadline-time.monotonic())));continue
        gpu=idle[0];attempt+=1;folder=a.output/('attempt-%02d'%attempt)
        args=[s.replace('{gpu}',gpu['uuid']).replace('{output}',str((folder/'results').resolve())) for s in command]
        launch=[sys.executable,str(Path(__file__).with_name('run_gpu.py')),
            '--gpu',gpu['uuid'],'--idle-memory-mib',str(a.idle_memory_mib),'--record',str(folder/'guard'),'--',*args]
        print('Starting on physical GPU '+gpu['index']+' '+gpu['uuid'],flush=True)
        with (a.output/('attempt-%02d.log'%attempt)).open('x') as f:done=subprocess.run(launch,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        if not (folder/'guard/launch.json').exists():continue
        record=dict(status='COMPLETED' if done.returncode==0 else 'INTERRUPTED' if done.returncode==75 else 'FAILED',
            returncode=done.returncode,gpu=gpu,raw=str((folder/'results').resolve()),
            finished_at=datetime.now(timezone.utc).isoformat())
        (a.output/'completion.json').write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(record),flush=True);raise SystemExit(done.returncode)
(a.output/'completion.json').write_text(json.dumps(dict(status='NOT_RUN',reason='No idle A100 within this waiting interval'),indent=2)+'\n')
