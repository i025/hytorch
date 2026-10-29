"""Three independent, rotating cross-shape benchmark repeats."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[4]
SCRIPTS=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--split',choices=['calibration','held-out','validation-2','validation-final'],required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--auto',action='store_true')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
records=[];methods=['original','fused','torch']+(['auto'] if a.auto else [])
for repeat in range(1,4):
    for method in methods[repeat-1:]+methods[:repeat-1]:
        name=method+'-r'+str(repeat)
        command=[sys.executable,str(SCRIPTS/'bench_shapes.py'),'--split',a.split,'--method',method,
            '--repeat',str(repeat),'--warmup','5','--iterations','30','--output',str(a.output/name)]
        row=dict(name=name,command=command,started_at=datetime.now(timezone.utc).isoformat());records.append(row)
        (a.output/'commands.json').write_text(json.dumps(records,indent=2)+'\n')
        print('START '+name,flush=True)
        with (a.output/(name+'.log')).open('x') as f:done=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        row.update(returncode=done.returncode,finished_at=datetime.now(timezone.utc).isoformat())
        (a.output/'commands.json').write_text(json.dumps(records,indent=2)+'\n');done.check_returncode()
(a.output/'completion.json').write_text(json.dumps(dict(status='COMPLETED',commands=len(records)),indent=2)+'\n')
