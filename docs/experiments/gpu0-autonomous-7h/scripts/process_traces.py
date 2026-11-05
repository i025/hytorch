"""After GPU children exit, export and analyze actual Nsight data on CPU."""
import argparse
import csv
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--schedule',type=Path,required=True)
    p.add_argument('--wait-window',action='store_true')
    a=p.parse_args();cfg=json.loads(a.schedule.read_text());raw=Path(cfg['raw'])
    gpu=raw/'gpu-window-01';state=gpu/'window.json'
    limit=datetime.fromisoformat(cfg['deadline_utc']).timestamp()-600
    while a.wait_window and time.time()<limit:
        if state.exists():
            value=json.loads(state.read_text())
            if 'finished_utc' in value:break
        time.sleep(20)
    if not state.exists() or 'finished_utc' not in json.loads(state.read_text()):
        (raw/'trace-analysis-status.json').write_text(json.dumps(dict(status='NOT_RUN',
            reason='No completed/closed GPU window before CPU analysis cutoff',timestamp=datetime.now(timezone.utc).isoformat()),indent=2)+'\n')
        return
    out=gpu/'offline-traces-01';out.mkdir(exist_ok=False)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    records=[]
    def run(name,command):
        row=dict(name=name,command=[str(v) for v in command],started_utc=datetime.now(timezone.utc).isoformat())
        records.append(row)
        with (out/(name+'.log')).open('x') as f:
            try:
                result=subprocess.run(row['command'],env=env,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,
                    timeout=max(1,min(180,int(limit-time.time()))))
                code=result.returncode
            except subprocess.TimeoutExpired:
                code=124;row['reason']='CPU export/analysis time budget exhausted'
        row.update(returncode=code,finished_utc=datetime.now(timezone.utc).isoformat())
        (out/'commands.json').write_text(json.dumps(records,indent=2)+'\n')
        return code
    for report in sorted(gpu.glob('*.nsys-rep')):
        if time.time()+60>limit:break
        stem=report.name[:-9];database=out/(stem+'.sqlite')
        if run('export-'+stem,['nsys','export','--type','sqlite','--output',database,report]):continue
        if stem=='trace-sum':
            rows=list(csv.DictReader((gpu/'trace-sum-data/timings.csv').open()))
            cases=len({(r['case'],r['repeat'],r['method']) for r in rows})
        elif stem=='trace-relu':cases=12
        else:cases=None
        if cases is not None:
            command=[sys.executable,ROOT/'docs/experiments/conv-dweight-fusion/scripts/analyze_micro_trace.py',
                '--input',database,'--output',out/(stem+'-analysis'),'--expected-cases',str(cases)]
        else:
            command=[sys.executable,HERE/'analyze_phases.py','--input',database,'--output',out/(stem+'-analysis')]
        run('analyze-'+stem,command)
    value=dict(status='NOT_RUN' if not records else 'COMPLETED' if all(r['returncode']==0 for r in records) else 'ANALYSIS_FAILURES_RETAINED',
        commands=len(records),gpu_compute='NONE: CPU export and SQL only',finished_utc=datetime.now(timezone.utc).isoformat())
    (raw/'trace-analysis-status.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value),flush=True)


if __name__=='__main__':main()
