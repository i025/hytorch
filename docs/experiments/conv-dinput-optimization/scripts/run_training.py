"""Matched A/D/F/H/E processes, separately for throughput and synchronized phases."""
import argparse
from datetime import datetime,timezone
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[4]
spec=importlib.util.spec_from_file_location('stage4e_runner',ROOT/'docs/experiments/conv-shape-dispatch/scripts/run_validation.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
previous.CONFIGS=dict(previous.CONFIGS)
previous.CONFIGS['H']=[*previous.CONFIGS['D'],'--conv-backward','fused_opt']
previous.SCRIPTS=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu-uuid',required=True)
    p.add_argument('--library',type=Path,required=True);p.add_argument('--traces',action='store_true')
    p.add_argument('--correctness',type=Path,required=True)
    a=p.parse_args();assert json.loads(a.correctness.read_text())['status']=='COMPLETED'
    a.output.mkdir(parents=True,exist_ok=False);jobs=[];methods=list('ADFHE')
    if a.traces:
        for model in ('donkey','spatial'):
            for method in 'AFH':
                name='trace-training-'+model+'-'+method
                jobs.extend(previous.trace_jobs(a,name,previous.training(a,name,model,method,'throughput',True),method))
    else:
        for mode in ('throughput','phases'):
            for repeat in range(1,4):
                order=methods[repeat-1:]+methods[:repeat-1]
                for model in ('donkey','spatial'):
                    for method in order:
                        name=f'training-{model}-{mode}-{method}-r{repeat}'
                        jobs.append((name,previous.training(a,name,model,method,mode,False),method))
    records=[]
    for name,command,method in jobs:
        env=dict(os.environ,HYTORCH_CUDA_LIBRARY=str(a.library.resolve()))
        if method=='A':env['HYTORCH_CUDA_LIBRARY']=str(a.output/'absent_native_library.so')
        row=dict(name=name,command=command,library=env['HYTORCH_CUDA_LIBRARY'],started_at=datetime.now(timezone.utc).isoformat())
        records.append(row);(a.output/'commands.json').write_text(json.dumps(records,indent=2)+'\n')
        print('START '+name,flush=True)
        with (a.output/(name+'.log')).open('x') as f:done=subprocess.run(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        row.update(returncode=done.returncode,finished_at=datetime.now(timezone.utc).isoformat())
        (a.output/'commands.json').write_text(json.dumps(records,indent=2)+'\n');done.check_returncode()
    (a.output/'completion.json').write_text(json.dumps(dict(status='COMPLETED',commands=len(records)),indent=2)+'\n')


if __name__=='__main__':main()
