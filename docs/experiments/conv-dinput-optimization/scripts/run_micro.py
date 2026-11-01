"""Rotate implementations in independent cross-shape benchmark processes."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[4]
SCRIPT=Path(__file__).with_name('bench_shapes.py')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--correctness',type=Path,required=True)
    p.add_argument('--traces',action='store_true')
    a=p.parse_args();assert json.loads(a.correctness.read_text())['status']=='COMPLETED'
    a.output.mkdir(parents=True,exist_ok=False);records=[]
    for scope in ('full','kernel'):
        methods=['original','fused','fused_opt','torch'] if scope=='full' else ['fused','fused_opt']
        for repeat in range(1,2 if a.traces and scope=='full' else 4):
            for split in ('history','fresh'):
                for method in methods[(repeat-1)%len(methods):]+methods[:(repeat-1)%len(methods)]:
                    name=f'{scope}-{split}-{method}-r{repeat}'
                    command=[sys.executable,str(SCRIPT),'--method',method,'--split',split,'--scope',scope,
                        '--repeat',str(repeat),'--warmup','5','--iterations','3' if a.traces and scope=='full' else '30','--output',str(a.output/name)]
                    jobs=[(name,command)]
                    if a.traces:
                        command+=['--nvtx'];prefix=a.output/name
                        jobs=[(name,['nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','-o',str(prefix),*command]),
                            (name+'-export',['nsys','export','--type','sqlite','--output',str(prefix)+'.sqlite',str(prefix)+'.nsys-rep'])]
                    for label,argv in jobs:
                        row=dict(name=label,command=argv,started_at=datetime.now(timezone.utc).isoformat());records.append(row)
                        (a.output/'commands.json').write_text(json.dumps(records,indent=2)+'\n');print('START '+label,flush=True)
                        with (a.output/(label+'.log')).open('x') as f:done=subprocess.run(argv,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
                        row.update(returncode=done.returncode,finished_at=datetime.now(timezone.utc).isoformat())
                        (a.output/'commands.json').write_text(json.dumps(records,indent=2)+'\n');done.check_returncode()
    (a.output/'completion.json').write_text(json.dumps(dict(status='COMPLETED',commands=len(records),traces=a.traces),indent=2)+'\n')


if __name__=='__main__':main()
