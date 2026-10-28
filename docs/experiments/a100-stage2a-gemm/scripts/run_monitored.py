"""Run a GEMM command on one currently idle GPU and retain load snapshots."""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def query(gpu, fields, kind='gpu'):
    return subprocess.check_output(['nvidia-smi', '--id='+gpu,
        '--query-'+kind+'='+fields,'--format=csv,noheader,nounits'],text=True).strip()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',required=True,help='Physical UUID selected after checking occupancy')
    parser.add_argument('--record',type=Path,required=True)
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    command=args.command[1:] if args.command[:1]==['--'] else args.command
    if not command: parser.error('A benchmark command is required')
    args.record.mkdir(parents=True,exist_ok=False)
    processes=query(args.gpu,'pid,used_memory','compute-apps')
    info=query(args.gpu,'index,uuid,memory.used,utilization.gpu')
    with (args.record/'before.json').open('x') as stream:
        json.dump(dict(gpu=info,processes=processes),stream,indent=2)
    row=next(csv.reader([info]))
    if processes or float(row[2])!=0 or float(row[3])!=0:
        raise SystemExit('Selected GPU is not idle; no benchmark was started')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpu,CUDA_DEVICE_ORDER='PCI_BUS_ID')
    with (args.record/'launch.json').open('x') as stream:
        json.dump(dict(command=command,physical_gpu=int(row[0]),logical_gpu=0,
            uuid=args.gpu,started_at=datetime.now(timezone.utc).isoformat(),
            environment={k:env.get(k) for k in ('CUDA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER',
                'CUPY_TF32','NVIDIA_TF32_OVERRIDE','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS')}),stream,indent=2)
    with (args.record/'benchmark.log').open('x') as log, (args.record/'gpu-load.jsonl').open('x') as monitor:
        proc=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        fields='index,uuid,memory.used,utilization.gpu,clocks.sm,clocks.mem,power.draw,temperature.gpu'
        try:
            while True:
                monitor.write(json.dumps(dict(timestamp=time.time(),benchmark_pid=proc.pid,
                    gpu=query(args.gpu,fields),gpu_fields=fields,
                    processes=query(args.gpu,'pid,used_memory','compute-apps')))+'\n')
                monitor.flush()
                if proc.poll() is not None: break
                time.sleep(1)
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait()
        code=proc.wait()
    with (args.record/'completion.json').open('x') as stream:
        json.dump(dict(returncode=code,finished_at=datetime.now(timezone.utc).isoformat()),stream,indent=2)
    print((args.record/'benchmark.log').read_text(),end='')
    return code


if __name__=='__main__':
    sys.exit(main())
