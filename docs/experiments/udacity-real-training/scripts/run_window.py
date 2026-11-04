"""One CPU controller for the explicitly authorized shared physical GPU1 window."""
import argparse
import csv
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]


def now():return datetime.now(timezone.utc).isoformat()


def sample(uuid='1'):
    fields='index,uuid,name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.sm,clocks.mem,driver_version'
    command=['nvidia-smi','--id='+uuid,'--query-gpu='+fields,'--format=csv,noheader,nounits']
    values=next(csv.reader(subprocess.check_output(command,text=True,timeout=5).splitlines()))
    gpu=dict(zip(fields.split(','),[s.strip() for s in values]))
    command=['nvidia-smi','--id='+gpu['uuid'],'--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader,nounits']
    rows=list(csv.reader(subprocess.check_output(command,text=True,timeout=5).splitlines()))
    processes=[dict(pid=int(r[0]),name=r[1].strip(),memory_mib=r[2].strip()) for r in rows if r and r[0].strip().isdigit()]
    return dict(timestamp_utc=now(),gpu=gpu,processes=processes)


def jobs(a):
    common=['--prepared',str(a.prepared),'--no-fingerprints']
    result=[dict(name='correctness-real',method=None,command=[sys.executable,str(HERE/'check_correctness.py'),
        '--prepared',str(a.prepared),'--output',str(a.output/'correctness-real')])]
    for method in 'AFIE':
        name='smoke-'+method
        result.append(dict(name=name,method=method,command=[sys.executable,str(HERE/'train.py'),'--method',method,'--task','smoke',*common,'--output',str(a.output/name)]))
    for task in ('compute','io','phases'):
        for repeat in range(1,4):
            order='AFIE'[repeat-1:]+'AFIE'[:repeat-1]
            for method in order:
                name=f'{task}-{method}-r{repeat}'
                result.append(dict(name=name,method=method,command=[sys.executable,str(HERE/'train.py'),'--method',method,'--task',task,*common,'--output',str(a.output/name)]))
    for method in 'FIEA':
        name='train-'+method
        result.append(dict(name=name,method=method,command=[sys.executable,str(HERE/'train.py'),'--method',method,'--task','train',*common,'--output',str(a.output/name)]))
    for method in 'FI':
        name='trace-'+method
        result.append(dict(name=name,method=method,command=['nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none',
            '-o',str(a.output/name),sys.executable,str(HERE/'train.py'),'--method',method,'--task','trace',*common,'--output',str(a.output/name)]))
    return result


class SharedMonitor:
    def __init__(self,path,baseline):
        self.path,self.baseline=path,baseline;self.uuid=baseline['gpu']['uuid']
        self.original={r['pid'] for r in baseline['processes']}
        self.stop=threading.Event();self.active=None;self.reason=None;self.job='baseline';self.high_since=None

    def run(self):
        with self.path.open('x') as log:
            while not self.stop.is_set():
                try:row=sample(self.uuid)
                except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as error:
                    self.reason='MONITOR_FAILED: '+str(error);break
                row['job']=self.job;current={r['pid'] for r in row['processes']}
                row['original_pids_present']=sorted(self.original&current)
                row['original_pids_missing']=sorted(self.original-current)
                own=[]
                if self.active is not None:
                    for pid in current:
                        try:
                            if os.getpgid(pid)==self.active.pid:own.append(pid)
                        except ProcessLookupError:pass  # A sampled process exited before this read.
                row['own_gpu_pids']=own
                # pmon exposes per-process SM/memory activity where supported;
                # its raw output is kept even when the driver returns N/A.
                try:pmon=subprocess.run(['nvidia-smi','pmon','-i','1','-s','um','-c','1'],capture_output=True,text=True,timeout=5)
                except subprocess.TimeoutExpired as error:
                    self.reason='PROCESS_MONITOR_TIMEOUT: '+str(error);break
                row['pmon']=pmon.stdout;row['pmon_returncode']=pmon.returncode;row['pmon_error']=pmon.stderr
                used,total=int(row['gpu']['memory.used']),int(row['gpu']['memory.total'])
                util=int(row['gpu']['utilization.gpu'])
                self.high_since=(self.high_since or time.monotonic()) if util>=95 else None
                if total-used<1024:self.reason='LOW_FREE_MEMORY_UNDER_1_GIB'
                if row['original_pids_missing']:self.reason='ORIGINAL_PROCESS_DISAPPEARED; causality unknown'
                if self.high_since and time.monotonic()-self.high_since>=15:self.reason='SUSTAINED_DEVICE_UTILIZATION_GE_95_PCT_FOR_15_SECONDS'
                row['stop_reason']=self.reason;log.write(json.dumps(row)+'\n');log.flush()
                if self.reason:break
                self.stop.wait(1)
        if self.reason and self.active is not None and self.active.poll() is None:
            os.killpg(self.active.pid,signal.SIGTERM)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--library',type=Path,required=True);p.add_argument('--plan',type=Path)
    a=p.parse_args();plan=jobs(a)
    if a.plan:
        with a.plan.open('x') as f:json.dump(plan,f,indent=2);f.write('\n')
        print('Prepared',len(plan),'sequential GPU1 jobs; GPU NOT_RUN');return
    a.output.mkdir(parents=True,exist_ok=False)
    baseline=sample();assert baseline['gpu']['index']=='1'
    (a.output/'before.json').write_text(json.dumps(baseline,indent=2)+'\n')
    free=int(baseline['gpu']['memory.total'])-int(baseline['gpu']['memory.used'])
    assert free>=3072 and int(baseline['gpu']['utilization.gpu'])<95,'Shared GPU1 lacks initial headroom'
    uuid=baseline['gpu']['uuid'];env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,CUDA_DEVICE_ORDER='PCI_BUS_ID',
        HYTORCH_CUDA_LIBRARY=str(a.library.resolve()),NVIDIA_TF32_OVERRIDE='0')
    state=dict(status='RUNNING',resource_condition='SHARED_GPU',physical_gpu=1,uuid=uuid,logical_gpu=0,
        started_at=now(),device_switches=0,gpu_reset_called=False,estimated_process_budget_mib=2048,
        original_task_latency='NOT_AVAILABLE; resource observations do not establish zero impact')
    (a.output/'window-start.json').write_text(json.dumps(state,indent=2)+'\n')
    monitor=SharedMonitor(a.output/'resources.jsonl',baseline);thread=threading.Thread(target=monitor.run)
    signal.signal(signal.SIGTERM,lambda *_:sys.exit(143))
    thread.start();records=[]
    try:
        time.sleep(4)  # Observe the original task before launching our first child.
        for job in plan:
            if monitor.reason:raise RuntimeError(monitor.reason)
            monitor.job=job['name'];local=dict(env)
            if job['method']=='A':local['HYTORCH_CUDA_LIBRARY']=str(a.output/'absent_native_library.so')
            row=dict(job,started_at=now(),cuda_visible_devices=uuid,library=local['HYTORCH_CUDA_LIBRARY']);records.append(row)
            (a.output/'commands.json').write_text(json.dumps(records,indent=2)+'\n');print('START '+job['name'],flush=True)
            with (a.output/(job['name']+'.log')).open('x') as f:
                monitor.active=subprocess.Popen(job['command'],cwd=ROOT,env=local,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                row['pid']=monitor.active.pid
                (a.output/'commands.json').write_text(json.dumps(records,indent=2)+'\n')
                returncode=monitor.active.wait()
            row.update(returncode=returncode,finished_at=now());monitor.active=None
            (a.output/'commands.json').write_text(json.dumps(records,indent=2)+'\n')
            if returncode:raise subprocess.CalledProcessError(returncode,job['command'])
            result=json.loads((a.output/job['name']/'summary.json').read_text());assert result['status'] in ('PASSED','COMPLETED')
        if monitor.reason:raise RuntimeError(monitor.reason)
        state['status']='COMPLETED'
    finally:
        if monitor.active is not None and monitor.active.poll() is None:
            os.killpg(monitor.active.pid,signal.SIGTERM);monitor.active.wait()
        monitor.stop.set();thread.join();state.update(finished_at=now(),stop_reason=monitor.reason,
            completed_jobs=sum(r.get('returncode')==0 for r in records),planned_jobs=len(plan),
            release='All own GPU child processes exited; parent never creates a CUDA context; no GPU reset',monitor_joined=not thread.is_alive())
        if state['status']=='RUNNING':state['status']='FAILED'
        (a.output/'after.json').write_text(json.dumps(sample(uuid),indent=2)+'\n')
        (a.output/'window.json').write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(state),flush=True)


if __name__=='__main__':main()
