"""One fixed GPU0 window, sequential children, independent correctness gates."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import psutil
from watch_gpu0 import snapshot

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--schedule',type=Path,required=True);p.add_argument('--plan',type=Path,required=True)
    a=p.parse_args();config=json.loads(a.schedule.read_text());plan=json.loads(a.plan.read_text())
    raw=Path(config['raw']);out=raw/'gpu-window-01';out.mkdir(exist_ok=False)
    cutoff=datetime.fromisoformat(config['gpu_cutoff_utc']).timestamp()
    uuid=config['uuid'];before=snapshot(uuid)
    assert not before['processes'] and before['memory_used_mib']<=config['idle_memory_mib']
    assert before['utilization_pct']<=config['idle_utilization_pct']
    state=dict(status='RUNNING',uuid=uuid,physical_gpu=0,logical_gpu=0,device_switches=0,
        started_utc=datetime.now(timezone.utc).isoformat(),before=before,commands=[],
        lease='Observed idle and user-authorized GPU0; no exclusive scheduler lease',gates={})
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,CUDA_DEVICE_ORDER='PCI_BUS_ID',NVIDIA_TF32_OVERRIDE='0',
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH=str(ROOT),
        HYTORCH_CUDA_LIBRARY=plan['library'],HYTORCH_REQUIRE_CUDA='1',HYTORCH_PRUNING_DEVICE='cuda',
        HYTORCH_SHAPE_TEST_OUTPUT=str(out/'test-arrays'),HYTORCH_PRUNING_TEST_OUTPUT=str(out/'pruning-arrays'),
        HYTORCH_SGD_TEST_OUTPUT=str(out/'sgd-arrays'),HYTORCH_CONV_TEST_OUTPUT=str(out/'legacy-arrays'),
        HYTORCH_REDUCTION_FAILURE_INPUTS=str(ROOT/'runs/cuda_reduction_01/cpu-reference-01'))
    def save():
        (out/'window.json').write_text(json.dumps(state,indent=2)+'\n')
    own=set();active=None
    def run(name,command,soft=False,method=None,min_remaining=180):
        nonlocal active
        if time.time()+min_remaining>cutoff:
            raise RuntimeError('GPU cutoff too near for '+name)
        gpu=snapshot(uuid)
        if any(v['pid'] not in own for v in gpu['processes']):
            raise RuntimeError('Foreign compute process; stop our window, never switch GPU')
        row=dict(name=name,command=[str(v) for v in command],started_utc=datetime.now(timezone.utc).isoformat())
        state['commands'].append(row);save();print('START',name,flush=True)
        local=dict(env)
        if method=='A':local['HYTORCH_CUDA_LIBRARY']=str(out/'absent_native_library.so')
        with (out/(name+'.log')).open('x') as log:
            active=subprocess.Popen(row['command'],cwd=ROOT,env=local,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            own.add(active.pid);row['pid']=active.pid;save()
            last_sample=0.
            while active.poll() is None:
                if time.time()>=cutoff:
                    raise RuntimeError('GPU cutoff reached')
                if time.monotonic()-last_sample>=5:
                    gpu=snapshot(uuid);last_sample=time.monotonic()
                    try:own.update(v.pid for v in psutil.Process(active.pid).children(recursive=True))
                    except psutil.NoSuchProcess:pass  # Child exited between poll and process-tree read.
                    if any(v['pid'] not in own for v in gpu['processes']):
                        state['conflict_snapshot']=gpu
                        raise RuntimeError('Foreign GPU task appeared; stop only our process group')
                    if gpu['memory_free_mib']<1024:
                        raise RuntimeError('Less than 1GiB free; stop for resource conflict')
                time.sleep(1)
            row.update(returncode=active.returncode,finished_utc=datetime.now(timezone.utc).isoformat())
        active=None;save()
        if row['returncode']:
            text=(out/(name+'.log')).read_text(errors='replace').lower()
            if 'out of memory' in text or 'outofmemoryerror' in text:
                raise RuntimeError('OOM reported; stop new GPU work and retain '+name)
        if row['returncode'] and not soft:raise RuntimeError('Task failed: '+name)
        return row['returncode']
    def pytest(name,files,soft=False):
        return run(name,[sys.executable,'-m','pytest',*files,'-q','-ra','--junitxml='+str(out/(name+'.xml'))],soft)
    try:
        run('environment',[sys.executable,HERE/'gpu_environment.py','--run-gpu','--output',out/'environment.json'])
        pytest('sum-tests',['tests/test_reduction_gpu.py'],soft=True)
        code=run('sum-diagnosis',[sys.executable,HERE/'diagnose_sum.py','--run-gpu','--output',out/'sum-diagnosis'],soft=True)
        summary=json.loads((out/'sum-diagnosis/summary.json').read_text()) if (out/'sum-diagnosis/summary.json').exists() else {}
        tests=ET.parse(out/'sum-tests.xml').getroot().findall('.//testcase')
        unexpected=[t.attrib['name'] for t in tests if any(t.find(k) is not None for k in ('failure','error','skipped'))
            and not t.attrib['name'].startswith('test_retained_cancellation_inputs')]
        sum_ok=code==0 and not unexpected and summary.get('ordinary_inputs_pass',False)
        state['gates']['sum_implementation_and_ordinary_inputs']=sum_ok
        state['gates']['sum_general_correctness']=summary.get('status','NOT_RUN')
        state['gates']['sum_unexpected_tests']=unexpected;save()
        relu_ok=pytest('relu-tests',['tests/test_relu_backward_gpu.py'],soft=True)==0
        state['gates']['relu_correctness']=relu_ok;save()
        pytest('gpu-regression',plan['regression_tests'])
        check=[sys.executable,HERE/'check_training.py','--run-gpu','--prepared',plan['prepared'],'--output',out/'training-check']
        if sum_ok:check.append('--sum-enabled')
        if not relu_ok:check.append('--relu-disabled')
        run('training-check',check,min_remaining=600)
        training=json.loads((out/'training-check/summary.json').read_text())
        state['gates']['training']=training;save()
        if sum_ok:
            run('sum-micro',[sys.executable,HERE/'bench_sum.py','--run-gpu','--eligibility',out/'sum-diagnosis/summary.json','--output',out/'sum-micro'])
        if relu_ok:
            run('relu-micro',[sys.executable,ROOT/'docs/experiments/autonomous-optimization-6h/scripts/bench_relu.py','--output',out/'relu-micro'])
        for task in ('compute','io','phases'):
            for repeat in range(3):
                methods=['A','I','P','R','S','J','E','E-prefetch'];methods=methods[repeat:]+methods[:repeat]
                for model in ('donkey','spatial'):
                    for method in methods:
                        if not training['eligible'][model].get(method,False):continue
                        name=f'{task}-{model}-{method}-r{repeat}'
                        command=[sys.executable,HERE/'measure_training.py','--run-gpu','--no-fingerprints',
                            '--method',method,'--model',model,'--task',task,'--steps','256',
                            '--prepared',plan['prepared'],'--output',out/name]
                        if sum_ok:command.append('--sum-enabled')
                        if not relu_ok:command.append('--relu-disabled')
                        run(name,command,method=method)
        if sum_ok:
            run('trace-sum',['nsys','profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none',
                '-o',out/'trace-sum',sys.executable,HERE/'bench_sum.py','--run-gpu','--nvtx',
                '--eligibility',out/'sum-diagnosis/summary.json','--output',out/'trace-sum-data'],min_remaining=600)
        if relu_ok:
            run('trace-relu',['nsys','profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none',
                '-o',out/'trace-relu',sys.executable,ROOT/'docs/experiments/autonomous-optimization-6h/scripts/bench_relu.py',
                '--trace','--output',out/'trace-relu-data'],min_remaining=600)
        for model in ('donkey','spatial'):
            for method in ('I','J','E-prefetch'):
                if not training['eligible'][model].get(method,False):continue
                name='trace-'+model+'-'+method
                command=['nsys','profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none','-o',out/name,
                    sys.executable,HERE/'measure_training.py','--run-gpu','--nvtx','--no-fingerprints','--method',method,
                    '--model',model,'--task','compute','--steps','16','--prepared',plan['prepared'],'--output',out/(name+'-data')]
                if sum_ok:command.append('--sum-enabled')
                if not relu_ok:command.append('--relu-disabled')
                run(name,command,min_remaining=600)
        # A single optional CPU-driven prototype may be prepared while GPU0 was busy.
        optional=raw/'optional-gpu-tasks.json'
        if optional.exists():
            for job in json.loads(optional.read_text()):run(job['name'],job['command'],min_remaining=600)
        state['status']='COMPLETED'
    except (RuntimeError,subprocess.SubprocessError,AssertionError) as error:
        state.update(status='STOPPED',reason=str(error))
        raise
    finally:
        if active and active.poll() is None:
            os.killpg(active.pid,signal.SIGTERM)
            try:active.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(active.pid,signal.SIGKILL);active.wait()
        state.update(finished_utc=datetime.now(timezone.utc).isoformat(),after=snapshot(uuid),
            release='All our GPU children exited; controller has no CUDA context; no reset or device switch')
        save();print(json.dumps(state),flush=True)


if __name__=='__main__':main()
