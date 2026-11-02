"""One fixed-UUID GPU lifecycle; no reselection, tuning, plotting or CPU export."""
import argparse
import ctypes as ct
from datetime import datetime,timezone
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('stage4e_jobs',ROOT/'docs/experiments/conv-shape-dispatch/scripts/run_validation.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
previous.CONFIGS={k:previous.CONFIGS[k] for k in ('A','F','E')}
previous.CONFIGS['I']=[*previous.CONFIGS['F'],'--conv-dweight','fused']
previous.SCRIPTS=HERE


def reset_function():
    fn=ct.CDLL(str(Path(os.environ['CUDA_PATH'])/'lib64/libcudart.so')).cudaDeviceReset
    fn.argtypes,fn.restype=[],ct.c_int
    return fn


def jobs(a):
    new=['tests/test_conv_dweight_gpu.py']
    old=['tests/test_conv_dinput_opt_gpu.py','tests/test_conv_shape_dispatch_gpu.py',
        'tests/test_conv_backward_fusion_gpu.py','tests/test_conv_dinput_pruning.py',
        'tests/test_fused_sgd_gpu.py','tests/test_cuda_device.py']
    result=[]
    for name,files in [('correctness-new',new),('correctness-regression',old)]:
        result.append(dict(name=name,method=None,command=[sys.executable,'-m','pytest',*files,'-q','-ra',
            '--junitxml='+str(a.output/(name+'.xml'))]))
    for repeat in range(1,4):
        methods=['original','fused','torch'];methods=methods[repeat-1:]+methods[:repeat-1]
        for method in methods:
            name=f'micro-{method}-r{repeat}'
            result.append(dict(name=name,method=None,command=[sys.executable,str(HERE/'bench_shapes.py'),
                '--method',method,'--repeat',str(repeat),'--output',str(a.output/name)]))
    for mode in ('throughput','phases'):
        for repeat in range(1,4):
            methods=list('AFIE');order=methods[repeat-1:]+methods[:repeat-1]
            for model in ('donkey','spatial'):
                for method in order:
                    name=f'training-{model}-{mode}-{method}-r{repeat}'
                    result.append(dict(name=name,method=method,command=previous.training(a,name,model,method,mode,False)))
    for model in ('donkey','spatial'):
        for method in 'AFIE':
            name='trace-training-'+model+'-'+method
            command=previous.training(a,name,model,method,'throughput',True)
            result.append(dict(name=name,method=method,command=['nsys','profile','--trace=cuda,nvtx',
                '--sample=none','--cpuctxsw=none','-o',str(a.output/name),*command]))
    for repeat in range(1,4):
        methods=['original','fused','torch'];methods=methods[repeat-1:]+methods[:repeat-1]
        for method in methods:
            name=f'trace-micro-{method}-r{repeat}'
            command=[sys.executable,str(HERE/'bench_shapes.py'),'--method',method,'--repeat',str(repeat),
                '--nvtx','--output',str(a.output/name)]
            result.append(dict(name=name,method=None,command=['nsys','profile','--trace=cuda,nvtx',
                '--sample=none','--cpuctxsw=none','-o',str(a.output/name),*command]))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu-uuid',required=True)
    p.add_argument('--physical-gpu',required=True);p.add_argument('--library',type=Path,required=True)
    p.add_argument('--plan',type=Path,help='Write the complete CPU-only plan and exit before importing CUDA')
    a=p.parse_args();plan=jobs(a)
    if a.plan:
        with a.plan.open('x') as f:json.dump(plan,f,indent=2);f.write('\n')
        print('Prepared',len(plan),'sequential commands; GPU NOT_RUN');return
    assert os.environ['CUDA_VISIBLE_DEVICES']==a.gpu_uuid
    a.output.mkdir(parents=True,exist_ok=False)
    reset=reset_function()
    import cupy as cp
    cp.cuda.Device(0).use();cp.cuda.Stream.null.synchronize()
    state=dict(status='RUNNING',physical_gpu=a.physical_gpu,uuid=a.gpu_uuid,logical_gpu=0,
        started_at=datetime.now(timezone.utc).isoformat(),device_switches=0,
        cuda_pci_bus_id=cp.cuda.Device(0).pci_bus_id,initial_memory=cp.cuda.runtime.memGetInfo(),
        context_policy='Controller keeps its initialized context alive across all child processes; no GPU busy loop or reservation')
    (a.output/'window-start.json').write_text(json.dumps(state,indent=2)+'\n')
    signal.signal(signal.SIGTERM,lambda *_:sys.exit(143))
    records=[]
    try:
        for job in plan:
            name=job['name'];env=dict(os.environ,HYTORCH_CUDA_LIBRARY=str(a.library.resolve()),
                HYTORCH_REQUIRE_CUDA='1',HYTORCH_PRUNING_DEVICE='cuda',
                HYTORCH_SHAPE_TEST_OUTPUT=str(a.output/(name+'-arrays')),
                HYTORCH_CONV_TEST_OUTPUT=str(a.output/'legacy-arrays'),
                HYTORCH_PRUNING_TEST_OUTPUT=str(a.output/'pruning-arrays'),
                HYTORCH_SGD_TEST_OUTPUT=str(a.output/'sgd-arrays'))
            if job['method']=='A':env['HYTORCH_CUDA_LIBRARY']=str(a.output/'absent_native_library.so')
            row=dict(job,library=env['HYTORCH_CUDA_LIBRARY'],started_at=datetime.now(timezone.utc).isoformat())
            records.append(row);(a.output/'commands.json').write_text(json.dumps(records,indent=2)+'\n')
            print('START '+name,flush=True)
            with (a.output/(name+'.log')).open('x') as log:
                done=subprocess.run(job['command'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            row.update(returncode=done.returncode,finished_at=datetime.now(timezone.utc).isoformat())
            (a.output/'commands.json').write_text(json.dumps(records,indent=2)+'\n');done.check_returncode()
            if name.startswith('correctness-'):
                import xml.etree.ElementTree as ET
                tests=ET.parse(a.output/(name+'.xml')).getroot().findall('.//testcase')
                assert tests and not any(t.find('skipped') is not None for t in tests),'GPU skips are not acceptance'
            if name.startswith(('micro-','training-','trace-micro-','trace-training-')):
                summary=json.loads((a.output/name/'summary.json').read_text())
                assert summary['status']=='COMPLETED'
        state['status']='COMPLETED'
    finally:
        state['status']='FAILED' if state['status']=='RUNNING' else state['status']
        state['finished_gpu_work_at']=datetime.now(timezone.utc).isoformat()
        cp.get_default_memory_pool().free_all_blocks();status=reset()
        state['reset_status']=status
        state['released_at']=datetime.now(timezone.utc).isoformat() if status==0 else None
        state['context_released']=status==0
        state['completed_commands']=sum(r.get('returncode')==0 for r in records)
        state['planned_commands']=len(plan)
        (a.output/'window.json').write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(state),flush=True)
        if status:raise RuntimeError('cudaDeviceReset returned '+str(status))


if __name__=='__main__':main()
