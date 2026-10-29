"""Matched five-way training after independent pointwise-policy validation."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[4]
SCRIPTS=Path(__file__).resolve().parent
CONFIGS={
    'A':['--framework','hytorch'],
    'D':['--framework','hytorch','--matmul','shared','--conv','im2col','--pool','cupy','--tile','16','--fused-sgd','--prune-input-grad'],
    'F':['--framework','hytorch','--matmul','shared','--conv','im2col','--pool','cupy','--tile','16','--fused-sgd','--prune-input-grad','--conv-backward','fused'],
    'G':['--framework','hytorch','--matmul','shared','--conv','im2col','--pool','cupy','--tile','16','--fused-sgd','--prune-input-grad','--conv-backward','auto'],
    'E':['--framework','torch'],
}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu-uuid',required=True)
    p.add_argument('--library',type=Path,required=True)
    p.add_argument('--only',choices=['measurements','traces'],required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    validation=ROOT/'docs/experiments/conv-shape-dispatch/results/pointwise-validation-01/analysis.json'
    assert json.loads(validation.read_text())['status']=='VALIDATED'
    jobs=[];methods=list(CONFIGS)
    if a.only=='measurements':
        jobs.append(('gpu-tests',[sys.executable,'-m','pytest','tests/test_conv_shape_dispatch_gpu.py',
            'tests/test_conv_backward_fusion_gpu.py','tests/test_conv_dinput_pruning.py',
            'tests/test_fused_sgd_gpu.py','tests/test_cuda_device.py','-q','-ra',
            '--junitxml='+str(a.output/'gpu-tests.xml')],None))
        jobs.append(('auto-shapes',[sys.executable,str(SCRIPTS/'run_shapes.py'),
            '--split','validation-final','--auto','--output',str(a.output/'auto-shapes')],None))
        for mode in ('throughput','phases'):
            for repeat in range(1,4):
                order=methods[repeat-1:]+methods[:repeat-1]
                for model in ('donkey','spatial'):
                    for method in order:
                        name=f'training-{model}-{mode}-{method}-r{repeat}'
                        jobs.append((name,training(a,name,model,method,mode,False),method))
    else:
        jobs.append(('gpu-fallback-tests',[sys.executable,'-m','pytest',
            'tests/test_conv_shape_dispatch_gpu.py::test_auto_uncovered_dtype_layout','-q','-ra',
            '--junitxml='+str(a.output/'gpu-fallback-tests.xml')],None))
        for model in ('donkey','spatial'):
            for method in methods:
                name=f'trace-training-{model}-{method}'
                jobs.extend(trace_jobs(a,name,training(a,name,model,method,'throughput',True),method))
        for split in ('calibration','held-out','validation-2','validation-final'):
            for method in ('original','fused','torch'):
                name='trace-shapes-'+split+'-'+method
                command=[sys.executable,str(SCRIPTS/'bench_shapes.py'),'--split',split,
                    '--method',method,'--warmup','5','--iterations','3','--nvtx','--output',str(a.output/name)]
                jobs.extend(trace_jobs(a,name,command,None))
        name='trace-shapes-validation-final-auto'
        jobs.extend(trace_jobs(a,name,[sys.executable,str(SCRIPTS/'bench_shapes.py'),
            '--split','validation-final','--method','auto','--warmup','5','--iterations','3',
            '--nvtx','--output',str(a.output/name)],None))
    env=dict(os.environ,HYTORCH_CUDA_LIBRARY=str(a.library.resolve()),HYTORCH_REQUIRE_CUDA='1',
        HYTORCH_PRUNING_DEVICE='cuda',HYTORCH_CONV_TEST_OUTPUT=str(a.output/'legacy-correctness'),
        HYTORCH_SHAPE_TEST_OUTPUT=str(a.output/'shape-correctness'),
        HYTORCH_PRUNING_TEST_OUTPUT=str(a.output/'pruning-correctness'),HYTORCH_SGD_TEST_OUTPUT=str(a.output/'sgd-correctness'))
    records=[]
    for name,command,method in jobs:
        local=dict(env)
        if method=='A':local['HYTORCH_CUDA_LIBRARY']=str(a.output/'absent_native_library.so')
        row=dict(name=name,command=command,library=local['HYTORCH_CUDA_LIBRARY'],started_at=datetime.now(timezone.utc).isoformat())
        records.append(row);(a.output/'commands.json').write_text(json.dumps(records,indent=2)+'\n')
        print('START '+name,flush=True)
        with (a.output/(name+'.log')).open('x') as f:done=subprocess.run(command,cwd=ROOT,env=local,stdout=f,stderr=subprocess.STDOUT)
        row.update(returncode=done.returncode,finished_at=datetime.now(timezone.utc).isoformat())
        (a.output/'commands.json').write_text(json.dumps(records,indent=2)+'\n');done.check_returncode()
    (a.output/'completion.json').write_text(json.dumps(dict(status='COMPLETED',commands=len(jobs),
        policy='Independent validation passed only for restricted small pointwise auto'),indent=2)+'\n')


def training(a,name,model,method,mode,trace):
    script=SCRIPTS/'profile_training.py' if trace else ROOT/'benchmarks/bench_cuda_training.py'
    command=[sys.executable,str(script),'--device','cuda','--gpu','0','--monitor-id',a.gpu_uuid,
        '--model',model,'--batch','4','--warmup','10','--seed','42','--lr','0.001','--workers','0',
        '--no-monitor','--no-fingerprints','--save-comparison-inputs','--samples','64' if trace else '256',
        '--epochs','1' if trace else '4','--mode',mode,*CONFIGS[method],'--output',str(a.output/name)]
    if trace:
        command+=['--nvtx','--operator-nvtx']
        if method!='E':command+=['--memory-stats']
    return command


def trace_jobs(a,name,command,method):
    profile=['nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','-o',str(a.output/name),*command]
    export=['nsys','export','--type','sqlite','--output',str(a.output/(name+'.sqlite')),str(a.output/(name+'.nsys-rep'))]
    return [(name,profile,method),(name+'-export',export,None)]


if __name__=='__main__':main()
