"""One guarded GPU window: correctness, microbench, six-way training, Nsight."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[4]
SCRIPTS=Path(__file__).resolve().parent
CONFIGS={
    'A':['--framework','hytorch'],
    'B':['--framework','hytorch','--matmul','shared','--conv','im2col','--pool','cupy','--tile','16'],
    'C':['--framework','hytorch','--matmul','shared','--conv','im2col','--pool','cupy','--tile','16','--fused-sgd'],
    'D':['--framework','hytorch','--matmul','shared','--conv','im2col','--pool','cupy','--tile','16','--fused-sgd','--prune-input-grad'],
    'F':['--framework','hytorch','--matmul','shared','--conv','im2col','--pool','cupy','--tile','16','--fused-sgd','--prune-input-grad','--conv-backward','fused'],
    'E':['--framework','torch'],
}


def commands(out,gpu):
    jobs=[('gpu-tests',[sys.executable,'-m','pytest','tests/test_conv_backward_fusion_gpu.py',
        'tests/test_conv_dinput_pruning.py','tests/test_fused_sgd_gpu.py','tests/test_cuda_device.py',
        '-q','-ra','--junitxml='+str(out/'gpu-tests.xml')],None)]
    for repeat in range(1,4):
        methods=['original','fused','torch']
        for method in methods[repeat-1:]+methods[:repeat-1]:
            name=f'micro-{method}-r{repeat}'
            jobs.append((name,[sys.executable,SCRIPTS/'bench_backward.py','--method',method,
                              '--warmup','5','--iterations','30','--output',out/name],None))
    common=['--device','cuda','--gpu','0','--monitor-id',gpu,'--model','donkey','--batch','4',
        '--warmup','10','--seed','42','--lr','0.001','--workers','0','--no-monitor',
        '--no-fingerprints','--save-comparison-inputs']
    methods=list(CONFIGS)
    # Finish the three throughput repeats first during short idle windows.
    for mode in ['throughput','phases']:
        for repeat in range(1,4):
            order=methods[repeat-1:]+methods[:repeat-1]
            for method in order:
                name=f'training-{mode}-{method}-r{repeat}'
                jobs.append((name,[sys.executable,ROOT/'benchmarks/bench_cuda_training.py',*common,
                    '--samples','256','--epochs','4',*CONFIGS[method],'--mode',mode,'--output',out/name],method))
    traces=[]
    # Capture the pruning/gather comparison first if the idle window is short.
    for method in 'DFABCE':
        name='trace-training-'+method
        args=[sys.executable,SCRIPTS/'profile_training.py',*common,'--samples','64','--epochs','1',
              *CONFIGS[method],'--mode','throughput','--nvtx','--operator-nvtx','--output',out/name]
        if method!='E':args+=['--memory-stats']
        traces.append((name,args,method))
    for method in ['original','fused','torch']:
        name='trace-micro-'+method
        traces.append((name,[sys.executable,SCRIPTS/'bench_backward.py','--method',method,
            '--warmup','5','--iterations','3','--nvtx','--output',out/name],None))
    for name,args,method in traces:
        jobs.append((name,['nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none',
                           '-o',out/name,*args],method))
        jobs.append((name+'-export',['nsys','export','--type','sqlite','--output',out/(name+'.sqlite'),
                                    out/(name+'.nsys-rep')],None))
    return jobs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--library',type=Path,required=True)
    p.add_argument('--gpu-uuid',required=True)
    p.add_argument('--plan-only',action='store_true',help='Save commands without executing GPU work')
    p.add_argument('--only',choices=['all','traces'],default='all',help='Resume profiling after completed correctness/performance stages')
    a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    jobs=commands(out,a.gpu_uuid)
    if a.only=='traces':jobs=[job for job in jobs if job[0].startswith('trace-')]
    env=dict(os.environ,HYTORCH_CUDA_LIBRARY=str(a.library.resolve()),HYTORCH_REQUIRE_CUDA='1',
        HYTORCH_CONV_TEST_OUTPUT=str(out/'correctness'),HYTORCH_PRUNING_DEVICE='cuda',
        HYTORCH_PRUNING_TEST_OUTPUT=str(out/'pruning-regression'),HYTORCH_SGD_TEST_OUTPUT=str(out/'sgd-regression'))
    records=[]
    for name,command,method in jobs:
        local=dict(env)
        if method=='A':local['HYTORCH_CUDA_LIBRARY']=str(out/'intentionally_absent_native_library.so')
        row=dict(name=name,command=[str(v) for v in command],library=local['HYTORCH_CUDA_LIBRARY'])
        records.append(row)
        if a.plan_only:continue
        row['started_at']=datetime.now(timezone.utc).isoformat()
        print('START '+name,flush=True)
        (out/'commands.json').write_text(json.dumps(records,indent=2)+'\n')
        with (out/(name+'.log')).open('x') as f:
            result=subprocess.run(row['command'],cwd=ROOT,env=local,stdout=f,stderr=subprocess.STDOUT)
        row.update(returncode=result.returncode,finished_at=datetime.now(timezone.utc).isoformat())
        (out/'commands.json').write_text(json.dumps(records,indent=2)+'\n')
        print(json.dumps(row),flush=True)
        result.check_returncode()
    status='NOT_RUN' if a.plan_only else 'COMPLETED'
    (out/('planned-commands.json' if a.plan_only else 'commands.json')).write_text(json.dumps(records,indent=2)+'\n')
    (out/'completion.json').write_text(json.dumps(dict(status=status,commands=len(jobs)),indent=2)+'\n')
    print(status,len(jobs),'commands')


if __name__=='__main__':main()
