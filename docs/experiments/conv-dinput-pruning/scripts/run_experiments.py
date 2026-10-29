"""Five matched configurations, one guarded GPU window, unique output paths."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
PREVIOUS = ROOT/'docs/experiments/donkeynet-backward-profiling/scripts'
CONFIGS = {
    'A': ['--framework','hytorch'],
    'B': ['--framework','hytorch','--matmul','shared','--conv','im2col','--pool','cupy','--tile','16'],
    'C': ['--framework','hytorch','--matmul','shared','--conv','im2col','--pool','cupy','--tile','16','--fused-sgd'],
    'D': ['--framework','hytorch','--matmul','shared','--conv','im2col','--pool','cupy','--tile','16','--fused-sgd','--prune-input-grad'],
    'E': ['--framework','torch'],
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--library',type=Path,required=True)
    p.add_argument('--gpu-uuid',required=True)
    a = p.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    env = dict(os.environ,HYTORCH_CUDA_LIBRARY=str(a.library.resolve()),
               HYTORCH_REQUIRE_CUDA='1',HYTORCH_PRUNING_DEVICE='cuda',
               HYTORCH_PRUNING_TEST_OUTPUT=str(out/'correctness'),
               HYTORCH_SGD_TEST_OUTPUT=str(out/'fused-regression'))
    records = []

    def run(name, command, method=None):
        local_env = dict(env)
        if method == 'A':
            # A must work without loading our native library. A native call fails loudly.
            local_env['HYTORCH_CUDA_LIBRARY'] = str(out/'intentionally_absent_native_library.so')
        row = dict(name=name,command=[str(v) for v in command],
                   started_at=datetime.now(timezone.utc).isoformat(),
                   library=local_env['HYTORCH_CUDA_LIBRARY'],log=str(out/(name+'.log')))
        records.append(row)
        (out/'commands.json').write_text(json.dumps(records,indent=2)+'\n')
        print('START '+name,flush=True)
        start = time.monotonic()
        with Path(row['log']).open('x') as log:
            result = subprocess.run(row['command'],cwd=ROOT,env=local_env,stdout=log,stderr=subprocess.STDOUT)
        row.update(returncode=result.returncode,wall_s=time.monotonic()-start,
                   finished_at=datetime.now(timezone.utc).isoformat())
        (out/'commands.json').write_text(json.dumps(records,indent=2)+'\n')
        print(json.dumps(row),flush=True)
        result.check_returncode()

    run('gpu-tests',[sys.executable,'-m','pytest','tests/test_conv_dinput_pruning.py',
                     'tests/test_fused_sgd_gpu.py','-q','-ra','--junitxml='+str(out/'gpu-tests.xml')])
    common = ['--device','cuda','--gpu','0','--monitor-id',a.gpu_uuid,'--model','donkey',
              '--batch','4','--warmup','10','--seed','42','--lr','0.001','--workers','0',
              '--no-monitor','--no-fingerprints','--save-comparison-inputs']
    methods = list(CONFIGS)
    for repeat in range(1,4):
        ordered = methods[repeat-1:]+methods[:repeat-1]
        for mode in ['throughput','phases']:
            for method in ordered:
                name = f'training-{mode}-{method}-r{repeat}'
                run(name,[sys.executable,ROOT/'benchmarks/bench_cuda_training.py',*common,
                          '--samples','256','--epochs','4',*CONFIGS[method],
                          '--mode',mode,'--output',out/name],method)
    for method in methods:
        name = 'trace-'+method
        run(name,['nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none',
                  '-o',out/name,sys.executable,PREVIOUS/'profile_training.py',*common,
                  '--samples','64','--epochs','1',*CONFIGS[method],'--mode','throughput',
                  '--nvtx','--operator-nvtx','--output',out/name],method)
        run(name+'-export',['nsys','export','--type','sqlite','--output',out/(name+'.sqlite'),
                            out/(name+'.nsys-rep')])
    (out/'completion.json').write_text(json.dumps(dict(status='PASSED',commands=len(records),
        finished_at=datetime.now(timezone.utc).isoformat()),indent=2)+'\n')


if __name__ == '__main__':
    main()
