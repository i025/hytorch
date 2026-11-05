"""Write commands for one future GPU window; this program never imports CUDA."""
import argparse
import json
from pathlib import Path

from cases import CASES
from training_config_reduction import METHODS

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu-output', type=Path, required=True)
    p.add_argument('--prepared', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    destination = a.gpu_output.resolve()
    commands = []
    def add(name, command):
        commands.append(dict(name=name, command=[str(v) for v in command]))
    python = ROOT/'.venv/bin/python'
    add('correctness', [python,'-m','pytest','tests/test_reduction_gpu.py','tests/test_relu_backward_gpu.py',
        '-q','-ra','--junitxml='+str(destination/'correctness.xml')])
    add('training-correctness', [python,HERE/'gpu_training_check.py','--run-gpu',
        '--prepared',a.prepared.resolve(),'--output',destination/'training-check'])
    add('micro', [python,HERE/'gpu_micro.py','--run-gpu','--output',destination/'micro'])
    add('relu-micro', [python,ROOT/'docs/experiments/autonomous-optimization-6h/scripts/bench_relu.py',
        '--output',destination/'relu-micro'])
    for task in ('compute','io','phases'):
        for model in ('donkey','spatial'):
            for repeat in range(3):
                order = list(METHODS[repeat:]+METHODS[:repeat])
                for method in order:
                    name = '{}-{}-r{}-{}'.format(task,model,repeat,method)
                    add(name,[python,HERE/'future_training.py','--run-gpu','--no-fingerprints',
                        '--method',method,'--model',model,'--task',task,'--steps','256',
                        '--prepared',a.prepared.resolve(),'--output',destination/name])
    add('trace-micro',['nsys','profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none',
        '--force-overwrite=false','--output',destination/'micro-trace',python,HERE/'gpu_micro.py',
        '--run-gpu','--nvtx','--output',destination/'micro-instrumented'])
    for model in ('donkey','spatial'):
        for method in ('I','S','R','SR','E'):
            name = 'trace-{}-{}'.format(model,method)
            add(name,['nsys','profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none',
                '--force-overwrite=false','--output',destination/name,python,HERE/'future_training.py',
                '--run-gpu','--nvtx','--no-fingerprints','--method',method,'--model',model,
                '--task','compute','--steps','16','--prepared',a.prepared.resolve(),
                '--output',destination/(name+'-instrumented')])
    plan = dict(status='PREPARED', gpu='NOT_RUN', cwd=str(ROOT), output=str(destination),
        environment=dict(HYTORCH_REQUIRE_CUDA='1',
            HYTORCH_SHAPE_TEST_OUTPUT=str(destination/'correctness-arrays'),
            HYTORCH_REDUCTION_FAILURE_INPUTS=str(ROOT/'runs/cuda_reduction_01/cpu-reference-01'),
            HYTORCH_CUDA_LIBRARY=str(ROOT/'runs/cuda_reduction_01/build-01/libhytorch_cuda.so'),
            OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONPATH=str(ROOT)),
        uuid='Must be explicitly supplied after permission/idle check; no device selected in CPU stage',
        stopping_rule='First failed command stops the window. Known CPU cancellation failures must not be waived.',
        repetitions=3, steps=256, warmup=10, shapes=len(CASES), commands=commands)
    (a.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(status=plan['status'],gpu=plan['gpu'],commands=len(commands))))


if __name__ == '__main__':
    main()
