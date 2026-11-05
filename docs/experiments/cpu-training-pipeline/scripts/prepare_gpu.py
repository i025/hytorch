"""Prepare rotated GPU commands and validate the shared loader policy using CPU only."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from pipeline_common import ROOT, load_data, write_csv, write_json
from training_config import METHODS, configuration
from hytorch.dataloaders import DataLoader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol, dataset = load_data(args.prepared, 'stream')
    indices = protocol['throughput'][:19] + [protocol['throughput'][0]]
    reference = list(DataLoader(dataset, 4, shuffle=False, indices=indices))
    rows = []
    for method in METHODS:
        config = configuration(method, 'donkey', protocol)
        for epoch in range(2):
            with DataLoader(dataset, 4, shuffle=False, indices=indices,
                            num_workers=config['workers'], prefetch_factor=config['prefetch_factor']) as loader:
                values = list(loader)
            for (x,t),(rx,rt) in zip(values,reference):
                np.testing.assert_array_equal(x, rx)
                np.testing.assert_array_equal(t, rt)
            assert len(values)==len(reference)
            rows.append(dict(method=method, epoch=epoch+1, engine=config['engine'],
                workers=config['workers'], prefetch_factor=config['prefetch_factor'],
                batches=len(values), samples=len(indices), exact_pixels=True, exact_labels=True))
    write_csv(args.output/'cpu-policy-validation.csv', rows)
    commands = []
    methods = list(METHODS)
    for repeat in range(3):
        order = methods[repeat:]+methods[:repeat]
        for model in ('donkey','spatial'):
            for task in ('io','compute','phases'):
                for method in order:
                    command = [str(ROOT/'.venv/bin/python'), str(Path(__file__).with_name('future_training.py').resolve()),
                        '--method',method,'--model',model,'--task',task,'--prepared',str(args.prepared.resolve()),
                        '--output',str(ROOT/f'runs/cpu_pipeline_future_gpu_01/{model}-{task}-{method}-r{repeat+1}'),
                        '--steps','256','--prefetch-factor','2','--no-fingerprints','--run-gpu']
                    commands.append(dict(repeat=repeat+1, model=model, task=task, method=method, command=command))
    write_json(args.output/'plan.json', dict(status='PREPARED_NOT_RUN', gpu='NOT_RUN',
        required_environment={'CUDA_VISIBLE_DEVICES':'ONE_AUTHORIZED_GPU_UUID', 'NVIDIA_TF32_OVERRIDE':'0',
                              'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONPATH':str(ROOT)},
        prerequisites=['Separately authorize an idle single-GPU window',
                       'Build existing native library including ReLU; pass existing GPU correctness and same-state tests before timing',
                       'Use CPU affinity 0-3, one process at a time; never run this plan automatically in CPU-only stage'],
        summary_ratios=['P/I existing async','R/I native ReLU','J/I combined','J/A cumulative','J/E','E-prefetch/E','J/E-prefetch'],
        jobs=commands))
    print('PASS: 14 CPU loader-policy checks; prepared 126 future jobs, none executed')


if __name__ == '__main__':
    main()
