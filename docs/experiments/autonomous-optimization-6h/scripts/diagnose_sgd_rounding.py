"""Isolate FP32 SGD rounding with identical saved parameters and gradients, CPU only."""
import argparse
import ctypes as ct
from pathlib import Path

import numpy as np
from common import ROOT, write_csv, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    torch.set_num_threads(4)
    initial = np.load(ROOT / 'runs/udacity_real_window_02/train-I/initial_weights.npz')
    values = np.load(ROOT / 'runs/udacity_real_window_02/correctness-real/step-0-I.npz')
    libm = ct.CDLL('libm.so.6')
    fmaf = libm.fmaf
    fmaf.argtypes, fmaf.restype = [ct.c_float] * 3, ct.c_float
    rows, samples = [], []
    for key in initial.files:
        p, g = initial[key], values['grad/' + key]
        separate = p - np.float32(.001) * g
        parameter = torch.nn.Parameter(torch.tensor(p.copy()))
        parameter.grad = torch.tensor(g.copy())
        torch.optim.SGD([parameter], lr=.001, foreach=False).step()
        actual = parameter.detach().numpy()
        delta = actual.astype(np.float64) - separate
        bad = np.flatnonzero(actual.ravel() != separate.ravel())
        rng = np.random.default_rng(42)
        chosen = np.unique(np.concatenate([bad[:100], rng.choice(p.size, min(100, p.size), replace=False)]))
        for index in chosen:
            fused = float(fmaf(-np.float32(.001), g.ravel()[index], p.ravel()[index]))
            reference = float(p.ravel()[index]) - float(np.float32(.001)) * float(g.ravel()[index])
            samples.append(dict(parameter=key, index=int(index), initial=float(p.ravel()[index]),
                gradient=float(g.ravel()[index]), separate=float(separate.ravel()[index]),
                torch=float(actual.ravel()[index]), fmaf=fused, fp64_update=reference,
                torch_matches_fmaf=float(actual.ravel()[index]) == fused,
                separate_abs_error=abs(float(separate.ravel()[index]) - reference),
                torch_abs_error=abs(float(actual.ravel()[index]) - reference)))
        rows.append(dict(parameter=key, elements=p.size, unequal=len(bad), max_abs=float(abs(delta).max()),
            tolerance_mismatches=int((abs(delta) > 1e-6 + 1e-5 * abs(separate)).sum()),
            sampled=len(chosen), sampled_torch_fmaf_equal=sum(r['torch_matches_fmaf'] for r in samples if r['parameter'] == key)))
    write_csv(args.output / 'summary.csv', rows)
    write_csv(args.output / 'samples.csv', samples)
    write_json(args.output / 'summary.json', dict(status='COMPLETED', device='CPU',
        elements=sum(r['elements'] for r in rows), unequal=sum(r['unequal'] for r in rows),
        max_abs=max(r['max_abs'] for r in rows), tolerance_mismatches=sum(r['tolerance_mismatches'] for r in rows),
        source='Stage 4H exact initial FP32 weights and first I gradients; identical inputs to both updates',
        caution='CPU PyTorch rounding does not by itself establish CUDA PyTorch update semantics'))


if __name__ == '__main__':
    main()
