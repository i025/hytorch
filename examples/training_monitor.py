"""Minimal reusable monitor integration: python examples/training_monitor.py --device cpu."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from hytorch.profiling import TrainingMonitor
from hytorch.monitor_loop import monitored_epochs
from hytorch.optimizers import SGD
import hytorch.functions as F
from benchmarks.bench_cuda_training import TinyNet, SyntheticDataset, initialize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--output', default='runs/monitor_example')
    args = p.parse_args()
    xp = np
    if args.device == 'cuda':
        import cupy as xp
    model = TinyNet()
    initialize(model, (3,16,16), 42)
    if args.device == 'cuda':
        model.to_gpu()
    optimizer = SGD(.001, backend='legacy').setup(model)
    data = SyntheticDataset(32, (3,16,16), 42)
    monitor = TrainingMonitor(enabled=True, output_dir=args.output,
                              cuda_backend='cupy' if args.device == 'cuda' else None,
                              gpu_index=0 if args.device == 'cuda' else None)
    with monitor:
        # The helper wraps next(iterator), not merely the body of `for batch`.
        monitored_epochs(data, 4, 1, monitor=monitor, transfer=xp.asarray, predict=model,
                         loss_fn=F.mean_square_error,
                         backward=lambda loss: (model.cleargrads(), loss.backward()),
                         update=optimizer.update, retain_loss=lambda loss: loss.data)
    print(monitor.run_dir)
    print(monitor.summary())


if __name__ == '__main__':
    main()
