"""Reuse stage 4H's canonical initialization and backend definitions."""
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OLD = ROOT / 'docs/experiments/udacity-real-training/scripts'
sys.path.insert(0, str(OLD))
from engine import Engine as PreviousEngine, CONFIGS
from benchmarks.bench_cuda_training import initialize, torch_model
from benchmarks.bench_cuda_training import SpatialNet
from donkeycar_hy.udacity_dataset import UdacityDataset
from donkeycar_hy.model import DonkeyNet
import hytorch
import hytorch.functions as F
from hytorch import cuda
from hytorch.optimizers import SGD
from hytorch.backends.cuda import backend


class Engine(PreviousEngine):
    def __init__(self, method, model='donkey', device='cuda'):
        # P: prefetch only; R: ReLU gradient only; J: both prototypes.
        self.native_relu = method in ('R', 'J')
        self.method = 'I' if method in ('P', 'R', 'J') else method
        self.device = device
        hy = SpatialNet() if model == 'spatial' else DonkeyNet(dropout_rate=0.)
        self.initial = initialize(hy, (3, 66, 200), 42)
        self.cp = None
        if device == 'cuda':
            import cupy as cp
            cp.cuda.Device(0).use()
            self.cp = cp
        if self.method == 'E':
            import torch
            self.torch = torch
            torch.set_num_threads(4)
            torch.manual_seed(42)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            self.model = torch_model(model, self.initial).to('cuda:0' if self.cp is not None else 'cpu')
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=.001)
        else:
            self.model = hy
            if self.cp is not None:
                self.model.to_gpu()
            self.optimizer = SGD(.001, fused=self.method in 'FI').setup(self.model)

    def context(self):
        return backend(**dict(CONFIGS[self.method], relu_backward='fused' if self.native_relu else 'original'))

    def load_state(self, state):
        self.initial = {k: v.copy() for k, v in state.items()}
        self.restore()

    def forward_loss(self, x, t):
        if self.method == 'E':
            prediction = self.model(x)
            return prediction, self.torch.nn.functional.mse_loss(prediction, t)
        if self.method in 'FI':
            x = hytorch.Variable(x, requires_grad=False)
        prediction = self.model(x)
        return prediction, F.mean_square_error(prediction, t)

    def numpy(self, value):
        return value.detach().cpu().numpy().copy() if self.method == 'E' else cuda.as_numpy(value.data).copy()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    if rows:
        with Path(path).open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def compare(value, target, name, **labels):
    value, target = np.asarray(value), np.asarray(target)
    delta = value.astype(np.float64) - target.astype(np.float64)
    atol, rtol = (1e-5, 1e-4) if name.startswith('grad/') or name.startswith('forward') else (1e-6, 1e-5)
    limit = atol + rtol * abs(target.astype(np.float64))
    index = np.unravel_index(np.argmax(abs(delta)), delta.shape) if delta.shape else ()
    return dict(**labels, value=name, atol=atol, rtol=rtol, size=value.size,
                finite=bool(np.isfinite(value).all() and np.isfinite(target).all()),
                max_abs=float(abs(delta).max()), relative_l2=float(np.linalg.norm(delta.ravel()) / max(np.linalg.norm(target.astype(np.float64).ravel()), 1e-30)),
                mismatches=int((abs(delta) > limit).sum()), unequal=int((value != target).sum()),
                bitwise_equal=bool(value.dtype == target.dtype and value.tobytes() == target.tobytes()),
                worst_index=str(index), worst_value=float(value[index]), worst_reference=float(target[index]))


def result(engine, x, t):
    """One forward/backward/update, with complete host values only for diagnosis."""
    prediction, loss = engine.forward_loss(engine.convert(x), engine.convert(t))
    engine.backward(loss)
    values = dict(forward=engine.numpy(prediction), loss=np.array(engine.value(loss)))
    values.update({'grad/' + k: v for k, v in engine.gradients().items()})
    engine.update()
    engine.sync()
    values.update({'weight/' + k: v for k, v in engine.state().items()})
    return values


def load_protocol(prepared):
    protocol = json.loads((prepared / 'protocol.json').read_text())
    return protocol, UdacityDataset(protocol['csv'])


def batch(dataset, indices):
    items = [dataset[int(i)] for i in indices]
    return np.stack([v[0] for v in items]), np.stack([v[1] for v in items])
