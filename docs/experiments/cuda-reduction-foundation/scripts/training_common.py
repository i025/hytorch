"""Reuse historical matched initialization/SGD, changing only explicit switches."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location('existing_training', ROOT/'docs/experiments/autonomous-optimization-6h/scripts/common.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
from hytorch.backends.cuda import backend


class Engine(previous.Engine):
    def __init__(self, method, model='donkey', device='cuda'):
        self.sum_native = method in ('S', 'SR')
        super().__init__('R' if method in ('R','SR') else 'I' if method == 'S' else method, model, device)

    def context(self):
        return backend(**dict(previous.CONFIGS[self.method],
            relu_backward='fused' if self.native_relu else 'original',
            reduction='native' if self.sum_native else 'original'))


compare = previous.compare
result = previous.result
write_csv = previous.write_csv
write_json = previous.write_json
