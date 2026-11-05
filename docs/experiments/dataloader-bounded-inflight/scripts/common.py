"""CPU-only inputs and record helpers for the bounded scheduling experiment."""
import csv
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from donkeycar_hy.udacity_dataset import UdacityDataset


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def write_csv(path, rows):
    if rows:
        with Path(path).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def loader_module(variant, baseline):
    if variant == 'bounded':
        import hytorch.dataloaders as module
    else:
        spec = importlib.util.spec_from_file_location('baseline_loader', baseline)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class StragglerDataset:
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if index == 0:
            time.sleep(1)
        time.sleep(.0005)
        return np.full((3, 66, 200), index, np.float32), np.array([index], np.float32)


def inputs(prepared, workload, size):
    if workload == 'straggler':
        return StragglerDataset(size), list(range(size))
    protocol = json.loads((prepared/'protocol.json').read_text())
    return UdacityDataset(protocol['csv'], open_stream=True), protocol['throughput']
