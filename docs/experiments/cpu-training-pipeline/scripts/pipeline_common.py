"""CPU-only experiment helpers; reuse the fixed 4H data protocol."""
import csv
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from donkeycar_hy.udacity_dataset import UdacityDataset


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    if rows:
        with Path(path).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def load_module(path):
    spec = importlib.util.spec_from_file_location('baseline_loader', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inplace_preprocess(path):
    with Image.open(path) as image:
        image = image.convert('RGB')
        image = image.crop((0, 70, image.width, image.height-25))
        image = image.resize((200, 66), Image.BILINEAR)
        value = np.asarray(image, dtype=np.float32)
        value /= np.float32(127.5)
        value -= np.float32(1)
    return np.ascontiguousarray(value.transpose(2, 0, 1))


class InplaceDataset(UdacityDataset):
    def __getitem__(self, index):
        return inplace_preprocess(self.paths[index]), self.labels[index:index+1].copy()


def load_data(prepared, variant, original=None):
    protocol = json.loads((prepared / 'protocol.json').read_text())
    if variant == 'stream':
        return protocol, UdacityDataset(protocol['csv'], open_stream=True)
    cls = InplaceDataset if variant == 'inplace' else UdacityDataset
    if original is not None and variant in ('existing', 'repaired'):
        cls = load_module(original).UdacityDataset
    dataset = cls(protocol['csv'])
    return protocol, dataset


def configurations():
    return [(0, 2)] + [(w, p) for w in (1, 2, 4) for p in (1, 2, 4)]
