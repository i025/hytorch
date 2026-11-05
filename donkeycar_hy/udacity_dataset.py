"""Udacity CSV labels and one RGB preprocessing path for training/inference."""
import csv
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from PIL import Image

PREPROCESS = dict(color='RGB', crop_top=70, crop_bottom=25, width=200,
                  height=66, resize='PIL bilinear', normalization='pixel/127.5-1',
                  layout='contiguous float32 CHW', camera='center', augmentation='none')


def preprocess_udacity(path):
    with Image.open(path) as image:
        image = image.convert('RGB')
        image = image.crop((0, 70, image.width, image.height-25))
        image = image.resize((200, 66), Image.BILINEAR)
        value = np.asarray(image, dtype=np.float32)/np.float32(127.5)-np.float32(1)
    return np.ascontiguousarray(value.transpose(2, 0, 1))


class UdacityDataset:
    def __init__(self, csv_path, open_stream=False):
        self.csv_path = Path(csv_path)
        self.open_stream = open_stream
        with self.csv_path.open(newline='') as stream:
            self.records = list(csv.DictReader(stream))
        self.paths = [self.image_path(r['center']) for r in self.records]
        self.labels = np.array([float(r['steering']) for r in self.records], np.float32)
        self.times = [datetime.strptime(p.stem[len('center_'):], '%Y_%m_%d_%H_%M_%S_%f')
                      for p in self.paths]

    def image_path(self, value):
        # Simulator CSVs can contain paths from another machine; IMG basename
        # is the dataset's explicit portable path convention.
        return self.csv_path.parent/'IMG'/value.strip().replace('\\', '/').split('/')[-1]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        if self.open_stream:
            with self.paths[index].open('rb') as stream:
                value = preprocess_udacity(stream)
        else:
            value = preprocess_udacity(self.paths[index])
        return value, self.labels[index:index+1].copy()

    def split_indices(self, train_fraction=.8, gap_seconds=10):
        order = np.argsort(self.times, kind='stable')
        cut = int(len(order)*train_fraction)
        validation = order[cut:]
        boundary = self.times[validation[0]]-timedelta(seconds=gap_seconds)
        training = np.array([i for i in order[:cut] if self.times[i] < boundary], dtype=np.int64)
        excluded = np.array([i for i in order[:cut] if self.times[i] >= boundary], dtype=np.int64)
        return training, validation, excluded
