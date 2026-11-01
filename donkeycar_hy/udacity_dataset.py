"""Center-camera Udacity CSV data and deterministic temporal holdouts.

CSV order and source indices are preserved. Temporal splitting requires strictly
increasing timestamps in center_YYYY_MM_DD_HH_MM_SS_mmm.jpg filenames; it never
silently sorts, shuffles or infers recording sessions from gaps. Filename times
have no timezone information and are retained as naive datetimes.
"""

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import math
from pathlib import Path
from typing import Tuple, Union

import numpy as np

from donkeycar_hy.preprocess import load_rgb_image, preprocess_rgb


@dataclass(frozen=True)
class UdacityRecord:
    index: int  # Zero-based data-record index in the original CSV, excluding blanks.
    csv_line: int  # One-based physical start line, including the header.
    image_path: Path
    steering: float
    timestamp: datetime


class UdacityDataset:
    """Read driving_log.csv under data_dir, using only center and steering.

    Relative image paths resolve against the CSV's directory. Paths and numeric
    fields are stripped. All referenced center files must exist. Images decode
    lazily, and labels are neither clipped nor rescaled. No input files change.
    """

    def __init__(self, data_dir: Union[str, Path]):
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.csv_path = self.data_dir / "driving_log.csv"
        records = []
        seen_paths = set()
        with self.csv_path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.reader(stream, skipinitialspace=True)
            header = [field.strip() for field in next(reader, [])]
            if len(set(header)) != len(header) or not {"center", "steering"}.issubset(header):
                raise ValueError("CSV needs unique column names including center and steering")
            center_column, steering_column = header.index("center"), header.index("steering")
            while True:
                csv_line = reader.line_num + 1
                row = next(reader, None)
                if row is None:
                    break
                if not row or not any(field.strip() for field in row):
                    continue
                context = "{}: line {}".format(self.csv_path, csv_line)
                if len(row) != len(header):
                    raise ValueError("{}: field count differs from header".format(context))
                raw_path = row[center_column].strip()
                if not raw_path:
                    raise ValueError("{}: empty center path".format(context))
                image_path = Path(raw_path).expanduser()
                if not image_path.is_absolute():
                    image_path = self.csv_path.parent / image_path
                image_path = image_path.resolve()
                if not image_path.is_file():
                    raise FileNotFoundError("{}: center image missing: {}".format(context, image_path))
                if image_path in seen_paths:
                    raise ValueError("{}: duplicate center image: {}".format(context, image_path))
                try:
                    steering = float(row[steering_column].strip())
                except ValueError as exc:
                    raise ValueError("{}: invalid steering".format(context)) from exc
                if not math.isfinite(steering) or abs(steering) > np.finfo(np.float32).max:
                    raise ValueError("{}: steering must be finite and representable as float32".format(context))
                try:
                    timestamp = datetime.strptime(image_path.stem, "center_%Y_%m_%d_%H_%M_%S_%f")
                except ValueError as exc:
                    raise ValueError("{}: center filename needs a Udacity timestamp".format(context)) from exc
                records.append(UdacityRecord(len(records), csv_line, image_path, steering, timestamp))
                seen_paths.add(image_path)
        if not records:
            raise ValueError("CSV contains no driving records: {}".format(self.csv_path))
        self.records = tuple(records)

    @property
    def indices(self) -> Tuple[int, ...]:
        """Original source indices, including when this dataset is a split view."""
        return tuple(record.index for record in self.records)

    @property
    def label(self) -> np.ndarray:
        """Labels for existing HyTorch dataset-analysis helpers."""
        return np.array([record.steering for record in self.records], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = preprocess_rgb(load_rgb_image(record.image_path))
        label = np.array([record.steering], dtype=np.float32)
        return image, label

    def _view(self, records):
        dataset = object.__new__(type(self))
        dataset.data_dir, dataset.csv_path = self.data_dir, self.csv_path
        dataset.records = tuple(records)
        return dataset

    def statistics(self) -> dict:
        labels = np.array([record.steering for record in self.records], dtype=np.float64)
        return {
            "count": len(self),
            "first_timestamp": self.records[0].timestamp.isoformat() if self.records else None,
            "last_timestamp": self.records[-1].timestamp.isoformat() if self.records else None,
            "steering_min": float(labels.min()) if labels.size else None,
            "steering_max": float(labels.max()) if labels.size else None,
            "steering_mean": float(labels.mean()) if labels.size else None,
            "steering_std": float(labels.std()) if labels.size else None,
            "zero_fraction": float((labels == 0).mean()) if labels.size else None,
        }

    def split_temporal(self, val_ratio: float = 0.1, gap_seconds: float = 2.0):
        """Hold out the last ceil(N * val_ratio) records, with a training embargo.

        Validation remains intact. Training retains only timestamps strictly
        earlier than validation_start - gap_seconds; intervening records are
        returned in excluded. All three views keep source order and indices.
        A split leaving no training records is rejected. No RNG is involved.
        """
        if not math.isfinite(val_ratio) or not 0 < val_ratio < 1:
            raise ValueError("val_ratio must be finite and strictly between 0 and 1")
        if not math.isfinite(gap_seconds) or gap_seconds < 0:
            raise ValueError("gap_seconds must be finite and nonnegative")
        if any(a.timestamp >= b.timestamp for a, b in zip(self.records, self.records[1:])):
            raise ValueError("CSV timestamps must be strictly increasing for temporal splitting")
        n_val = max(1, math.ceil(len(self) * val_ratio))
        if n_val >= len(self):
            raise ValueError("Temporal split must retain training records")
        boundary = len(self) - n_val
        validation_start = self.records[boundary].timestamp
        if (validation_start - self.records[0].timestamp).total_seconds() <= gap_seconds:
            raise ValueError("Isolation window leaves no training records")
        cutoff = validation_start - timedelta(seconds=gap_seconds)
        train_end = boundary
        while train_end and self.records[train_end - 1].timestamp >= cutoff:
            train_end -= 1
        if not train_end:
            raise ValueError("Isolation window leaves no training records")
        return TemporalSplit(
            train=self._view(self.records[:train_end]),
            validation=self._view(self.records[boundary:]),
            excluded=self._view(self.records[train_end:boundary]),
            val_ratio=val_ratio,
            gap_seconds=gap_seconds,
            validation_start=validation_start,
        )


@dataclass(frozen=True)
class TemporalSplit:
    train: UdacityDataset
    validation: UdacityDataset
    excluded: UdacityDataset
    val_ratio: float
    gap_seconds: float
    validation_start: datetime

    def to_dict(self) -> dict:
        """JSON-ready split indices, source CSV lines, rules and label statistics."""
        result = {
            "csv_path": str(self.train.csv_path),
            "source_index_base": 0,
            "csv_line_base": 1,
            "val_ratio": self.val_ratio,
            "gap_seconds": self.gap_seconds,
            "validation_start": self.validation_start.isoformat(),
            "rule": "validation=last ceil(N*ratio); train timestamp < validation_start-gap",
            "timezone": "unspecified (filename wall time)",
        }
        for name in ("train", "validation", "excluded"):
            dataset = getattr(self, name)
            result[name] = dict(dataset.statistics(), indices=list(dataset.indices),
                                csv_lines=[record.csv_line for record in dataset.records])
        return result

    def save_manifest(self, path: Union[str, Path]):
        """Write a new JSON manifest; refuse to overwrite any existing file."""
        path = Path(path).expanduser()
        with path.open("x", encoding="utf-8") as stream:
            json.dump(self.to_dict(), stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
