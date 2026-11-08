import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from hytorch import cuda
from hytorch.core import Variable


def make_run_id(prefix=None):
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if prefix:
        return f"{prefix}-{stamp}"
    return stamp


def _json_default(value):
    if isinstance(value, Variable):
        value = value.data
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "get"):
        value = cuda.as_numpy(value)
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


class MetricRecorder:
    def __init__(self, run_dir=None, root_dir="runs", run_id=None, config=None, overwrite=False):
        if run_dir is None:
            run_id = run_id or make_run_id()
            run_dir = Path(root_dir) / run_id
        self.run_dir = Path(run_dir)
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.artifact_dir = self.run_dir / "artifacts"
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.summary_path = self.run_dir / "summary.json"
        self.config_path = self.run_dir / "config.json"

        if self.run_dir.exists() and overwrite:
            for child in self.run_dir.iterdir():
                if child.is_file():
                    child.unlink()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

        self._latest = {}
        self._start_time = time.time()
        if config is not None:
            self.save_config(config)

    @property
    def run_id(self):
        return self.run_dir.name

    def save_config(self, config):
        with self.config_path.open("w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2, default=_json_default)

    def log(self, record_type="metric", step=None, epoch=None, phase=None, **metrics):
        record = {
            "time": time.time(),
            "elapsed": time.time() - self._start_time,
            "type": record_type,
        }
        if step is not None:
            record["step"] = int(step)
        if epoch is not None:
            record["epoch"] = int(epoch)
        if phase is not None:
            record["phase"] = phase
        record.update(metrics)
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
        self._latest.update(record)
        return record

    def log_batch(self, step, epoch, phase="train", **metrics):
        return self.log("batch", step=step, epoch=epoch, phase=phase, **metrics)

    def log_epoch(self, epoch, phase="train", **metrics):
        return self.log("epoch", epoch=epoch, phase=phase, **metrics)

    def save_summary(self, summary=None, **extra):
        payload = dict(summary or {})
        payload.update(extra)
        payload.setdefault("run_id", self.run_id)
        payload.setdefault("run_dir", str(self.run_dir))
        payload.setdefault("updated_at", datetime.now().isoformat(timespec="seconds"))
        with self.summary_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=_json_default)
        return payload

    def load_metrics(self):
        if not self.metrics_path.exists():
            return []
        records = []
        with self.metrics_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def latest(self):
        return dict(self._latest)

    def artifact_path(self, name):
        return self.artifact_dir / name

    def checkpoint_path(self, name):
        return self.checkpoint_dir / name
