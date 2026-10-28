"""Versioned configuration shared by Launcher, PS, Workers and external agents."""
import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass
class TrainingConfig:
    schema_version: int = 1
    task_factory: str = "donkeycar_hy.ps_task:DonkeyTask"
    data_dir: str = "D:\\python\\_project\\hytorch_gpu_transformer\\data\\cardata"
    output_dir: str = "runs/distributed"
    run_name: str = "donkey-ps"
    mode: str = "ps"
    transport: str = "socket_json"
    num_workers: int = 2
    device: str = "gpu"
    gpu_id: int = 0
    gpu_memory_limit_mb: int = 0
    epochs: int = 5
    global_batch_size: int = 8
    eval_batch_size: int = 8
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    optimizer: str = "adamw"
    seed: int = 42
    val_ratio: float = 0.1
    dropout: float = 0.0
    augment: bool = False
    host: str = "127.0.0.1"
    port: int = 0
    rpc_timeout_s: float = 300.0
    startup_timeout_s: float = 300.0
    run_timeout_s: float = 86400.0
    monitor_interval_s: float = 1.0
    max_message_mb: int = 256
    benchmark_repeats: int = 1

    @classmethod
    def from_dict(cls, values):
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        config = cls(**values)
        config.validate()
        return config

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8-sig")))

    def to_dict(self):
        return asdict(self)

    def validate(self, require_data=False):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported configuration schema_version")
        for name in ("num_workers", "epochs", "global_batch_size", "eval_batch_size",
                     "max_message_mb", "benchmark_repeats"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("gpu_id", "gpu_memory_limit_mb", "seed", "port"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.port > 65535 or self.seed >= 2**32:
            raise ValueError("port or seed out of range")
        for name in ("learning_rate", "rpc_timeout_s", "startup_timeout_s",
                     "run_timeout_s", "monitor_interval_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("val_ratio", "dropout", "weight_decay"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 0 < self.val_ratio < 1 or not 0 <= self.dropout < 1 or self.weight_decay < 0:
            raise ValueError("Invalid val_ratio, dropout or weight_decay")
        if type(self.augment) is not bool:
            raise ValueError("augment must be boolean")
        if self.mode not in ("single", "ps", "ring") or self.transport not in ("socket_json", "grpc_protobuf"):
            raise ValueError("Unsupported mode or transport")
        if self.mode == "single" and self.num_workers != 1:
            raise ValueError("single mode requires num_workers=1")
        if self.mode == "ring" and (self.num_workers < 2 or self.transport != "grpc_protobuf" or self.port != 0):
            raise ValueError("ring requires num_workers>=2, grpc_protobuf and port=0 (one dynamic endpoint per worker)")
        if self.device not in ("cpu", "gpu") or self.optimizer not in ("sgd", "adamw"):
            raise ValueError("Unsupported device or optimizer")
        if self.host != "127.0.0.1":
            raise ValueError("This local simulation listens on 127.0.0.1 only")
        if not isinstance(self.task_factory, str) or self.task_factory.count(":") != 1:
            raise ValueError("task_factory must be module:class")
        for name in ("data_dir", "output_dir", "run_name"):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"{name} must be a string")
        if not self.output_dir or not self.run_name or any(c in self.run_name for c in '/\\:<>"|?*') or self.run_name in (".", ".."):
            raise ValueError("Invalid output_dir or run_name")
        if require_data and (not self.data_dir.strip() or not Path(self.data_dir).is_dir()):
            raise ValueError("Fill data_dir in configs/donkey_ps.json, or pass --data D:/your/dataset")
        return self
