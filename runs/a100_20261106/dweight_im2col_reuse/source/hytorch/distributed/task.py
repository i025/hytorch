"""Adapter boundary: other groups can supply a module:class factory."""
import importlib
import random
from typing import Protocol

import numpy as np
from hytorch import cuda, optimizers


class TrainingTask(Protocol):
    def prepare_manifest(self): ...
    def build_model(self): ...
    def load_split(self, manifest, split): ...
    def loss(self, prediction, target): ...
    def batch_statistics(self, prediction, target): ...
    def quality(self, totals): ...


def create_task(config):
    module, name = config.task_factory.split(":")
    return getattr(importlib.import_module(module), name)(config)


def seed_process(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))


def select_device(config):
    if config.device == "cpu":
        return np
    if not cuda.gpu_enable:
        raise RuntimeError("GPU requested but CuPy is unavailable in this Python environment")
    cp = cuda.cupy
    cp.cuda.Device(config.gpu_id).use()
    if config.gpu_memory_limit_mb:
        cp.get_default_memory_pool().set_limit(size=config.gpu_memory_limit_mb * 1024**2)
    cp.random.seed(config.seed % (2**32))
    return cp


def synchronize(xp):
    if xp is not np:
        xp.cuda.get_current_stream().synchronize()


def named_parameters(model):
    flat = {}
    model._flatten_params(flat)
    result = dict(sorted(flat.items()))
    if not result or any(p.data is None for p in result.values()):
        raise ValueError("Task.build_model() must return a fully initialized model")
    if len({id(p) for p in result.values()}) != len(result):
        raise ValueError("Shared/tied parameters need an explicit task adapter")
    return result


def export_parameters(params):
    return {name: np.array(cuda.as_numpy(p.data), copy=True) for name, p in params.items()}


def normalize_gradients(model, xp):
    """Keep optimizer and wire gradients in the corresponding parameter dtype.

    Existing scalar operations can promote GPU backward results to float64 even
    when model parameters are float32. Apply the same policy to PS and baseline.
    """
    for name, param in named_parameters(model).items():
        if param.grad is None:
            raise ValueError(f"No gradient for parameter {name}")
        gradient = param.grad.data
        if gradient.shape != param.data.shape or gradient.dtype.kind != "f":
            raise ValueError(f"Invalid gradient shape/type: {name}")
        gradient = xp.asarray(gradient, dtype=param.data.dtype)
        if not bool(xp.isfinite(gradient).all()):
            raise FloatingPointError(f"Non-finite gradient after dtype conversion: {name}")
        param.grad.data = gradient


def load_parameters(params, values, xp):
    if set(params) != set(values):
        raise ValueError("Parameter names do not match the local model")
    for name, param in params.items():
        value = values[name]
        if value.shape != param.data.shape or value.dtype != param.data.dtype:
            raise ValueError(f"Parameter shape/dtype mismatch: {name}")
        if not np.isfinite(value).all():
            raise ValueError(f"Non-finite parameter: {name}")
        param.data[...] = xp.asarray(value)


def make_optimizer(config, model):
    if config.optimizer == "sgd":
        optimizer = optimizers.SGD(lr=config.learning_rate)
    else:
        optimizer = optimizers.AdamW(lr=config.learning_rate, weight_decay=config.weight_decay)
    return optimizer.setup(model)


def epoch_batches(size, batch_size, seed, epoch):
    order = np.random.RandomState((seed + epoch) % (2**32)).permutation(size)
    for start in range(0, size, batch_size):
        yield order[start:start + batch_size]


def load_batch(dataset, indices, xp):
    batch = [dataset[int(index)] for index in indices]
    x = np.stack([item[0] for item in batch]).astype(np.float32, copy=False)
    target = np.stack([item[1] for item in batch]).astype(np.float32, copy=False)
    return xp.asarray(x), xp.asarray(target)
