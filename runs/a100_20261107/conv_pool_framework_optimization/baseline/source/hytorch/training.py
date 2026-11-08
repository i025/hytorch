import time
import numbers

from hytorch.callbacks import CallbackList, HistoryCallback, MetricLoggerCallback
from hytorch import cuda
from hytorch.core import Variable, no_grad, test_mode
from hytorch.metrics import MetricAverager, to_float


def _as_variable(x):
    if isinstance(x, Variable):
        return x
    return Variable(x)


def _batch_size(x, t):
    try:
        return int(len(t))
    except TypeError:
        try:
            return int(len(x))
        except TypeError:
            return 1


def _numeric_items(values):
    return {
        key: value
        for key, value in values.items()
        if isinstance(value, numbers.Number) or hasattr(value, "data")
    }


def _to_numpy(x):
    if isinstance(x, Variable):
        x = x.data
    if x is None:
        return None
    return cuda.as_numpy(x)


def _as_flat_array(x):
    arr = _to_numpy(x)
    if arr is None:
        return None
    return arr.reshape(-1)


def _histogram(values, bins=32):
    if values is None or values.size == 0:
        return {"bins": [], "counts": []}
    counts, edges = __import__("numpy").histogram(values, bins=bins)
    return {
        "bins": [float(v) for v in edges.tolist()],
        "counts": [int(v) for v in counts.tolist()],
    }


def _heatmap_from_param(param, max_size=32):
    import numpy as np

    arr = _to_numpy(param)
    if arr is None:
        return []
    if arr.ndim == 4:
        heat = arr[0, 0]
    elif arr.ndim == 3:
        heat = arr[0]
    elif arr.ndim == 2:
        heat = arr
    elif arr.ndim == 1:
        side = int(np.ceil(np.sqrt(arr.size)))
        padded = np.zeros(side * side, dtype=arr.dtype)
        padded[:arr.size] = arr
        heat = padded.reshape(side, side)
    else:
        return []
    heat = np.asarray(heat, dtype=np.float64)
    heat = heat[:max_size, :max_size]
    return [[float(v) for v in row] for row in heat.tolist()]


def collect_model_diagnostics(model, bins=32):
    import numpy as np

    weights, grads = [], []
    heatmap = []
    if not hasattr(model, "params"):
        return {}
    for param in model.params():
        data = getattr(param, "data", None)
        if data is not None:
            flat = _as_flat_array(data)
            if flat is not None:
                weights.append(flat)
            if not heatmap:
                heatmap = _heatmap_from_param(data)
        grad = getattr(param, "grad", None)
        if grad is not None:
            flat_grad = _as_flat_array(grad)
            if flat_grad is not None:
                grads.append(flat_grad)

    diagnostics = {}
    if weights:
        all_weights = np.concatenate(weights).astype(np.float64)
        diagnostics.update({
            "weight_mean": float(all_weights.mean()),
            "weight_std": float(all_weights.std()),
            "weight_min": float(all_weights.min()),
            "weight_max": float(all_weights.max()),
            "weight_histogram": _histogram(all_weights, bins=bins),
        })
    if grads:
        all_grads = np.concatenate(grads).astype(np.float64)
        diagnostics.update({
            "grad_mean": float(all_grads.mean()),
            "grad_std": float(all_grads.std()),
            "global_grad_norm": float(np.linalg.norm(all_grads)),
            "zero_grad_ratio": float(np.mean(all_grads == 0)),
            "gradient_histogram": _histogram(all_grads, bins=bins),
        })
    if heatmap:
        diagnostics["weight_heatmap"] = heatmap
    return diagnostics


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        metrics=None,
        callbacks=None,
        recorder=None,
        log_batches=True,
        batch_augment=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metrics = metrics or {}
        self.batch_augment = batch_augment
        self.stop_training = False
        self.global_step = 0

        if getattr(self.optimizer, "target", None) is None:
            self.optimizer.setup(model)

        self.history_callback = HistoryCallback()
        callback_items = [self.history_callback]
        self.recorder = recorder
        if recorder is not None:
            callback_items.append(MetricLoggerCallback(recorder, log_batches=log_batches))
        callback_items.extend(callbacks or [])
        self.callbacks = CallbackList(callback_items)
        self.callbacks.set_trainer(self)

    @property
    def history(self):
        return self.history_callback.history

    def _compute_metrics(self, y, t):
        values = {}
        for name, metric in self.metrics.items():
            value = metric(y, t)
            if isinstance(value, dict):
                for sub_name, sub_value in value.items():
                    values[sub_name] = to_float(sub_value)
            else:
                values[name] = to_float(value)
        return values

    def _augment_batch(self, x, t):
        if self.batch_augment is None:
            return x, t, {}
        result = self.batch_augment(x, t)
        info = getattr(self.batch_augment, "last_info", {}) or {}
        if isinstance(result, tuple) and len(result) == 3:
            x, t, extra = result
            info.update(extra or {})
            return x, t, info
        x, t = result
        return x, t, info

    def train_one_epoch(self, train_loader, epoch=1):
        averager = MetricAverager()
        started = time.perf_counter()

        for batch_index, (x, t) in enumerate(train_loader, start=1):
            self.global_step += 1
            batch_started = time.perf_counter()
            self.callbacks.on_batch_begin(self.global_step, {"epoch": epoch, "phase": "train"})
            x, t, augment_info = self._augment_batch(x, t)

            y = self.model(_as_variable(x))
            loss = self.loss_fn(y, t)
            self.model.cleargrads()
            loss.backward()
            self.optimizer.update()

            n = _batch_size(x, t)
            logs = {"loss": to_float(loss)}
            logs.update(self._compute_metrics(y, t))
            logs["batch_time"] = time.perf_counter() - batch_started
            logs["samples_per_sec"] = n / logs["batch_time"] if logs["batch_time"] > 0 else 0.0
            logs["epoch"] = epoch
            logs["phase"] = "train"
            logs.update({f"augment_{key}": value for key, value in augment_info.items() if key != "permutation"})
            averager.update(_numeric_items({k: v for k, v in logs.items() if k not in {"epoch", "phase"}}), n)
            self.callbacks.on_batch_end(self.global_step, logs)

        summary = averager.summary()
        summary["epoch_time"] = time.perf_counter() - started
        return summary

    def evaluate(self, loader, prefix=None):
        evaluator = Evaluator(self.model, self.loss_fn, self.metrics)
        results = evaluator.evaluate(loader)
        if prefix:
            return {f"{prefix}_{key}": value for key, value in results.items()}
        return results

    def fit(self, train_loader, val_loader=None, epochs=1):
        self.stop_training = False
        self.callbacks.on_train_begin({"epochs": epochs})
        last_logs = {}

        for epoch in range(1, epochs + 1):
            if self.stop_training:
                break
            self.callbacks.on_epoch_begin(epoch, {})
            train_logs = self.train_one_epoch(train_loader, epoch)
            epoch_logs = dict(train_logs)
            if val_loader is not None:
                epoch_logs.update(self.evaluate(val_loader, prefix="val"))
            epoch_logs.update(collect_model_diagnostics(self.model))
            last_logs = epoch_logs
            self.callbacks.on_epoch_end(epoch, epoch_logs)

        self.callbacks.on_train_end(last_logs)
        return self.history


class Evaluator:
    def __init__(self, model, loss_fn, metrics=None):
        self.model = model
        self.loss_fn = loss_fn
        self.metrics = metrics or {}

    def _compute_metrics(self, y, t):
        values = {}
        for name, metric in self.metrics.items():
            value = metric(y, t)
            if isinstance(value, dict):
                for sub_name, sub_value in value.items():
                    values[sub_name] = to_float(sub_value)
            else:
                values[name] = to_float(value)
        return values

    def evaluate(self, loader):
        averager = MetricAverager()
        started = time.perf_counter()
        with no_grad(), test_mode():
            for x, t in loader:
                y = self.model(_as_variable(x))
                loss = self.loss_fn(y, t)
                n = _batch_size(x, t)
                logs = {"loss": to_float(loss)}
                logs.update(self._compute_metrics(y, t))
                averager.update(logs, n)
        summary = averager.summary()
        summary["epoch_time"] = time.perf_counter() - started
        return summary
