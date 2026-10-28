import math
from collections import defaultdict

import numpy as np

from hytorch import cuda
from hytorch.core import Variable


def to_numpy(x):
    """Convert HyTorch/CuPy/NumPy/scalar values to a NumPy array."""
    if isinstance(x, Variable):
        x = x.data
    if x is None:
        return None
    x = cuda.as_numpy(x)
    return np.asarray(x)


def to_float(x):
    arr = to_numpy(x)
    if arr is None:
        return None
    return float(np.asarray(arr).mean())


def _labels_from_prediction(y_pred):
    y = to_numpy(y_pred)
    if y.ndim > 1:
        return y.argmax(axis=1)
    return y.astype(np.int64)


def _labels(y_true):
    t = to_numpy(y_true)
    if t.ndim > 1:
        return t.argmax(axis=1)
    return t.astype(np.int64).reshape(-1)


def accuracy_score(y_pred, y_true):
    pred = _labels_from_prediction(y_pred).reshape(-1)
    true = _labels(y_true)
    if true.size == 0:
        return 0.0
    return float((pred == true).mean())


def top_k_accuracy(y_pred, y_true, k=5):
    y = to_numpy(y_pred)
    true = _labels(y_true)
    if y.ndim == 1:
        return accuracy_score(y, true)
    k = min(int(k), y.shape[1])
    topk = np.argsort(y, axis=1)[:, -k:]
    hits = [true[i] in topk[i] for i in range(len(true))]
    return float(np.mean(hits)) if hits else 0.0


def confusion_matrix(y_true, y_pred, num_classes=None):
    true = _labels(y_true)
    pred = _labels_from_prediction(y_pred).reshape(-1)
    if num_classes is None:
        max_label = 0
        if true.size:
            max_label = max(max_label, int(true.max()))
        if pred.size:
            max_label = max(max_label, int(pred.max()))
        num_classes = max_label + 1
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(true, pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            matrix[int(t), int(p)] += 1
    return matrix


def precision_recall_f1(y_true, y_pred, num_classes=None, average="macro", eps=1e-12):
    matrix = confusion_matrix(y_true, y_pred, num_classes)
    tp = np.diag(matrix).astype(np.float64)
    fp = matrix.sum(axis=0) - tp
    fn = matrix.sum(axis=1) - tp

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    if average is None or average == "none":
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    if average == "micro":
        total_tp = tp.sum()
        total_fp = fp.sum()
        total_fn = fn.sum()
        p = total_tp / (total_tp + total_fp + eps)
        r = total_tp / (total_tp + total_fn + eps)
        return {
            "precision": float(p),
            "recall": float(r),
            "f1": float(2 * p * r / (p + r + eps)),
        }
    if average == "weighted":
        support = matrix.sum(axis=1).astype(np.float64)
        weights = support / (support.sum() + eps)
        return {
            "precision": float((precision * weights).sum()),
            "recall": float((recall * weights).sum()),
            "f1": float((f1 * weights).sum()),
        }
    if average != "macro":
        raise ValueError("average must be one of: macro, micro, weighted, none")
    return {
        "precision": float(precision.mean()) if precision.size else 0.0,
        "recall": float(recall.mean()) if recall.size else 0.0,
        "f1": float(f1.mean()) if f1.size else 0.0,
    }


def classification_report_metrics(y_pred, y_true, num_classes=None, topk=(1, 5)):
    values = {}
    if 1 in topk:
        values["accuracy"] = accuracy_score(y_pred, y_true)
    for k in topk:
        if k != 1:
            values[f"top{k}_accuracy"] = top_k_accuracy(y_pred, y_true, k)
    values.update(precision_recall_f1(y_true, y_pred, num_classes=num_classes))
    return values


def mean_squared_error(y_pred, y_true):
    y = to_numpy(y_pred).astype(np.float64)
    t = to_numpy(y_true).astype(np.float64)
    return float(np.mean((y - t) ** 2))


def mean_absolute_error(y_pred, y_true):
    y = to_numpy(y_pred).astype(np.float64)
    t = to_numpy(y_true).astype(np.float64)
    return float(np.mean(np.abs(y - t)))


def root_mean_squared_error(y_pred, y_true):
    return math.sqrt(mean_squared_error(y_pred, y_true))


def r2_score(y_pred, y_true, eps=1e-12):
    y = to_numpy(y_pred).astype(np.float64)
    t = to_numpy(y_true).astype(np.float64)
    ss_res = np.sum((t - y) ** 2)
    ss_tot = np.sum((t - t.mean()) ** 2)
    return float(1.0 - ss_res / (ss_tot + eps))


def regression_report_metrics(y_pred, y_true):
    return {
        "mse": mean_squared_error(y_pred, y_true),
        "mae": mean_absolute_error(y_pred, y_true),
        "rmse": root_mean_squared_error(y_pred, y_true),
        "r2": r2_score(y_pred, y_true),
    }


class AverageMeter:
    def __init__(self, name=None):
        self.name = name
        self.reset()

    def reset(self):
        self.total = 0.0
        self.count = 0
        self.last = None

    def update(self, value, n=1):
        value = to_float(value)
        if value is None:
            return
        n = int(n)
        self.last = value
        self.total += value * n
        self.count += n

    @property
    def avg(self):
        if self.count == 0:
            return 0.0
        return self.total / self.count


class MetricAverager:
    def __init__(self):
        self.meters = defaultdict(AverageMeter)

    def update(self, values, n=1):
        for key, value in values.items():
            if value is None:
                continue
            self.meters[key].update(value, n)

    def summary(self):
        return {key: meter.avg for key, meter in self.meters.items()}

    def latest(self):
        return {key: meter.last for key, meter in self.meters.items()}
