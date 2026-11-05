"""CPU mathematical reference and FP32 lane/tree model; never executes CUDA."""
import numpy as np
from hytorch.backends.cuda.reduction import CHUNK, plan_sum


def reference_sum(x, axis=None, keepdims=False):
    return x.astype(np.float64).sum(axis=axis, keepdims=keepdims).astype(np.float32)


def block_sum(rows):
    values = np.zeros((len(rows), 256), np.float32)
    for start in range(0, rows.shape[1], 256):
        n = min(256, rows.shape[1]-start)
        np.add(values[:, :n], rows[:, start:start+n], out=values[:, :n])
    warps = values.reshape(-1, 8, 32)
    for delta in (16, 8, 4, 2, 1):
        old = warps.copy()
        warps[:, :, :32-delta] = np.add(old[:, :, :32-delta], old[:, :, delta:], dtype=np.float32)
    values = np.zeros((len(rows), 32), np.float32)
    values[:, :8] = warps[:, :, 0]
    for delta in (16, 8, 4, 2, 1):
        old = values.copy()
        values[:, :32-delta] = np.add(old[:, :32-delta], old[:, delta:], dtype=np.float32)
    return values[:, 0]


def modeled_sum(x, axis=None, keepdims=False):
    layout, shape, parts = plan_sum(x.shape, axis, keepdims)
    kept = [i for i in range(x.ndim) if i not in axes_of(x.ndim, axis)]
    axes = axes_of(x.ndim, axis)
    if not axes:
        return np.add(np.float32(0), x).reshape(shape)
    matrix = x.transpose(kept + list(axes)).reshape(layout.outputs, layout.reduction)
    if parts > 1:
        padded = np.zeros((layout.outputs, parts*CHUNK), np.float32)
        padded[:, :layout.reduction] = matrix
        partial = block_sum(padded.reshape(-1, CHUNK)).reshape(layout.outputs, parts)
        result = block_sum(partial)
    else:
        result = block_sum(matrix)
    return result.reshape(shape)


def axes_of(ndim, axis):
    if axis is None:
        return tuple(range(ndim))
    if ndim == 0 and isinstance(axis, (int, np.integer)) and axis in (0, -1):
        return ()
    return tuple(sorted(np.core.numeric.normalize_axis_tuple(axis, ndim)))
