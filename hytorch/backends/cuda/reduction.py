"""CPU-only shape/axis planning for the optional FP32 CUDA sum."""
import ctypes as ct
import math

import numpy as np

MAX_DIMS = 8
CHUNK = 4096


class SumLayout(ct.Structure):
    _fields_ = [('out_ndim', ct.c_int), ('reduce_ndim', ct.c_int),
                ('outputs', ct.c_int), ('reduction', ct.c_int),
                ('out_shape', ct.c_int * MAX_DIMS), ('out_stride', ct.c_int * MAX_DIMS),
                ('reduce_shape', ct.c_int * MAX_DIMS), ('reduce_stride', ct.c_int * MAX_DIMS)]


def plan_sum(shape, axis=None, keepdims=False):
    shape = tuple(shape)
    ndim = len(shape)
    if ndim > MAX_DIMS:
        raise ValueError('Native sum supports at most 8 dimensions')
    if axis is None:
        axes = tuple(range(ndim))
    elif ndim == 0 and isinstance(axis, (int, np.integer)) and axis in (0, -1):
        axes = ()
    else:
        axes = tuple(sorted(np.core.numeric.normalize_axis_tuple(axis, ndim)))
    kept = tuple(i for i in range(ndim) if i not in axes)
    strides = [math.prod(shape[i+1:]) for i in range(ndim)]
    layout = SumLayout()
    layout.out_ndim, layout.reduce_ndim = len(kept), len(axes)
    layout.outputs = math.prod(shape[i] for i in kept)
    layout.reduction = math.prod(shape[i] for i in axes)
    for j, i in enumerate(kept):
        layout.out_shape[j], layout.out_stride[j] = shape[i], strides[i]
    for j, i in enumerate(axes):
        layout.reduce_shape[j], layout.reduce_stride[j] = shape[i], strides[i]
    output = tuple(1 if i in axes else shape[i] for i in range(ndim)) if keepdims else tuple(shape[i] for i in kept)
    return layout, output, (layout.reduction + CHUNK - 1) // CHUNK
