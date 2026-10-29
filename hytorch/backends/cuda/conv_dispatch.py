"""Offline-calibrated A100 pointwise dInput selection; no runtime tuning.

Broader 3x3/5x5 rules failed independent validation. Keep those on original.
Evidence: docs/experiments/conv-shape-dispatch/results/pointwise-frozen-policy.json.
The current convolution API has NCHW/OIHW, groups=1 and dilation=1 semantics.
"""
from hytorch.utils import pair


def select_conv_dinput(input_shape, weight_shape, stride=1, pad=0,
                      dtype="float32", contiguous=True):
    """Return (implementation, reason) using operator metadata only.

    Other dtypes/layouts/geometries and shapes outside the calibrated envelope
    retain original; explicit fused is independent of this conservative rule.
    """
    co, wc, kh, kw = weight_shape
    if (kh, kw) != (1, 1) or pair(stride) != (1, 1) or pair(pad) != (0, 0):
        return "original", "geometry"
    if dtype != "float32" or not contiguous:
        return "original", "dtype_layout"
    n, ci, h, w = input_shape
    if not (1 <= n <= 16 and 3 <= ci <= 64 and 16 <= co <= 64
            and 3 <= h <= 65 and 5 <= w <= 67 and ci == wc):
        return "original", "shape"
    if n * ci * h * w * co > 1161216:
        return "original", "work"
    return "fused", "pointwise"
