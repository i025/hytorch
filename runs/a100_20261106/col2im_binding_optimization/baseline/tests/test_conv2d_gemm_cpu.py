import unittest
import numpy as np

from hytorch import Variable
from hytorch.functions_conv import (
    conv2d,
    conv2d_cpu_tensordot_array,
    conv2d_cpu_im2col_gemm_array,
)


class TestConv2dGemmCPU(unittest.TestCase):
    def test_non4_multiple_shape_matches_legacy(self):
        rng = np.random.default_rng(123)
        x = rng.standard_normal((3, 5, 11, 10), dtype=np.float32)   # N, IC, H, W
        w = rng.standard_normal((7, 5, 3, 2), dtype=np.float32)     # OC, IC, KH, KW
        b = rng.standard_normal((7,), dtype=np.float32)             # OC
        stride = (2, 1)
        pad = (1, 0)

        y_old = conv2d_cpu_tensordot_array(x, w, b, stride=stride, pad=pad)
        y_new = conv2d_cpu_im2col_gemm_array(x, w, b, stride=stride, pad=pad)
        np.testing.assert_allclose(y_new, y_old, rtol=1e-5, atol=1e-5)

    def test_forward_api_uses_new_cpu_path(self):
        rng = np.random.default_rng(7)
        x = rng.standard_normal((2, 3, 9, 7), dtype=np.float32)
        w = rng.standard_normal((5, 3, 3, 3), dtype=np.float32)
        b = rng.standard_normal((5,), dtype=np.float32)
        stride = 1
        pad = 1

        y_expected = conv2d_cpu_im2col_gemm_array(x, w, b, stride=stride, pad=pad)
        y_api = conv2d(Variable(x), Variable(w), Variable(b), stride=stride, pad=pad).data
        np.testing.assert_allclose(y_api, y_expected, rtol=1e-5, atol=1e-5)

    def test_no_bias_case_matches_legacy(self):
        rng = np.random.default_rng(99)
        x = rng.standard_normal((1, 2, 8, 6), dtype=np.float32)
        w = rng.standard_normal((3, 2, 2, 3), dtype=np.float32)
        stride = (1, 2)
        pad = (0, 1)

        y_old = conv2d_cpu_tensordot_array(x, w, None, stride=stride, pad=pad)
        y_new = conv2d_cpu_im2col_gemm_array(x, w, None, stride=stride, pad=pad)
        np.testing.assert_allclose(y_new, y_old, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
