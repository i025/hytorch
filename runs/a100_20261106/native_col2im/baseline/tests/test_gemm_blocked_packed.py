import unittest
import numpy as np

from hytorch.backends.cpu.gemm import cpu_gemm, c_backend_available


class TestGemmBlockedPacked(unittest.TestCase):
    def _assert_kernel_matches_numpy(self, dtype, m, n, k):
        rng = np.random.default_rng(2026)
        a = rng.standard_normal((m, k)).astype(dtype)
        b = rng.standard_normal((k, n)).astype(dtype)

        y_np = a @ b
        y_blk = cpu_gemm(a, b, kernel="blocked_packed_c")

        rtol = 1e-5 if dtype == np.float32 else 1e-10
        atol = 1e-5 if dtype == np.float32 else 1e-10
        np.testing.assert_allclose(y_blk, y_np, rtol=rtol, atol=atol)

    def test_float32_non4_shape(self):
        if not c_backend_available():
            self.skipTest("C backend unavailable")
        self._assert_kernel_matches_numpy(np.float32, m=7, n=10, k=13)

    def test_float64_non4_shape(self):
        if not c_backend_available():
            self.skipTest("C backend unavailable")
        self._assert_kernel_matches_numpy(np.float64, m=7, n=10, k=13)

    def test_float32_regular_shape(self):
        if not c_backend_available():
            self.skipTest("C backend unavailable")
        self._assert_kernel_matches_numpy(np.float32, m=64, n=96, k=80)

    def test_baseline_and_blocked_consistency(self):
        if not c_backend_available():
            self.skipTest("C backend unavailable")
        rng = np.random.default_rng(11)
        a = rng.standard_normal((31, 29), dtype=np.float32)
        b = rng.standard_normal((29, 27), dtype=np.float32)
        y_naive = cpu_gemm(a, b, kernel="naive_c")
        y_4x4 = cpu_gemm(a, b, kernel="4x4_c")
        y_blk = cpu_gemm(a, b, kernel="blocked_packed_c")
        np.testing.assert_allclose(y_naive, y_4x4, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(y_blk, y_4x4, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
