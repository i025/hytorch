"""Minimal experiment-only binding; callers own contiguous FP32 device buffers."""
import ctypes as ct
from pathlib import Path
import cupy as cp


class Block32Tree:
    def __init__(self, library):
        self.lib = ct.CDLL(str(Path(library).resolve()))
        self.lib.block32_tree_gemm.argtypes = [ct.c_void_p] * 3 + [ct.c_int] * 3 + [ct.c_void_p]
        self.lib.block32_tree_gemm.restype = ct.c_int
        self.lib.block32_tree_resources.argtypes = [ct.c_void_p, ct.POINTER(ct.c_int)]
        self.lib.block32_tree_resources.restype = ct.c_int

    def call(self, a, b, out):
        assert a.dtype == b.dtype == out.dtype == cp.float32
        assert a.flags.c_contiguous and b.flags.c_contiguous and out.flags.c_contiguous
        m, k = a.shape
        assert b.shape[0] == k and out.shape == (m, b.shape[1])
        return self.prepared(a, b, out)()

    def prepared(self, a, b, out):
        m, k = a.shape
        arguments = (a.data.ptr, b.data.ptr, out.data.ptr, m, b.shape[1], k,
                     cp.cuda.get_current_stream().ptr)

        def launch():
            status = self.lib.block32_tree_gemm(*arguments)
            if status:
                raise RuntimeError(f'block32_tree_gemm CUDA status {status}')
            return out
        return launch

    def resources(self, symbol=None):
        values = (ct.c_int * 6)()
        status = self.lib.block32_tree_resources(symbol, values)
        if status:
            raise RuntimeError(f'block32_tree_resources CUDA status {status}')
        return dict(zip(('registers_per_thread', 'shared_bytes_per_block', 'local_bytes_per_thread',
                         'max_threads_per_block', 'theoretical_active_blocks_per_sm', 'binary_version'), values))
