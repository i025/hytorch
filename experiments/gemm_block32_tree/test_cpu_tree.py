import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'docs/experiments/gemm-fp32-accuracy/scripts'))
from accuracy import pairwise_sum, sequential_fma, blocked_fma
from experiments.gemm_block32_tree.cpu_tree import online_tree


@pytest.mark.parametrize('blocks', [1, 2, 3, 4, 5, 6, 7, 9, 11, 31, 32, 33, 63, 64, 65, 127, 128])
def test_online_tree_matches_adjacent_pairwise(blocks):
    rng = np.random.default_rng(500 + blocks)
    partials = rng.standard_normal(blocks, dtype=np.float32) * np.float32(100)
    assert online_tree(partials).view(np.uint32) == pairwise_sum(partials).view(np.uint32)


@pytest.mark.parametrize('k', [31, 32, 33, 65, 127, 255, 4095, 4096])
def test_fma_blocks_match_stage3a(k):
    rng = np.random.default_rng(700 + k)
    a = rng.standard_normal(k, dtype=np.float32)
    b = rng.standard_normal(k, dtype=np.float32)
    partials = [sequential_fma(a[i:i+32], b[i:i+32]) for i in range(0, k, 32)]
    assert online_tree(partials).view(np.uint32) == blocked_fma(a, b, 32).view(np.uint32)


def test_right_to_left_final_fold_is_required():
    partials = np.array([1e8, 0, 0, 0, -1e8, 0, 1], dtype=np.float32)
    assert online_tree(partials) == pairwise_sum(partials) == np.float32(0)
    # A left-to-right fold of the three remaining subtrees would incorrectly give 1.
    assert np.float32(np.float32(partials[0] + partials[4]) + partials[6]) == 1
