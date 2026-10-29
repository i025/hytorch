"""Arithmetic-semantic regressions, not performance tests."""
import numpy as np
from accuracy import sequential_fma,pairwise_product,blocked_fma,sequential_fp64,ulp_distance


def test_fma_has_one_rounding_and_pairwise_rounds_products():
    a=np.array([-1,1+2**-23],np.float32);b=np.array([1,1-2**-23],np.float32)
    assert sequential_fma(a,b)==np.float32(-2**-46)
    assert pairwise_product(a,b)==np.float32(0)
    assert np.float32(sequential_fp64(a,b))==np.float32(-2**-46)


def test_odd_length_tree_and_blocks_preserve_dtype():
    a=np.arange(1,8,dtype=np.float32);b=np.ones(7,dtype=np.float32)
    for value in [sequential_fma(a,b),pairwise_product(a,b),blocked_fma(a,b,3),blocked_fma(a,b,32)]:
        assert value==28 and value.dtype==np.float32


def test_ulp_ordering_across_negative_and_zero():
    assert ulp_distance(np.float32(-0.),np.float32(0.))==0
    assert ulp_distance(np.float32(-1),np.nextafter(np.float32(-1),np.float32(0)))==1
    assert ulp_distance(np.float32(1),np.nextafter(np.float32(1),np.float32(2)))==1
    assert ulp_distance(np.nextafter(np.float32(0),np.float32(-1)),np.nextafter(np.float32(0),np.float32(1)))==2


def test_pairwise_is_not_universally_better():
    a=np.array([1e8,1,-1e8,1],np.float32);b=np.ones(4,np.float32)
    assert sequential_fp64(a,b)==2
    assert sequential_fma(a,b)==1
    assert pairwise_product(a,b)==0
