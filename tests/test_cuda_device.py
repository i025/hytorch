"""Real CUDA tests: skipped on Mac; required by scripts/validate_cuda.py."""
import os
import numpy as np
import pytest
from hytorch import Variable
import hytorch.functions as F
from hytorch.functions_conv import conv2d, pooling, conv2d_cpu_tensordot_array
from hytorch.backends.cuda import backend, dispatch_stats
from hytorch.backends.cuda.runtime import gemm, load_library, max_pool2d


@pytest.fixture(scope="module")
def cp():
    try:
        import cupy
        if cupy.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("No CUDA device")
        cupy.cuda.Device(int(os.environ.get("HYTORCH_TEST_DEVICE","0"))).use()
    except Exception as e:
        if os.environ.get("HYTORCH_REQUIRE_CUDA")=="1":
            pytest.fail(f"Required GPU unavailable: {e}")
        pytest.skip(f"CUDA NOT EXECUTED: {e}")
    load_library()  # GPU with missing/stale library must fail, not skip.
    return cupy


@pytest.mark.parametrize("shape",[(1,1,1),(17,31,23),(35,65,19),(32,32,32)])
@pytest.mark.parametrize("variant",["basic","blocked","shared"])
@pytest.mark.parametrize("tile",[16,32])
def test_gemm_rectangular_and_current_stream(cp,shape,variant,tile):
    m,k,n=shape
    rng=np.random.default_rng(123)
    a=rng.normal(size=(m,k)).astype(np.float32)
    b=rng.normal(size=(k,n)).astype(np.float32)
    stream=cp.cuda.Stream(non_blocking=True)
    with stream:
        da,db=cp.asarray(a),cp.asarray(b)
        actual=gemm(da,db,variant,tile)
        # Consumer on same non-default stream tests producer/consumer ordering.
        actual=actual+np.float32(.125)
        stream.synchronize()
        result=cp.asnumpy(actual)
    np.testing.assert_allclose(result,a@b+.125,atol=2e-5,rtol=2e-5)


def test_gemm_views_and_alias_rejection(cp):
    a=cp.arange(35,dtype=cp.float32).reshape(5,7).T
    b=cp.ones((5,3),cp.float32)
    np.testing.assert_allclose(cp.asnumpy(gemm(a,b)),cp.asnumpy(a@b),atol=1e-5)
    with pytest.raises(TypeError):
        gemm(a.astype(cp.float64),b)
    square=cp.ones((3,3),cp.float32)
    with pytest.raises(ValueError,match="overlap"):
        gemm(square,square,out=square)


@pytest.mark.parametrize("variant",["direct","im2col"])
@pytest.mark.parametrize("bias",[False,True])
@pytest.mark.parametrize("stride,pad",[((1,1),(0,0)),((2,1),(1,2))])
def test_conv_forward_backward(cp,variant,bias,stride,pad):
    rng=np.random.default_rng(1)
    arrays=[rng.normal(size=s).astype(np.float32) for s in [(2,3,9,8),(5,3,3,2),(5,)]]
    outputs=[]
    for mode in ("cupy",variant):
        x,w,b=[Variable(cp.asarray(a)) for a in arrays]
        with backend(conv=mode):
            y=conv2d(x,w,b if bias else None,stride,pad)
            loss=(y*y).sum()
            loss.backward()
            cp.cuda.get_current_stream().synchronize()
        outputs.append([cp.asnumpy(v) for v in [y.data,x.grad.data,w.grad.data]+([b.grad.data] if bias else [])])
    for actual,expected in zip(outputs[1],outputs[0]):
        np.testing.assert_allclose(actual,expected,atol=2e-3,rtol=2e-4)
    reference=conv2d_cpu_tensordot_array(arrays[0],arrays[1],arrays[2] if bias else None,stride,pad)
    np.testing.assert_allclose(outputs[1][0],reference,atol=1e-4,rtol=1e-4)


@pytest.mark.parametrize("pad",[0,1])
def test_pool_ties_negative_padding_backward(cp,pad):
    a=-np.ones((2,3,5,7),np.float32)
    a[:,:,1:3,2:4]=2
    outputs=[]
    for mode in ("cupy","native"):
        x=Variable(cp.asarray(a))
        with backend(pool=mode):
            y=pooling(x,(3,2),stride=(1,2),pad=pad)
            y.sum().backward()
            cp.cuda.get_current_stream().synchronize()
            outputs.append((cp.asnumpy(y.data),cp.asnumpy(x.grad.data)))
    for actual,expected in zip(outputs[1],outputs[0]):
        np.testing.assert_array_equal(actual,expected)


def test_pool_nan_propagates_first_index(cp):
    a=cp.asarray([[[[1.,np.nan],[np.nan,3.]]]],dtype=cp.float32)
    y,indexes=max_pool2d(a,2,2)
    assert np.isnan(cp.asnumpy(y)).all()
    assert int(indexes.item())==1


def test_framework_linear_forward_native_backward_original(cp):
    rng=np.random.default_rng(5)
    a=rng.normal(size=(3,7)).astype(np.float32)
    b=rng.normal(size=(7,5)).astype(np.float32)
    x,w=Variable(cp.asarray(a)),Variable(cp.asarray(b))
    dispatch_stats(reset=True)
    with backend(matmul="shared"):
        y=F.linear(x,w)
        y.sum().backward()
        cp.cuda.get_current_stream().synchronize()
    np.testing.assert_allclose(cp.asnumpy(y.data),a@b,atol=1e-5)
    np.testing.assert_allclose(cp.asnumpy(x.grad.data),np.ones((3,5))@b.T,atol=1e-5)
    np.testing.assert_allclose(cp.asnumpy(w.grad.data),a.T@np.ones((3,5)),atol=1e-5)
    assert dispatch_stats().get("gemm_shared")==1
