"""Binding cache safety and optional ABI compatibility; no shape-specific fast paths."""
import struct
import numpy as np
import pytest
from hytorch.backends.cuda import runtime as rt
from test_col2im_cuda import cp, scatter_reference


def test_metadata_pod_layout_and_bound_cpu():
    rt._col2im_metadata.cache_clear()
    for w in range(1,151):
        args=((1,1,1,w),(1,1,1,1,1,w),(4*w,4*w,4*w,4*w,4*w,4),(1,1),(0,0))
        values,packed=rt._col2im_metadata(*args)
        assert isinstance(values,tuple) and isinstance(packed,bytes)
        assert len(packed)==144 and struct.unpack('=18q',packed)==values
    assert rt._col2im_metadata.cache_info().currsize==128


def test_cache_changes_buffers_streams(cp):
    rt._col2im_metadata.cache_clear()
    for repeat in range(3):
        for w in [5,7,5]:
            host=np.arange(2*3*3*5*w,dtype=np.float32).reshape(1,2,3,3,5,w)
            col=cp.asarray(host);shape=(1,2,5,w)
            if repeat==1:col=col[...,::-1]
            if repeat==2:col=cp.broadcast_to(col[...,:1],col.shape)
            out=cp.empty(shape,cp.float32)
            cp.cuda.get_current_stream().synchronize()
            stream=cp.cuda.Stream(non_blocking=True)
            with stream:
                rt.col2im(col,shape,pad=1,out=out)
            stream.synchronize()
            np.testing.assert_array_equal(cp.asnumpy(out),scatter_reference(cp.asnumpy(col),shape,(1,1),(1,1)))
    # Same metadata, different live addresses: no pointers retained in the cache.
    a=cp.ones((1,1,1,1,2,3),cp.float32);b=a*7
    np.testing.assert_array_equal(cp.asnumpy(rt.col2im(a,(1,1,2,3))),np.ones((1,1,2,3),np.float32))
    np.testing.assert_array_equal(cp.asnumpy(rt.col2im(b,(1,1,2,3))),np.full((1,1,2,3),7,np.float32))
    assert rt._col2im_metadata.cache_info().hits>0


def test_warm_cache_checks_each_call(cp,monkeypatch):
    col=cp.ones((1,1,1,1,2,3),cp.float32);shape=(1,1,2,3)
    rt.col2im(col,shape)
    with pytest.raises(TypeError):rt.col2im(col,(1.,1,2,3))
    with pytest.raises(TypeError):rt.col2im(col.astype(cp.float64),shape)
    for out in [cp.empty(shape,cp.float64),cp.empty((1,1,3,2),cp.float32),cp.empty((1,1,2,6),cp.float32)[...,::2],col.reshape(shape)]:
        with pytest.raises(ValueError):rt.col2im(col,shape,out=out)
    # Mock current-device ID, without touching another user's GPU.
    with monkeypatch.context() as m:
        m.setattr(cp.cuda.runtime,'getDevice',lambda:col.device.id+1)
        with pytest.raises(ValueError,match='current CUDA device'):rt.col2im(col,shape)
    class WrongDeviceArray(cp.ndarray):
        @property
        def device(self):
            class D:id=999
            return D()
    wrong=cp.empty(shape,cp.float32).view(type=WrongDeviceArray)
    with pytest.raises(ValueError,match='input device'):rt.col2im(col,shape,out=wrong)
    with pytest.raises(ValueError):rt.col2im(col,shape,stride=0)
    with pytest.raises(ValueError):rt.col2im(col,shape,pad=-1)


def test_legacy_col2im_abi_fallback(cp,monkeypatch):
    lib=rt.load_library()
    class Legacy:
        hy_col2im=lib.hy_col2im
        hy_cuda_error=lib.hy_cuda_error
    monkeypatch.setattr(rt,'load_library',lambda:Legacy())
    col=cp.ones((1,1,1,1,2,3),cp.float32)
    np.testing.assert_array_equal(cp.asnumpy(rt.col2im(col,(1,1,2,3))),np.ones((1,1,2,3),np.float32))


def test_fast_abi_null_params(cp):
    assert rt.load_library().hy_col2im_fast(None,None,None,cp.cuda.get_current_stream().ptr)!=0
