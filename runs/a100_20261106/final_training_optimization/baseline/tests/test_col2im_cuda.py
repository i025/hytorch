"""Independent scatter/Fold references, signed strides, fallback and higher gradients."""
import os,itertools
import numpy as np
import pytest
import hytorch
from hytorch import Variable
from hytorch import functions as F
from hytorch import functions_conv as FC
from hytorch.backends.cuda import runtime as rt

CASES=[((2,3,9,11),(k,k),(s,s),(p,p)) for k,s,p in itertools.product([1,3,5],[1,2,3],[0,1,2])]
CASES += [((1,2,8,13),(3,5),(2,3),(1,2)),((3,1,7,10),(1,3),(3,1),(2,0)),((2,2,11,7),(5,1),(1,3),(0,1))]

@pytest.fixture(scope='module')
def cp():
    try:
        import cupy
        cupy.cuda.Device(0).use();cupy.zeros(1)
    except Exception as exc:
        if os.environ.get('HYTORCH_REQUIRE_CUDA')=='1':pytest.fail(str(exc))
        pytest.skip('CUDA unavailable')
    assert hasattr(rt.load_library(),'hy_col2im'),'Rebuild CUDA col2im extension'
    return cupy

def scatter_reference(col,shape,stride,pad):
    n,c,h,w=shape;sh,sw=stride;ph,pw=pad;kh,kw,oh,ow=col.shape[2:]
    out=np.zeros(shape,dtype=col.dtype)
    for ky in range(kh):
        for kx in range(kw):
            for iy in range(oh):
                y=iy*sh+ky-ph
                if not 0<=y<h:continue
                for ix in range(ow):
                    x=ix*sw+kx-pw
                    if 0<=x<w:out[:,:,y,x]+=col[:,:,ky,kx,iy,ix]
    return out

def view_array(cp,host,layout):
    if layout=='contiguous':return cp.asarray(host)
    if layout=='training':return cp.asarray(host.transpose(1,2,3,0,4,5).copy()).transpose(3,0,1,2,4,5)
    if layout=='negative':return cp.asarray(host[...,::-1].copy())[...,::-1]
    if layout=='sliced':
        storage=cp.empty((*host.shape[:-1],host.shape[-1]*2),dtype=host.dtype)
        view=storage[...,::2];view[...] = cp.asarray(host);return view
    if layout=='broadcast':return cp.broadcast_to(cp.asarray(host[:1]),host.shape)
    raise AssertionError(layout)

@pytest.mark.parametrize('shape,kernel,stride,pad',CASES)
@pytest.mark.parametrize('layout',['contiguous','training','negative'])
def test_col2im_shapes_strides_reference(cp,shape,kernel,stride,pad,layout):
    n,c,h,w=shape;kh,kw=kernel;sh,sw=stride;ph,pw=pad;oh=(h+2*ph-kh)//sh+1;ow=(w+2*pw-kw)//sw+1
    rng=np.random.default_rng(123);host=rng.normal(size=(n,c,kh,kw,oh,ow)).astype(np.float32)
    col=view_array(cp,host,layout);before=cp.asnumpy(col);out=cp.full(shape,cp.nan,dtype=cp.float32);cp.cuda.get_current_stream().synchronize()
    stream=cp.cuda.Stream(non_blocking=True)
    with stream:
        for _ in range(3):assert rt.col2im(col,shape,stride,pad,out=out) is out
        baseline=FC._col2im_gpu(col,sh,sw,ph,pw,h,w)
        stream.begin_capture();rt.col2im(col,shape,stride,pad,out=out);g=stream.end_capture()
        assert 'col2im_strided' in g.debug_dot_str(flags=1)
        g.launch();stream.synchronize()
    actual=cp.asnumpy(out);np.testing.assert_array_equal(actual,cp.asnumpy(baseline))
    np.testing.assert_array_equal(actual,scatter_reference(before,shape,stride,pad))
    np.testing.assert_array_equal(cp.asnumpy(col),before)
    assert out.flags.c_contiguous

@pytest.mark.parametrize('layout',['sliced','broadcast'])
def test_zero_and_nonunit_strides(cp,layout):
    shape=(2,2,7,9);host=np.random.default_rng(6).normal(size=(2,2,3,5,4,3)).astype(np.float32)
    col=view_array(cp,host,layout);actual=rt.col2im(col,shape,(2,3),(1,1))
    np.testing.assert_array_equal(cp.asnumpy(actual),scatter_reference(cp.asnumpy(col),shape,(2,3),(1,1)))

@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('matrix',[False,True])
def test_fold_and_higher_gradients(cp,dtype,matrix):
    import torch
    shape=(2,2,7,9);kernel=(3,5);stride=(2,3);pad=(1,1);oh,ow=4,3
    rng=np.random.default_rng(7);host=rng.normal(0,.1,(2,2,3,5,oh,ow)).astype(dtype)
    ar=host.transpose(0,4,5,1,2,3).reshape(2*oh*ow,-1) if matrix else host
    x=Variable(cp.asarray(ar));tx=torch.tensor(host.reshape(2,30,oh*ow),requires_grad=True)
    rt.dispatch_stats(reset=True)
    with rt.backend(col2im='native'):
        y=FC.col2im(x,shape,kernel,stride,pad,to_matrix=matrix)
        # Avoid Pow's unrelated Python scalar promotion when testing second derivatives.
        loss=(y*y).sum();loss.backward(create_graph=True)
        grad=x.grad; x.cleargrad();grad.sum().backward()
    ty=torch.nn.functional.fold(tx,(7,9),kernel_size=kernel,padding=pad,stride=stride)
    tg=torch.autograd.grad((ty*ty).sum(),tx,create_graph=True)[0];tgg=torch.autograd.grad(tg.sum(),tx)[0]
    def reference(t):
        z=t.detach().numpy().reshape(host.shape)
        return z.transpose(0,4,5,1,2,3).reshape(ar.shape) if matrix else z
    np.testing.assert_allclose(cp.asnumpy(y.data),ty.detach().numpy(),rtol=2e-6,atol=2e-7)
    np.testing.assert_allclose(cp.asnumpy(grad.data),reference(tg),rtol=2e-6,atol=2e-7)
    np.testing.assert_allclose(cp.asnumpy(x.grad.data),reference(tgg),rtol=2e-6,atol=2e-7)
    assert y.dtype==dtype and grad.dtype==dtype and x.grad.dtype==dtype
    counts=rt.dispatch_stats()
    assert counts.get('col2im_native',0)>0 if dtype==np.float32 else counts.get('col2im_fallback_cpu_or_dtype',0)>0
    if dtype==np.float64:assert counts.get('col2im_native',0)==0

def test_validation_and_default(cp):
    assert rt.current_config()['col2im']=='cupy'
    with pytest.raises(ValueError):
        with rt.backend(col2im='bad'):pass
    col=cp.ones((1,1,1,1,2,3),dtype=cp.float32)
    with pytest.raises(ValueError,match='overlap'):rt.col2im(col,(1,1,2,3),out=col.reshape(1,1,2,3))
    with pytest.raises(ValueError):rt.col2im(col,(1,1,3,3))
    with pytest.raises(TypeError):rt.col2im(col.astype(cp.float64),(1,1,2,3))
    # Stride > kernel leaves pixels with no contributor; those must be exactly zero.
    c=cp.asarray(np.array([[[[[[2,3],[4,5]]]]]],np.float32))
    y=rt.col2im(c,(1,1,4,4),stride=3)
    ref=np.zeros((1,1,4,4),np.float32);ref[0,0,::3,::3]=[[2,3],[4,5]]
    np.testing.assert_array_equal(cp.asnumpy(y),ref)

def test_fallback_for_old_library(cp,monkeypatch):
    col=cp.ones((1,1,1,1,2,3),cp.float32)
    monkeypatch.setattr(rt,'load_library',lambda:object())
    with rt.backend(col2im='native'):
        y=FC.col2im_array(col,(1,1,2,3),1,1,0,False)
    np.testing.assert_array_equal(cp.asnumpy(y),np.ones((1,1,2,3),np.float32))
