import numpy as np
import pytest
import hytorch
from hytorch import Variable, Parameter, no_grad
from hytorch import functions_conv as FC
from hytorch.backends.cuda import runtime as rt
from hytorch.optimizers import SGD, FreezeParam, WeightDecay, ClipGrad
from test_col2im_cuda import cp

class Target:
    def __init__(self,params):self.items=params
    def params(self):return iter(self.items)
def make_params(xp,n):
    rng=np.random.default_rng(91)
    return [make_param(xp,rng.normal(size=(i*13+1,)).astype(np.float32),rng.normal(size=(i*13+1,)).astype(np.float32)) for i in range(n)]
def make_param(xp,p,g):
    q=Parameter(xp.asarray(p));q.grad=Variable(xp.asarray(g));return q

@pytest.mark.parametrize('n',[0,1,16,32,33,65])
def test_sgd_multi_counts(cp,n):
    p=make_params(cp,n);a=[q.data.copy() for q in p];opt=SGD(.001,backend='native').setup(Target(p))
    rt.dispatch_stats(True)
    stream=cp.cuda.Stream(non_blocking=True);cp.cuda.get_current_stream().synchronize()
    with stream:
        for lr in [.001,.25,-.125,np.float32(.05)]:
            opt.lr=lr
            for q in p:q.grad=Variable(q.grad.data*.99)
            for x,q in zip(a,p):x-=lr*q.grad.data
            opt.update()
        stream.synchronize()
    for x,q in zip(a,p):np.testing.assert_array_equal(cp.asnumpy(x).view(np.uint32),cp.asnumpy(q.data).view(np.uint32))
    assert rt.dispatch_stats().get('sgd_multi',0)==0  # rejected candidate falls back

@pytest.mark.parametrize('kind',['empty','noncontiguous','fp64','explicit_lr64','duplicate','overlap','selfgrad','crossgrad'])
def test_sgd_fallbacks(cp,kind):
    p=make_params(cp,2);lr=.1
    if kind=='empty':p=[make_param(cp,np.zeros(0,np.float32),np.zeros(0,np.float32))]
    if kind=='noncontiguous':
        p[0].data=cp.ones(2,cp.float32)[::2];p[0].grad.data=cp.ones(2,cp.float32)[::2]
        p[1].data=cp.ones(28,cp.float32)[::2]
    if kind=='fp64':p[0].data=p[0].data.astype(cp.float64);p[0].grad.data=p[0].grad.data.astype(cp.float64)
    if kind=='explicit_lr64':lr=np.float64(.1)
    if kind=='duplicate':p=[p[0],p[0]]
    if kind in ['overlap','selfgrad','crossgrad']:
        store=cp.arange(6,dtype=cp.float32)
        p=[make_param(cp,np.ones(3,np.float32),np.ones(3,np.float32)) for _ in range(2)]
        p[0].data=store[:3];p[1].data=store[2:5] if kind=='overlap' else store[3:6]
        if kind=='selfgrad':p[0].grad.data=p[0].data
        if kind=='crossgrad':p[0].grad.data=p[1].data
    # Preserve aliasing in reference via a first sequential update and restoring all backing arrays.
    arrays=[v for q in p for v in [q.data,q.grad.data]];initial=[v.copy() for v in arrays]
    for q in p:q.data-=lr*q.grad.data
    expected=[q.data.copy() for q in p]
    for v,x in zip(arrays,initial):v[...] = x
    rt.dispatch_stats(True);SGD(lr,backend='native').setup(Target(p)).update()
    for q,x in zip(p,expected):np.testing.assert_array_equal(cp.asnumpy(q.data),cp.asnumpy(x))
    assert rt.dispatch_stats().get('sgd_multi',0)==0

@pytest.mark.parametrize('lr',[.001,1.,-0.,np.float32(.5)])
def test_sgd_rounding_special(cp,lr):
    v=np.array([0.,-0.,1e-45,-1e-45,1e-38,-1e-38,1e30,-1e30,1.,-1.,np.inf,-np.inf,np.nan],np.float32)
    p=make_param(cp,v,v[::-1].copy());ref=p.data.copy();ref-=lr*p.grad.data
    assert (lr*p.grad.data).dtype==cp.float32
    SGD(lr,backend='native').setup(Target([p])).update()
    a,b=cp.asnumpy(ref),cp.asnumpy(p.data);finite=np.isfinite(a)
    np.testing.assert_array_equal(a[finite].view(np.uint32),b[finite].view(np.uint32))
    np.testing.assert_array_equal(np.isnan(a),np.isnan(b));np.testing.assert_array_equal(np.isposinf(a),np.isposinf(b));np.testing.assert_array_equal(np.isneginf(a),np.isneginf(b))

def test_sgd_hooks_subclass_cpu():
    p=make_params(np,3);seen=[]
    for backend in ['legacy','native']:
        q=make_params(np,3);o=SGD(.1,backend).setup(Target(q))
        o.add_hook(lambda ps:seen.append(len(ps)));o.add_hook(FreezeParam(q[0]));o.add_hook(lambda ps:seen.append((len(ps),ps[0].grad is None)));o.update()
    assert seen==[3,(3,True),3,(3,True)]
    class Custom(SGD):
        def update_one(self,p):p.data+=7
    p=make_params(np,1);old=p[0].data.copy();Custom(backend='native').setup(Target(p)).update();np.testing.assert_array_equal(p[0].data,old+7)

def test_sgd_hooks_gpu(cp):
    a=make_params(cp,16);b=make_params(cp,16)
    for items,mode in [(a,'legacy'),(b,'native')]:
        o=SGD(.01,mode).setup(Target(items));o.add_hook(WeightDecay(.1));o.add_hook(ClipGrad(1.));o.add_hook(FreezeParam(items[0]));o.update()
    for x,y in zip(a,b):np.testing.assert_array_equal(cp.asnumpy(x.data),cp.asnumpy(y.data))

@pytest.mark.parametrize('layout',['contiguous','negative','slice'])
@pytest.mark.parametrize('shape,kernel,stride,pad',[((2,3,9,13),(3,5),(2,3),(1,2)),((1,2,7,8),(1,3),(1,2),(0,1)),((3,1,8,11),(2,2),(3,1),(1,0))])
def test_dweight_layouts(cp,shape,kernel,stride,pad,layout):
    rng=np.random.default_rng(777);x=cp.asarray(rng.normal(size=shape).astype(np.float32));kh,kw=kernel;oh=(shape[2]+2*pad[0]-kh)//stride[0]+1;ow=(shape[3]+2*pad[1]-kw)//stride[1]+1;g=cp.asarray(rng.normal(size=(shape[0],4,oh,ow)).astype(np.float32))
    if layout=='negative':x=x[...,::-1];g=g[...,::-1]
    if layout=='slice':
        xx=cp.empty((*x.shape[:-1],x.shape[-1]*2),cp.float32);xx[...,::2]=x;x=xx[...,::2]
        gg=cp.empty((*g.shape[:-1],g.shape[-1]*2),cp.float32);gg[...,::2]=g;g=gg[...,::2]
    cp.cuda.get_current_stream().synchronize();stream=cp.cuda.Stream(non_blocking=True)
    with stream:b=rt.packed_dweight(x,g,kernel,stride,pad);stream.synchronize()
    col=FC.im2col_array(cp.asnumpy(x).astype(np.float64),kernel,stride,pad,True);ref=cp.asnumpy(g).astype(np.float64).transpose(1,0,2,3).reshape(4,-1)@col
    np.testing.assert_allclose(cp.asnumpy(b).reshape(4,-1),ref,atol=2e-5,rtol=2e-4)

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_dweight_second_derivative(device,request):
    import torch
    xp=np if device=='cpu' else request.getfixturevalue('cp')
    rng=np.random.default_rng(11);a=rng.normal(0,.1,(2,2,5,7)).astype(np.float32);g=rng.normal(0,.1,(2,3,3,4)).astype(np.float32);v=rng.normal(size=(3,2,3,2)).astype(np.float32)
    for mode in ['cupy','packed']:
        x=Variable(xp.asarray(a));gy=Variable(xp.asarray(g));w=Variable(xp.zeros(v.shape,np.float32))
        with rt.backend(dweight=mode):
            y=FC.conv2d(x,w,stride=(1,2),pad=(0,1));dw=FC.Conv2DGradW(y.creator)(x,gy);(dw*Variable(xp.asarray(v))).sum().backward()
        tx=torch.tensor(a,requires_grad=True);tg=torch.tensor(g,requires_grad=True);tw=torch.tensor(v);ty=torch.nn.functional.conv2d(tx,tw,stride=(1,2),padding=(0,1));(ty*tg).sum().backward()
        np.testing.assert_allclose(hytorch.cuda.as_numpy(x.grad.data),tx.grad.numpy(),atol=2e-6,rtol=2e-5)
        np.testing.assert_allclose(hytorch.cuda.as_numpy(gy.grad.data),tg.grad.numpy(),atol=2e-6,rtol=2e-5)

@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_dweight_mutation_repeated_fallback(cp,dtype):
    x=Variable(cp.ones((1,2,5,5),dtype));w=Variable(cp.ones((3,2,3,3),dtype));rt.dispatch_stats(True)
    with rt.backend(dweight='packed'):
        y=FC.conv2d(x,w);x.data*=2
        y.sum().backward();first=w.grad.data.copy();x.cleargrad();w.cleargrad();y.cleargrad();y.sum().backward()
        np.testing.assert_array_equal(cp.asnumpy(w.grad.data),cp.asnumpy(first))
        np.testing.assert_array_equal(cp.asnumpy(first),np.full(w.shape,18,dtype))
        with no_grad():assert FC.conv2d(x,w).creator is None
    if dtype==np.float64:assert rt.dispatch_stats().get('dweight_packed',0)==0

def test_dweight_mixed_dtype(cp):
    x=cp.ones((1,1,3,3),cp.float32);g=cp.ones((1,1,1,1),cp.float64)
    with pytest.raises(NotImplementedError):rt.packed_dweight(x,g,(3,3))


def test_dweight_geometry_fallback(cp):
    x=cp.ones((1,1,3,3),cp.float32);g=cp.ones((1,1,1,1),cp.float32)
    with pytest.raises(NotImplementedError,match='int32'):
        rt.packed_dweight(x,g,(1,1),stride=(2**32,2**32))
    with pytest.raises(NotImplementedError,match='int32'):
        rt.packed_dweight(x,g,(1,1),stride=(3.5,3.5))
