"""CUDA dInput and training parity. Run only on an explicitly allocated GPU."""
import csv
import os
from pathlib import Path

import numpy as np
import pytest

from hytorch import Variable
import hytorch.functions as F
import hytorch.functions_conv as FC
from hytorch.backends.cuda import backend, dispatch_stats
from hytorch.backends.cuda.runtime import conv2d_dinput, load_library
from hytorch.optimizers import SGD


@pytest.fixture(scope='module')
def device():
    if os.environ.get('HYTORCH_REQUIRE_CUDA') != '1':
        pytest.skip('CUDA NOT_RUN: allocate an idle GPU and set HYTORCH_REQUIRE_CUDA=1')
    import cupy as cp
    import torch
    cp.cuda.Device(0).use()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    load_library()
    return cp, torch


def compare(case, actual, expected, atol=1e-5, rtol=1e-4):
    from hytorch.cuda import as_numpy
    a,b = as_numpy(actual),as_numpy(expected)
    delta = a.astype(np.float64)-b.astype(np.float64)
    norm = np.linalg.norm(b.astype(np.float64))
    row = dict(case=case,elements=a.size,max_abs=float(np.abs(delta).max()),
               relative_l2=float(np.linalg.norm(delta)/norm) if norm else float(np.linalg.norm(delta)),
               reference_l2=float(norm),atol=atol,rtol=rtol,
               mismatches=int((np.abs(delta)>atol+rtol*np.abs(b)).sum()),
               finite=bool(np.isfinite(a).all() and np.isfinite(b).all()))
    destination = os.environ.get('HYTORCH_CONV_TEST_OUTPUT')
    if destination:
        out = Path(destination)
        out.mkdir(parents=True,exist_ok=True)
        with (out/(case+'.npz')).open('xb') as f:
            np.savez(f,actual=a,expected=b)
        path = out/'comparisons.csv'
        exists = path.exists()
        with path.open('a',newline='') as f:
            writer = csv.DictWriter(f,fieldnames=list(row))
            if not exists:writer.writeheader()
            writer.writerow(row)
    assert row['finite'], case
    np.testing.assert_allclose(a,b,atol=atol,rtol=rtol,err_msg=case)


CASES = [
    ((1,1,5,7),(1,1,1,1),1,0),
    ((2,3,8,9),(5,3,3,3),1,1),
    ((3,2,9,10),(7,2,2,3),2,1),
    ((1,5,11,13),(9,5,5,3),(2,1),(1,0)),
    ((4,24,31,98),(36,24,5,5),2,0),
    ((4,36,14,47),(48,36,5,5),2,0),
    ((4,48,5,22),(64,48,3,3),1,0),
    ((4,64,3,20),(64,64,3,3),1,0),
]


@pytest.mark.parametrize('shape,wshape,stride,pad',CASES)
@pytest.mark.parametrize('nondefault',[False,True])
def test_independent_dinput(device,shape,wshape,stride,pad,nondefault,request):
    cp,torch = device
    from hytorch.utils import pair
    sh,sw = pair(stride);ph,pw = pair(pad)
    gyshape = (shape[0],wshape[0],(shape[2]+2*ph-wshape[2])//sh+1,(shape[3]+2*pw-wshape[3])//sw+1)
    rng = np.random.default_rng(42)
    w = rng.normal(size=wshape).astype(np.float32)*.1
    gy = rng.normal(size=gyshape).astype(np.float32)*.1
    stream = cp.cuda.Stream(non_blocking=True) if nondefault else cp.cuda.Stream.null
    with stream:
        cw = cp.asarray(w)
        cgy = cp.asarray(gy)+cp.float32(.03125)  # Producer on the tested stream.
        outputs = [conv2d_dinput(cgy,cw,shape,stride,pad)+cp.float32(.015625) for _ in range(3)]
        old = FC.deconv2d(cgy,cw,stride=stride,pad=pad,outsize=shape[2:]).data+cp.float32(.015625)
    stream.synchronize()
    gy = gy+np.float32(.03125)
    expected = torch.nn.grad.conv2d_input(shape,torch.tensor(w,device='cuda'),
                  torch.tensor(gy,device='cuda'),stride=stride,padding=pad).cpu().numpy()+np.float32(.015625)
    for i,actual in enumerate(outputs):
        compare(request.node.name+f'_r{i}_original',actual,old)
        compare(request.node.name+f'_r{i}_torch',actual,expected)
    compare(request.node.name+'_weight_unchanged',cw,w,0.,0.)
    compare(request.node.name+'_gy_unchanged',cgy,gy,0.,0.)


@pytest.mark.parametrize('pruned',[False,True])
def test_conv_parameters_and_higher_order(device,pruned,request):
    cp,torch = device
    rng = np.random.default_rng(42)
    a = rng.normal(size=(2,2,7,8)).astype(np.float32)*.1
    w = rng.normal(size=(3,2,3,3)).astype(np.float32)*.1
    b = rng.normal(size=3).astype(np.float32)*.1
    rows=[]
    for mode in ('original','fused'):
        x,weight,bias = Variable(cp.asarray(a),requires_grad=not pruned),Variable(cp.asarray(w)),Variable(cp.asarray(b))
        with backend(conv_backward=mode):
            y = FC.conv2d(x,weight,bias,stride=2,pad=1)
            F.sum(y*y).backward()
        assert (x.grad is None)==pruned
        rows.append((y.data,weight.grad.data,bias.grad.data))
    for label,actual,old in zip(('y','dWeight','dBias'),rows[1],rows[0]):
        compare(request.node.name+'_'+label,actual,old,0.,0.)
    # A requested higher-order graph must use differentiable original operations.
    x,weight = Variable(cp.asarray(a),requires_grad=not pruned),Variable(cp.asarray(w))
    tx,tw = torch.tensor(a,device='cuda',requires_grad=not pruned),torch.tensor(w,device='cuda',requires_grad=True)
    dispatch_stats(reset=True)
    with backend(conv_backward='fused'):
        y = FC.conv2d(x,weight,stride=2,pad=1)
        F.sum(y*y).backward(create_graph=True)
        assert dispatch_stats().get('conv_dinput_fused',0)==0
        gw=weight.grad
        weight.cleargrad();x.cleargrad()
        F.sum(gw*gw).backward()
    ty=torch.nn.functional.conv2d(tx,tw,stride=2,padding=1)
    tgw,=torch.autograd.grad((ty*ty).sum(),tw,create_graph=True)
    (tgw*tgw).sum().backward()
    compare(request.node.name+'_second_weight',weight.grad.data,tw.grad.cpu().numpy())
    if not pruned:compare(request.node.name+'_mixed',x.grad.data,tx.grad.cpu().numpy())


@pytest.mark.parametrize('fused_sgd',[False,True])
def test_donkeynet_three_steps(device,fused_sgd,request):
    cp,torch = device
    from benchmarks.bench_cuda_training import initialize,parameters,torch_model
    from donkeycar_hy.model import DonkeyNet
    models=[DonkeyNet(dropout_rate=0.) for _ in range(2)]
    states=[initialize(m,(3,66,200),42) for m in models]
    for model in models:model.to_gpu()
    tm=torch_model('donkey',states[0]).cuda()
    opts=[SGD(.001,fused=fused_sgd).setup(m) for m in models]
    topt=torch.optim.SGD(tm.parameters(),lr=.001)
    rng=np.random.default_rng(42)
    for step in range(3):
        a=rng.normal(size=(4,3,66,200)).astype(np.float32)
        t=rng.normal(size=(4,1)).astype(np.float32)
        losses=[]
        for mode,model,opt in zip(('original','fused'),models,opts):
            dispatch_stats(reset=True)
            with backend(matmul='shared',conv='im2col',conv_backward=mode):
                x=Variable(cp.asarray(a),requires_grad=False)
                model.cleargrads()
                loss=F.mean_square_error(model(x),cp.asarray(t))
                loss.backward();opt.update()
            assert x.grad is None
            assert dispatch_stats().get('conv_dinput_fused',0)==(4 if mode=='fused' else 0)
            losses.append(loss.data)
        topt.zero_grad(set_to_none=True)
        tl=torch.nn.functional.mse_loss(tm(torch.tensor(a,device='cuda')),torch.tensor(t,device='cuda'))
        tl.backward();topt.step()
        key=request.node.name+f'_step{step}'
        compare(key+'_loss_original',losses[1],losses[0],1e-6,1e-5)
        compare(key+'_loss_torch',losses[1],np.array(tl.item()),1e-6,1e-5)
        old=parameters(models[0])
        for name,p in parameters(models[1]).items():
            layer,attr=name.split('/')
            tp=getattr(getattr(tm,layer),'weight' if attr=='W' else 'bias')
            tw,tg=tp.detach().cpu().numpy(),tp.grad.cpu().numpy()
            if tw.ndim==2:tw,tg=tw.T,tg.T
            k=key+'_'+name.replace('/','_')
            compare(k+'_weight_original',p.data,old[name].data,1e-6,1e-5)
            compare(k+'_grad_original',p.grad.data,old[name].grad.data)
            compare(k+'_weight_torch',p.data,tw,1e-6,1e-5)
            compare(k+'_grad_torch',p.grad.data,tg)


def test_unsupported_layout_and_dtype(device):
    cp,_=device
    gy=cp.ones((1,2,3,3),dtype=cp.float32)
    w=cp.ones((2,3,3,3),dtype=cp.float32)
    with pytest.raises(ValueError,match='contiguous'):
        conv2d_dinput(gy[:,:,:,::-1],w,(1,3,5,5))
    with pytest.raises(TypeError,match='float32'):
        conv2d_dinput(gy.astype(cp.float64),w,(1,3,5,5))
    with pytest.raises(ValueError,match='stride'):
        conv2d_dinput(gy,w,(1,3,9,9),stride=3)
