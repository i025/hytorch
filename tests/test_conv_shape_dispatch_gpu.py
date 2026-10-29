"""Cross-shape/cross-CNN evidence for the independently validated pointwise auto."""
import csv
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

from hytorch import Variable
from hytorch import cuda
import hytorch.functions as F
import hytorch.functions_conv as FC
from hytorch.backends.cuda import backend,dispatch_stats
from hytorch.backends.cuda.runtime import load_library,select_conv_dinput
from hytorch.optimizers import SGD
from benchmarks.bench_cuda_training import SpatialNet,initialize,parameters,torch_model

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('shape_cases',ROOT/'docs/experiments/conv-shape-dispatch/scripts/cases.py')
shapes=importlib.util.module_from_spec(spec);spec.loader.exec_module(shapes)


@pytest.fixture(scope='module')
def device():
    if os.environ.get('HYTORCH_REQUIRE_CUDA')!='1':
        pytest.skip('GPU NOT_RUN: explicitly allocate one idle CUDA device')
    import cupy as cp
    import torch
    cp.cuda.Device(0).use();torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    load_library()
    return cp,torch


def compare(name,actual,reference,atol=1e-5,rtol=1e-4):
    a,b=cuda.as_numpy(actual),cuda.as_numpy(reference)
    delta=a.astype(np.float64)-b;norm=np.linalg.norm(b.astype(np.float64))
    row=dict(case=name,max_abs=float(np.abs(delta).max()),
        relative_l2=float(np.linalg.norm(delta)/norm) if norm else None,atol=atol,rtol=rtol,
        finite=bool(np.isfinite(a).all() and np.isfinite(b).all()),
        mismatches=int((np.abs(delta)>atol+rtol*np.abs(b)).sum()))
    destination=os.environ.get('HYTORCH_SHAPE_TEST_OUTPUT')
    if destination:
        out=Path(destination);out.mkdir(parents=True,exist_ok=True)
        np.savez(out/(name+'.npz'),actual=a,reference=b)
        path=out/'comparisons.csv';exists=path.exists()
        with path.open('a',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(row))
            if not exists:writer.writeheader()
            writer.writerow(row)
    assert row['finite']
    np.testing.assert_allclose(a,b,atol=atol,rtol=rtol,err_msg=name)


@pytest.mark.parametrize('case',shapes.cases('held-out'),ids=lambda r:r['case'])
@pytest.mark.parametrize('nondefault',[False,True])
def test_held_out_full_gradients(device,case,nondefault,request):
    full_gradients(device,case,nondefault,request,('original','fused'))


@pytest.mark.parametrize('case',shapes.cases('validation-final'),ids=lambda r:r['case'])
@pytest.mark.parametrize('nondefault',[False,True])
def test_auto_full_gradients(device,case,nondefault,request):
    full_gradients(device,case,nondefault,request,('original','auto'))
    selected=select_conv_dinput(case['shape'],case['weight_shape'],case['stride'],case['pad'])[0]
    assert dispatch_stats().get('conv_dinput_auto_'+selected,0)==1
    assert dispatch_stats().get('conv_dinput_fused',0)==int(selected=='fused')


def full_gradients(device,case,nondefault,request,modes):
    cp,torch=device
    shape,wshape=case['shape'],case['weight_shape'];s,pad=case['stride'],case['pad']
    n,ci,h,w=shape;co,_,kh,kw=wshape
    rng=np.random.default_rng(42)
    a=rng.normal(size=shape).astype(np.float32)*.1
    weight=rng.normal(size=wshape).astype(np.float32)*.1
    bias=rng.normal(size=co).astype(np.float32)*.1
    dy=rng.normal(size=(n,co,(h+2*pad-kh)//s+1,(w+2*pad-kw)//s+1)).astype(np.float32)*.1
    stream=cp.cuda.Stream(non_blocking=True) if nondefault else cp.cuda.Stream.null
    results=[]
    for mode in modes:
        dispatch_stats(reset=True)
        with stream,backend(conv_backward=mode):
            x,ww,bb=[Variable(cp.asarray(v)) for v in (a,weight,bias)]
            gy=cp.asarray(dy)+cp.float32(.03125)
            y=FC.conv2d(x,ww,bb,stride=s,pad=pad)
            F.sum(y*gy).backward()
            consumed=x.grad.data+cp.float32(.015625)
            results.append((y.data,consumed,ww.grad.data,bb.grad.data))
    stream.synchronize()
    tx,tw,tb=[torch.tensor(v,device='cuda',requires_grad=True) for v in (a,weight,bias)]
    ty=torch.nn.functional.conv2d(tx,tw,tb,stride=s,padding=pad)
    (ty*torch.tensor(dy+np.float32(.03125),device='cuda')).sum().backward()
    expected=(ty.detach(),tx.grad+float(.015625),tw.grad,tb.grad)
    key=request.node.name
    for name,actual,old,target in zip(('forward','dInput','dWeight','dBias'),results[1],results[0],expected):
        compare(key+'_'+name+'_original',actual,old)
        compare(key+'_'+name+'_torch',actual,target.detach().cpu().numpy())


@pytest.mark.parametrize('kind',['float64','noncontiguous'])
def test_auto_uncovered_dtype_layout(device,kind,request):
    cp,_=device
    rng=np.random.default_rng(42)
    dtype=np.float64 if kind=='float64' else np.float32
    a=cp.asarray(rng.normal(size=(2,3,5,7)).astype(dtype)*.1)
    w=cp.asarray(rng.normal(size=(16,3,1,1)).astype(dtype)*.1)
    if kind=='noncontiguous':a=a[:,:,:,::-1]
    values=[]
    for mode in ('original','auto'):
        dispatch_stats(reset=True);x,ww=Variable(a),Variable(w)
        with backend(conv_backward=mode):F.sum(FC.conv2d(x,ww)**2).backward()
        values.append((x.grad.data,ww.grad.data))
    assert dispatch_stats().get('conv_dinput_auto_original',0)==1
    assert dispatch_stats().get('conv_dinput_auto_reason_dtype_layout',0)==1
    assert dispatch_stats().get('conv_dinput_fused',0)==0
    for index,(a,b) in enumerate(zip(*values)):compare(request.node.name+'_'+str(index),a,b,0.,0.)


@pytest.mark.parametrize('pruned',[False,True])
def test_auto_higher_order(device,pruned,request):
    cp,torch=device
    rng=np.random.default_rng(42)
    a=rng.normal(size=(2,3,5,7)).astype(np.float32)*.1
    w=rng.normal(size=(16,3,1,1)).astype(np.float32)*.1
    x,weight=Variable(cp.asarray(a),requires_grad=not pruned),Variable(cp.asarray(w))
    tx=torch.tensor(a,device='cuda',requires_grad=not pruned)
    tw=torch.tensor(w,device='cuda',requires_grad=True)
    dispatch_stats(reset=True)
    with backend(conv_backward='auto'):
        y=FC.conv2d(x,weight)
        F.sum(y*y).backward(create_graph=True)
        assert dispatch_stats().get('conv_dinput_auto_reason_higher_order',0)==int(not pruned)
        assert dispatch_stats().get('conv_dinput_fused',0)==0
        gw=weight.grad;weight.cleargrad();x.cleargrad()
        F.sum(gw*gw).backward()
    ty=torch.nn.functional.conv2d(tx,tw)
    tgw,=torch.autograd.grad((ty*ty).sum(),tw,create_graph=True)
    (tgw*tgw).sum().backward()
    compare(request.node.name+'_second_weight',weight.grad.data,tw.grad.cpu().numpy())
    if not pruned:compare(request.node.name+'_mixed_input',x.grad.data,tx.grad.cpu().numpy())


@pytest.mark.parametrize('pruned',[False,True])
@pytest.mark.parametrize('fused_sgd',[False,True])
def test_spatial_multistep(device,pruned,fused_sgd,request):
    cp,torch=device
    models=[SpatialNet(),SpatialNet(),SpatialNet()]
    states=[initialize(m,(3,16,16),42) for m in models]
    for m in models:m.to_gpu()
    reference=torch_model('spatial',states[0]).cuda()
    optimizers=[SGD(.001,fused=fused_sgd).setup(m) for m in models]
    other=torch.optim.SGD(reference.parameters(),lr=.001)
    rng=np.random.default_rng(42)
    for step in range(3):
        a=rng.normal(size=(4,3,16,16)).astype(np.float32)*.1
        t=rng.normal(size=(4,1)).astype(np.float32)*.1
        losses=[]
        for mode,model,optimizer in zip(('original','fused','auto'),models,optimizers):
            dispatch_stats(reset=True)
            with backend(matmul='shared',conv='im2col',conv_backward=mode):
                x=Variable(cp.asarray(a),requires_grad=not pruned)
                model.cleargrads();loss=F.mean_square_error(model(x),cp.asarray(t))
                loss.backward()
                assert all(p.grad.dtype==cp.float32 for p in parameters(model).values())
                optimizer.update()
            assert (x.grad is None)==pruned
            expected=(2 if pruned else 3) if mode=='fused' else 1 if mode=='auto' else 0
            assert dispatch_stats().get('conv_dinput_fused',0)==expected
            losses.append(loss.data)
        other.zero_grad(set_to_none=True)
        tx=torch.tensor(a,device='cuda',requires_grad=not pruned)
        target=torch.nn.functional.mse_loss(reference(tx),torch.tensor(t,device='cuda'))
        target.backward();other.step()
        key=request.node.name+'_'+str(step)
        for mode,loss in zip(('fused','auto'),losses[1:]):
            compare(key+'_'+mode+'_loss_original',loss,losses[0],1e-6,1e-5)
            compare(key+'_'+mode+'_loss_torch',loss,np.array(target.item()),1e-6,1e-5)
        if not pruned:compare(key+'_dInput',x.grad.data,tx.grad.cpu().numpy())
        old=parameters(models[0])
        for mode,model in zip(('fused','auto'),models[1:]):
            for name,p in parameters(model).items():
                layer,attr=name.split('/')
                tp=getattr(getattr(reference,layer),'weight' if attr=='W' else 'bias')
                ww,gg=tp.detach().cpu().numpy(),tp.grad.cpu().numpy()
                if ww.ndim==2:ww,gg=ww.T,gg.T
                label=key+'_'+mode+'_'+name.replace('/','_')
                compare(label+'_weight_original',p.data,old[name].data,1e-6,1e-5)
                compare(label+'_weight_torch',p.data,ww,1e-6,1e-5)
                compare(label+'_grad_original',p.grad.data,old[name].grad.data)
                compare(label+'_grad_torch',p.grad.data,gg)
