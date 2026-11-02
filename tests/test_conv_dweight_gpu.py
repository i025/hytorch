"""Real CUDA acceptance; explicit allocation required, never CPU fallback."""
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

from hytorch import Variable
import hytorch.functions as F
import hytorch.functions_conv as FC
from hytorch.backends.cuda import backend,dispatch_stats
from hytorch.backends.cuda.runtime import conv2d_dweight
from hytorch.optimizers import SGD
from benchmarks.bench_cuda_training import initialize,parameters,torch_model
from test_conv_shape_dispatch_gpu import device,compare

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('dweight_cases',ROOT/'docs/experiments/conv-dweight-fusion/scripts/cases.py')
shapes=importlib.util.module_from_spec(spec);spec.loader.exec_module(shapes)


@pytest.mark.parametrize('case',shapes.cases(),ids=lambda r:r['case'])
@pytest.mark.parametrize('nondefault',[False,True])
def test_dweight_gradients(device,case,nondefault,request):
    # Save exact producer inputs before execution so type/launch errors also
    # leave reproducible inputs, not only the comparisons that finished.
    from hytorch.utils import pair
    n,ci,h,w=case['shape'];co,_,kh,kw=case['weight_shape']
    sh,sw=pair(case['stride']);ph,pw=pair(case['pad'])
    rng=np.random.default_rng(42)
    a=rng.normal(size=case['shape']).astype(np.float32)*.1
    weight=rng.normal(size=case['weight_shape']).astype(np.float32)*.1
    bias=rng.normal(size=co).astype(np.float32)*.1
    dy=rng.normal(size=(n,co,(h+2*ph-kh)//sh+1,(w+2*pw-kw)//sw+1)).astype(np.float32)*.1
    if os.environ.get('HYTORCH_SHAPE_TEST_OUTPUT'):
        out=Path(os.environ['HYTORCH_SHAPE_TEST_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
        np.savez(out/(request.node.name+'-inputs.npz'),input=a,weight=weight,bias=bias,dy=dy)
    cp,torch=device
    stream=cp.cuda.Stream(non_blocking=True) if nondefault else cp.cuda.Stream.null
    results=[]
    for mode in ('original','fused'):
        dispatch_stats(reset=True)
        with stream,backend(conv_dweight=mode):
            x,ww,bb=[Variable(cp.asarray(v)) for v in (a,weight,bias)]
            gy=cp.asarray(dy)+cp.float32(.03125)
            y=FC.conv2d(x,ww,bb,stride=(sh,sw),pad=(ph,pw));F.sum(y*gy).backward()
            results.append((y.data,x.grad.data+cp.float32(.015625),ww.grad.data,bb.grad.data))
    stream.synchronize()
    assert dispatch_stats().get('conv_dweight_fused',0)==1
    tx,tw,tb=[torch.tensor(v,device='cuda',requires_grad=True) for v in (a,weight,bias)]
    # Correctness reference uses native FP32 uniformly. The separately retained
    # c23/v19 diagnosis establishes cuDNN forward errors against CPU FP64;
    # training/performance references continue to use cuDNN.
    with torch.backends.cudnn.flags(enabled=False):
        ty=torch.nn.functional.conv2d(tx,tw,tb,stride=(sh,sw),padding=(ph,pw))
        (ty*torch.tensor(dy+np.float32(.03125),device='cuda')).sum().backward()
    expected=(ty.detach(),tx.grad+float(.015625),tw.grad,tb.grad)
    for name,actual,original,target in zip(('forward','dInput','dWeight','dBias'),results[1],results[0],expected):
        compare(request.node.name+'_'+name+'_original',actual,original)
        compare(request.node.name+'_'+name+'_torch_native',actual,target.detach().cpu().numpy())
    with stream:
        cx,cgy=cp.asarray(a),cp.asarray(dy)+cp.float32(.03125)
        for repeat in range(3):
            new=conv2d_dweight(cx,cgy,(kh,kw),case['stride'],case['pad'])
            consumed=new+cp.float32(.015625)
            stream.synchronize()
            compare(request.node.name+'_repeat_'+str(repeat),consumed,results[0][2]+cp.float32(.015625))


@pytest.mark.parametrize('model_name',['donkey','spatial'])
@pytest.mark.parametrize('pruned',[False,True])
@pytest.mark.parametrize('fused_sgd',[False,True])
def test_dweight_models(device,model_name,pruned,fused_sgd,request):
    from benchmarks.bench_cuda_training import SpatialNet
    from donkeycar_hy.model import DonkeyNet
    cp,torch=device;factory=SpatialNet if model_name=='spatial' else lambda:DonkeyNet(dropout_rate=0.)
    shape=(3,16,16) if model_name=='spatial' else (3,66,200)
    models=[factory() for _ in range(2)];state=[initialize(m,shape,42) for m in models]
    for m in models:m.to_gpu()
    reference=torch_model(model_name,state[0]).cuda()
    optimizers=[SGD(.001,fused=fused_sgd).setup(m) for m in models]
    other=torch.optim.SGD(reference.parameters(),lr=.001);rng=np.random.default_rng(42)
    for step in range(3):
        a=rng.normal(size=(4,*shape)).astype(np.float32)*.1;t=rng.normal(size=(4,1)).astype(np.float32)*.1
        results=[]
        for mode,m,opt in zip(('original','fused'),models,optimizers):
            x=Variable(cp.asarray(a),requires_grad=not pruned);m.cleargrads();dispatch_stats(reset=True)
            with backend(matmul='shared',conv='im2col',conv_backward='fused',conv_dweight=mode):
                loss=F.mean_square_error(m(x),cp.asarray(t));loss.backward();opt.update()
            assert (x.grad is None)==pruned
            if mode=='fused':assert dispatch_stats().get('conv_dweight_fused',0)==(3 if model_name=='spatial' else 5)
            results.append(loss.data)
        other.zero_grad(set_to_none=True);tx=torch.tensor(a,device='cuda',requires_grad=not pruned)
        target=torch.nn.functional.mse_loss(reference(tx),torch.tensor(t,device='cuda'));target.backward();other.step()
        key=request.node.name+'_'+str(step)
        compare(key+'_loss_original',results[-1],results[0],1e-6,1e-5)
        compare(key+'_loss_torch',results[-1],np.array(target.item()),1e-6,1e-5)
        if not pruned:compare(key+'_input',x.grad.data,tx.grad.cpu().numpy())
        old=parameters(models[0])
        for name,p in parameters(models[-1]).items():
            layer,attr=name.split('/');tp=getattr(getattr(reference,layer),'weight' if attr=='W' else 'bias')
            ww,gg=tp.detach().cpu().numpy(),tp.grad.cpu().numpy()
            if ww.ndim==2:ww,gg=ww.T,gg.T
            label=key+'_'+name.replace('/','_')
            compare(label+'_weight_old',p.data,old[name].data,1e-6,1e-5)
            compare(label+'_weight_torch',p.data,ww,1e-6,1e-5)
            compare(label+'_gradient',p.grad.data,gg)


@pytest.mark.parametrize('pruned',[False,True])
def test_dweight_higher_order(device,pruned,request):
    cp,torch=device;rng=np.random.default_rng(42)
    a=rng.normal(size=(2,3,5,7)).astype(np.float32)*.1
    w=rng.normal(size=(5,3,3,3)).astype(np.float32)*.1
    x,weight=Variable(cp.asarray(a),requires_grad=not pruned),Variable(cp.asarray(w))
    tx,tw=torch.tensor(a,device='cuda',requires_grad=not pruned),torch.tensor(w,device='cuda',requires_grad=True)
    dispatch_stats(reset=True)
    with backend(conv_dweight='fused'):
        y=FC.conv2d(x,weight,pad=1);F.sum(y*y).backward(create_graph=True)
        assert dispatch_stats().get('conv_dweight_fused',0)==0
        gw=weight.grad;weight.cleargrad();x.cleargrad();F.sum(gw*gw).backward()
    ty=torch.nn.functional.conv2d(tx,tw,padding=1)
    tgw,=torch.autograd.grad((ty*ty).sum(),tw,create_graph=True);(tgw*tgw).sum().backward()
    compare(request.node.name+'_weight',weight.grad.data,tw.grad.cpu().numpy())
    if not pruned:compare(request.node.name+'_input',x.grad.data,tx.grad.cpu().numpy())


@pytest.mark.parametrize('kind',['dtype','layout','geometry'])
def test_dweight_contract(device,kind):
    cp,_=device;x=cp.zeros((1,3,7,9),cp.float32);gy=cp.zeros((1,5,5,7),cp.float32)
    if kind=='dtype':
        with pytest.raises(TypeError,match='float32'):conv2d_dweight(x.astype(cp.float64),gy,3)
    elif kind=='layout':
        with pytest.raises(ValueError,match='contiguous'):conv2d_dweight(x[:,:,:,::-1],gy,3)
    else:
        with pytest.raises(ValueError,match='geometry'):conv2d_dweight(x,cp.ascontiguousarray(gy[:,:,:-1,:]),3)
