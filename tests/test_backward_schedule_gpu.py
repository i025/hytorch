"""Optional prototype GPU checks, only after existing Sum/ReLU/primary training."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

import hytorch
import hytorch.functions as F
from test_conv_shape_dispatch_gpu import device,compare


@pytest.mark.parametrize('width',[1,32,256])
@pytest.mark.parametrize('nondefault',[False,True])
def test_exact_schedule_order_on_gpu(device,width,nondefault,request):
    cp,_=device
    a=np.random.default_rng(42).normal(0,.1,(4,32)).astype(np.float32)
    stream=cp.cuda.Stream(non_blocking=True) if nondefault else cp.cuda.Stream.null
    values=[]
    with stream:
        for mode in ('list','heap'):
            with hytorch.using_config('backward_schedule',mode):
                x=hytorch.Variable(cp.asarray(a));root=x*x
                branches=[F.tanh(root+np.float32(i/width)) for i in range(width)]
                y=branches[0]
                for value in branches[1:]:y=y+value
                F.sum(y).backward(create_graph=True)
                first=x.grad;x.cleargrad();F.sum(first).backward()
                values.append((first.data,x.grad.data))
    stream.synchronize()
    for i,(new,old) in enumerate(zip(values[1],values[0])):
        compare(request.node.name+str(i),new,old,atol=0,rtol=0)


@pytest.mark.parametrize('model',['donkey','spatial'])
def test_cnn_three_gpu_steps_identical(device,model,request):
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('seven_hour_common',root/'docs/experiments/gpu0-autonomous-7h/scripts/common.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    data=np.load(root/'runs/udacity_real_prepared_01/throughput-inputs.npz')
    values=[]
    for mode in ('list','heap'):
        engine=module.Engine('I',model)
        with engine.context(),hytorch.using_config('backward_schedule',mode):
            values.append([module.result(engine,data['x'][i*4:i*4+4],data['t'][i*4:i*4+4]) for i in range(3)])
    for step,(new,old) in enumerate(zip(values[1],values[0])):
        for key in new:
            compare(request.node.name+str(step)+'_'+key.replace('/','_'),new[key],old[key],atol=0,rtol=0)
