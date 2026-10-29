"""Explicit input-gradient demand, first/higher derivatives and training parity.

Run once on CPU and once with HYTORCH_PRUNING_DEVICE=cuda on an allocated GPU.
"""
import csv
import os
from pathlib import Path

import numpy as np
import pytest

from hytorch import Variable
from hytorch.core import Function
import hytorch.functions as F
import hytorch.functions_conv as FC
from hytorch.optimizers import SGD
from hytorch.backends.cuda import backend


@pytest.fixture(scope="module")
def device():
    import torch
    name = os.environ.get("HYTORCH_PRUNING_DEVICE", "cpu")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if name == "cuda":
        import cupy as xp
        xp.cuda.Device(0).use()
    else:
        assert name == "cpu"
        xp = np
    return xp, name, torch


def compare(case, actual, expected, atol=1e-6, rtol=1e-5):
    from hytorch.cuda import as_numpy
    a, b = as_numpy(actual), as_numpy(expected)
    error = np.abs(a.astype(np.float64)-b.astype(np.float64))
    nz = b != 0
    rel = error[nz]/np.abs(b[nz].astype(np.float64))
    destination = os.environ.get("HYTORCH_PRUNING_TEST_OUTPUT")
    if destination:
        out = Path(destination)
        out.mkdir(parents=True, exist_ok=True)
        with (out/(case+".npz")).open("xb") as f:
            np.savez(f, actual=a, expected=b)
        path = out/"comparisons.csv"
        exists = path.exists()
        with path.open("a", newline="") as f:
            row = dict(case=case, elements=a.size, max_abs=float(error.max()),
                       max_rel_nonzero=float(rel.max()) if rel.size else 0.,
                       zeros=int((~nz).sum()), atol=atol, rtol=rtol,
                       mismatches=int((error > atol+rtol*np.abs(b)).sum()),
                       finite=bool(np.isfinite(a).all() and np.isfinite(b).all()))
            writer = csv.DictWriter(f, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)
    np.testing.assert_allclose(a, b, atol=atol, rtol=rtol, err_msg=case)


def test_none_gradient_does_not_schedule_its_creator(device):
    xp, _, _ = device

    class Stop(Function):
        def forward(self, x, w):
            return x+w

        def backward(self, gy):
            return None, gy

    x, w = Variable(xp.ones(3, xp.float32)), Variable(xp.ones(3, xp.float32))
    intermediate = x*2
    F.sum(Stop()(intermediate, w)).backward()
    assert x.grad is None and intermediate.grad is None
    compare("none_other_input", w.grad.data, np.ones(3, np.float32), 0., 0.)


def test_leaf_and_general_higher_order(device):
    xp, _, _ = device
    leaf = Variable(xp.array(2., xp.float32))
    leaf.backward()
    compare("leaf_requested", leaf.grad.data, np.array(1., np.float32), 0., 0.)
    constant = Variable(xp.array(2., xp.float32), requires_grad=False)
    with pytest.raises(RuntimeError, match="requires_grad"):
        constant.backward()
    x = Variable(xp.array(2., xp.float32))
    (x**3).backward(create_graph=True)
    gx = x.grad
    x.cleargrad()
    gx.backward()
    compare("cubic_second", x.grad.data, np.array(12., np.float32), 0., 0.)
    y = constant*3
    assert y.requires_grad  # No automatic flag inference: explicitly documented scope.
    y.backward()
    assert constant.grad is None


@pytest.mark.parametrize("demand", [None, True, False], ids=["default", "requested", "pruned"])
@pytest.mark.parametrize("bias", [False, True])
def test_conv_input_weight_bias(device, monkeypatch, demand, bias, request):
    xp, name, torch = device
    rng = np.random.default_rng(42)
    x0 = rng.normal(size=(2, 2, 7, 8)).astype(np.float32)
    w0 = rng.normal(size=(3, 2, 3, 3)).astype(np.float32)*.1
    b0 = rng.normal(size=3).astype(np.float32)*.1
    x = Variable(xp.asarray(x0)) if demand is None else Variable(xp.asarray(x0), requires_grad=demand)
    w, b = Variable(xp.asarray(w0)), Variable(xp.asarray(b0)) if bias else None
    tx = torch.tensor(x0, device=name, requires_grad=demand is not False)
    tw = torch.tensor(w0, device=name, requires_grad=True)
    tb = torch.tensor(b0, device=name, requires_grad=True) if bias else None
    calls = []
    original = FC.deconv2d

    def counted(*args, **kwargs):
        calls.append(kwargs.get("outsize"))
        return original(*args, **kwargs)

    monkeypatch.setattr(FC, "deconv2d", counted)
    with backend(matmul="shared" if name == "cuda" else "cupy", conv="im2col" if name == "cuda" else "cupy"):
        y = FC.conv2d(x, w, b, stride=2, pad=1)
        ty = torch.nn.functional.conv2d(tx, tw, tb, stride=2, padding=1)
        F.sum(y*y).backward()
        (ty*ty).sum().backward()
    key = request.node.name
    compare(key+"_forward", y.data, ty.detach().cpu().numpy())
    compare(key+"_dWeight", w.grad.data, tw.grad.cpu().numpy(), 1e-5, 1e-4)
    if bias:
        compare(key+"_dBias", b.grad.data, tb.grad.cpu().numpy(), 1e-5, 1e-4)
    if demand is False:
        assert x.grad is None and calls == []
    else:
        assert calls == [(7, 8)]
        compare(key+"_dInput", x.grad.data, tx.grad.cpu().numpy(), 1e-5, 1e-4)


@pytest.mark.parametrize("pruned", [False, True])
def test_two_layer_propagation(device, monkeypatch, pruned, request):
    xp, name, torch = device
    rng = np.random.default_rng(42)
    arrays = [rng.normal(size=s).astype(np.float32)*.1 for s in [(2,2,8,8),(3,2,3,3),(4,3,3,3)]]
    x, w1, w2 = [Variable(xp.asarray(a)) for a in arrays]
    x.requires_grad = not pruned
    tx, tw1, tw2 = [torch.tensor(a, device=name, requires_grad=True) for a in arrays]
    tx.requires_grad_(not pruned)
    calls = []
    original = FC.deconv2d

    def counted(*args, **kwargs):
        calls.append(kwargs["outsize"])
        return original(*args, **kwargs)

    monkeypatch.setattr(FC, "deconv2d", counted)
    h = FC.conv2d(x,w1)
    y = FC.conv2d(h,w2)
    th = torch.nn.functional.conv2d(tx,tw1)
    th.retain_grad()
    ty = torch.nn.functional.conv2d(th,tw2)
    F.sum(y*y).backward(retain_grad=True)
    (ty*ty).sum().backward()
    assert h.requires_grad and h.grad is not None
    assert calls == ([(6,6)] if pruned else [(6,6),(8,8)])
    for label, a, b in [("intermediate",h.grad.data,th.grad.cpu().numpy()),
                         ("w1",w1.grad.data,tw1.grad.cpu().numpy()),("w2",w2.grad.data,tw2.grad.cpu().numpy())]:
        compare(request.node.name+label,a,b,1e-5,1e-4)
    assert (x.grad is None) == pruned


@pytest.mark.parametrize("pruned", [False, True])
def test_conv_weight_higher_order(device, pruned, request):
    xp, name, torch = device
    rng = np.random.default_rng(42)
    x0 = rng.normal(size=(1,2,4,4)).astype(np.float32)*.1
    w0 = rng.normal(size=(2,2,2,2)).astype(np.float32)*.1
    x, w = Variable(xp.asarray(x0),requires_grad=not pruned), Variable(xp.asarray(w0))
    tx = torch.tensor(x0,device=name,requires_grad=not pruned)
    tw = torch.tensor(w0,device=name,requires_grad=True)
    y, ty = FC.conv2d(x,w), torch.nn.functional.conv2d(tx,tw)
    F.sum(y*y).backward(create_graph=True)
    gw = w.grad
    tgw, = torch.autograd.grad((ty*ty).sum(),tw,create_graph=True)
    x.cleargrad(); w.cleargrad()
    F.sum(gw*gw).backward()
    (tgw*tgw).sum().backward()
    compare(request.node.name+"_second_weight",w.grad.data,tw.grad.cpu().numpy(),1e-5,1e-4)
    if pruned:
        assert x.grad is None
    else:
        compare(request.node.name+"_mixed_input",x.grad.data,tx.grad.cpu().numpy(),1e-5,1e-4)


@pytest.mark.parametrize("fused", [False, True])
def test_donkeynet_three_steps(device, fused, request):
    xp, name, torch = device
    from benchmarks.bench_cuda_training import initialize, parameters, torch_model
    from donkeycar_hy.model import DonkeyNet
    baseline, pruned = DonkeyNet(dropout_rate=0.), DonkeyNet(dropout_rate=0.)
    state = initialize(baseline,(3,66,200),42)
    initialize(pruned,(3,66,200),42)
    if name == "cuda":
        baseline.to_gpu(); pruned.to_gpu()
    tm = torch_model("donkey",state).to(name)
    opts = [SGD(.001,fused=fused).setup(m) for m in (baseline,pruned)]
    topt = torch.optim.SGD(tm.parameters(),lr=.001,foreach=True)
    rng = np.random.default_rng(43)
    for step in range(3):
        a = rng.normal(size=(4,3,66,200)).astype(np.float32)
        t = rng.normal(size=(4,1)).astype(np.float32)
        losses, outputs, inputs = [], [], []
        with backend(matmul="shared" if name=="cuda" else "cupy",conv="im2col" if name=="cuda" else "cupy",pool="cupy",tile=16):
            for i,(model,opt) in enumerate(zip((baseline,pruned),opts)):
                x = Variable(xp.asarray(a),requires_grad=i==0)
                model.cleargrads()
                y = model(x)
                loss = F.mean_square_error(y,xp.asarray(t))
                loss.backward()
                outputs.append(y.data); losses.append(loss.data); inputs.append(x)
                opt.update()
        topt.zero_grad(set_to_none=True)
        ty = tm(torch.tensor(a,device=name))
        tl = torch.nn.functional.mse_loss(ty,torch.tensor(t,device=name))
        tl.backward(); topt.step()
        key = request.node.name+f"_step{step}"
        assert inputs[0].grad is not None and inputs[1].grad is None
        compare(key+"_output_baseline",outputs[1],outputs[0],0.,0.)
        compare(key+"_output_torch",outputs[1],ty.detach().cpu().numpy())
        compare(key+"_loss_baseline",losses[1],losses[0],0.,0.)
        compare(key+"_loss_torch",losses[1],np.array(tl.item()))
        bp = parameters(baseline)
        for label,p in parameters(pruned).items():
            layer,attr = label.split("/")
            tp = getattr(getattr(tm,layer),"weight" if attr=="W" else "bias")
            w,g = tp.detach().cpu().numpy(),tp.grad.cpu().numpy()
            if w.ndim==2:w,g=w.T,g.T
            k=key+"_"+label.replace("/","_")
            compare(k+"_weight_baseline",p.data,bp[label].data,0.,0.)
            compare(k+"_gradient_baseline",p.grad.data,bp[label].grad.data,0.,0.)
            compare(k+"_weight_torch",p.data,w)
            compare(k+"_gradient_torch",p.grad.data,g,1e-5,1e-4)
