"""Real CUDA checks. Set HYTORCH_REQUIRE_CUDA=1 after allocating an idle GPU."""
import os
import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hytorch import Variable
from hytorch.core import Parameter
from hytorch.optimizers import SGD, FreezeParam, WeightDecay
from hytorch.backends.cuda import backend
from hytorch.backends.cuda.runtime import load_library, multi_tensor_sgd


ATOL, RTOL = 1e-6, 1e-5  # Existing PyTorch parameter-update parity tolerance.


def save_case(case, arrays, comparisons):
    """Preserve actual arrays locally and error rows before numerical assertions."""
    destination = os.environ.get("HYTORCH_SGD_TEST_OUTPUT")
    if not destination:
        return
    output = Path(destination)
    output.mkdir(parents=True, exist_ok=True)
    with (output / (case + ".npz")).open("xb") as file:
        np.savez(file, **arrays)
    rows = []
    for actual, expected, atol, rtol in comparisons:
        a, b = arrays[actual].astype(np.float64), arrays[expected].astype(np.float64)
        error = np.abs(a-b)
        relative = np.divide(error, np.abs(b), out=np.full_like(error, np.inf), where=b != 0)
        relative = np.where((b == 0) & (error == 0), 0., relative)
        rows.append(dict(case=case, actual=actual, expected=expected, elements=a.size,
                         max_abs_error=float(error.max()), max_rel_error=float(relative.max()),
                         zero_reference_elements=int(np.count_nonzero(b == 0)),
                         mismatches=int(np.count_nonzero(error > atol+rtol*np.abs(b))),
                         finite=bool(np.isfinite(a).all() and np.isfinite(b).all()),
                         atol=atol, rtol=rtol, arrays_file=case+".npz"))
    csv_path = output / "correctness.csv"
    new_file = not csv_path.exists()
    with csv_path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


@pytest.fixture(scope="module")
def cp():
    if os.environ.get("HYTORCH_REQUIRE_CUDA") != "1":
        pytest.skip("CUDA NOT_RUN: explicitly allocate a GPU and set HYTORCH_REQUIRE_CUDA=1")
    import cupy
    cupy.cuda.Device(int(os.environ.get("HYTORCH_TEST_DEVICE", "0"))).use()
    load_library()  # An explicitly requested CUDA run must fail, never skip.
    return cupy


def make_model(cp, values, gradients):
    params = [Parameter(cp.asarray(value)) for value in values]
    for param, gradient in zip(params, gradients):
        param.grad = None if gradient is None else Variable(cp.asarray(gradient))
    return SimpleNamespace(params=lambda: iter(params)), params


@pytest.mark.parametrize("shapes", [
    [()], [(1,)], [(17,)], [(257,)], [(1048583,)],
    [(1,), (7,), (257,), (31, 33), (4099,)],
], ids=["scalar", "one", "tiny", "block_tail", "large", "mixed_sizes"])
@pytest.mark.parametrize("foreach", [False, True])
def test_kernel_matches_baseline_and_torch_inplace(cp, shapes, foreach, request):
    import torch
    rng = np.random.default_rng(42)
    values = [rng.normal(size=shape).astype(np.float32) for shape in shapes]
    gradients = [rng.normal(size=shape).astype(np.float32) for shape in shapes]
    baseline, bp = make_model(cp, values, gradients)
    fused, fp = make_model(cp, values, gradients)
    identities = [param.data for param in fp]
    addresses = [param.data.data.ptr for param in fp]
    tp = [torch.nn.Parameter(torch.tensor(value, device="cuda")) for value in values]
    for param, gradient in zip(tp, gradients):
        param.grad = torch.tensor(gradient, device="cuda")
    SGD(lr=.001).setup(baseline).update()
    # Direct kernel validation precedes optimizer integration validation.
    multi_tensor_sgd([p.data for p in fp], [p.grad.data for p in fp], .001)
    torch.optim.SGD(tp, lr=.001, foreach=foreach).step()
    cp.cuda.get_current_stream().synchronize()
    for i, (actual, expected, reference) in enumerate(zip(fp, bp, tp)):
        save_case(request.node.name+f"_tensor_{i}", dict(initial=values[i],
                  gradient=gradients[i], baseline=cp.asnumpy(expected.data),
                  fused=cp.asnumpy(actual.data), torch=reference.detach().cpu().numpy(),
                  gradient_after=cp.asnumpy(actual.grad.data)),
                  [("fused", "baseline", ATOL, RTOL), ("fused", "torch", ATOL, RTOL),
                   ("gradient_after", "gradient", 0., 0.)])
        assert actual.data is identities[i] and actual.data.data.ptr == addresses[i]
        np.testing.assert_allclose(cp.asnumpy(actual.data), cp.asnumpy(expected.data), atol=ATOL, rtol=RTOL)
        np.testing.assert_allclose(cp.asnumpy(actual.data), reference.detach().cpu().numpy(), atol=ATOL, rtol=RTOL)
        np.testing.assert_array_equal(cp.asnumpy(actual.grad.data), gradients[i])


def test_repeated_updates_use_fresh_gradient_addresses(cp):
    rng = np.random.default_rng(42)
    values = [rng.normal(size=shape).astype(np.float32) for shape in [(17,), (4099,)]]
    baseline, bp = make_model(cp, values, [None, None])
    fused, fp = make_model(cp, values, [None, None])
    optimizers = [SGD(lr=.001).setup(baseline), SGD(lr=.001, fused=True).setup(fused)]
    old_gradients = []  # Keep storage alive so allocator reuse cannot hide stale pointers.
    previous = None
    for step in range(10):
        gradients = [rng.normal(size=value.shape).astype(np.float32) for value in values]
        for params in (bp, fp):
            for param, gradient in zip(params, gradients):
                param.grad = Variable(cp.asarray(gradient))
                old_gradients.append(param.grad.data)
        current = tuple(param.grad.data.data.ptr for param in fp)
        assert current != previous
        previous = current
        for optimizer in optimizers:
            optimizer.update()
        cp.cuda.get_current_stream().synchronize()
        for i, (a, b, gradient) in enumerate(zip(fp, bp, gradients)):
            save_case(f"fresh_gradients_step_{step}_tensor_{i}",
                      dict(initial=values[i], fused=cp.asnumpy(a.data), baseline=cp.asnumpy(b.data),
                           gradient=gradient, gradient_after=cp.asnumpy(a.grad.data),
                           gradient_pointer=np.array(current[i], np.uint64)),
                      [("fused", "baseline", ATOL, RTOL), ("gradient_after", "gradient", 0., 0.)])
            np.testing.assert_allclose(cp.asnumpy(a.data), cp.asnumpy(b.data), atol=ATOL, rtol=RTOL,
                                       err_msg=f"step {step}")
            np.testing.assert_array_equal(cp.asnumpy(a.grad.data), gradient)


def test_current_nondefault_stream_producer_update_consumer(cp):
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        param = cp.arange(4099, dtype=cp.float32)
        gradient = cp.full_like(param, .75)
        # Both producer and consumer are queued on the caller's stream.
        gradient *= np.float32(2)
        multi_tensor_sgd([param], [gradient], .125)
        consumed = param + np.float32(.25)
        stream.synchronize()
        actual, grad = cp.asnumpy(consumed), cp.asnumpy(gradient)
    save_case("nondefault_stream", dict(actual=actual, expected=np.arange(4099, dtype=np.float32)+.0625,
              gradient_after=grad, gradient=np.full(4099, 1.5, np.float32)),
              [("actual", "expected", 0., 0.), ("gradient_after", "gradient", 0., 0.)])
    np.testing.assert_array_equal(actual, np.arange(4099, dtype=np.float32)+.0625)
    np.testing.assert_array_equal(grad, np.full(4099, 1.5, np.float32))


def test_hooks_missing_gradient_freeze_and_all_frozen(cp):
    values = [np.ones(7, np.float32) for _ in range(3)]
    gradients = [np.ones(7, np.float32), np.ones(7, np.float32), None]
    outputs = []
    for fused in (False, True):
        model, params = make_model(cp, values, gradients)
        seen = []
        opt = SGD(lr=.125, fused=fused).setup(model)
        opt.add_hook(lambda active: seen.append(tuple(active)))
        opt.add_hook(WeightDecay(.5))
        opt.add_hook(FreezeParam(params[1]))
        opt.update()
        assert seen == [tuple(params[:2])]
        assert params[1].grad is None and params[2].grad is None
        cp.cuda.get_current_stream().synchronize()
        outputs.append([cp.asnumpy(p.data) for p in params])
        opt.add_hook(FreezeParam(params[0]))
        opt.update()  # All filtered out: no empty CUDA launch.
        for param, expected in zip(params, outputs[-1]):
            np.testing.assert_array_equal(cp.asnumpy(param.data), expected)
    for i, (actual, expected) in enumerate(zip(*outputs)):
        save_case(f"hooks_parameter_{i}", dict(baseline=actual, fused=expected, initial=values[i]),
                  [("fused", "baseline", 0., 0.)])
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(outputs[1][0], np.full(7, .8125, np.float32))
    for frozen in outputs[1][1:]:
        np.testing.assert_array_equal(frozen, np.ones(7, np.float32))


def test_direct_missing_gradients_empty_and_supported_boundary(cp):
    params = [cp.ones(3, cp.float32) for _ in range(64)]
    grads = [cp.ones(3, cp.float32) for _ in range(64)]
    multi_tensor_sgd([], [], .125)
    multi_tensor_sgd(params, [None]*64, .125)
    multi_tensor_sgd(params, grads, .125)
    cp.cuda.get_current_stream().synchronize()
    for param in params:
        np.testing.assert_array_equal(cp.asnumpy(param), np.full(3, .875, np.float32))


def test_unsupported_arrays_rejected_without_parameter_updates(cp):
    param = cp.ones(8, cp.float32)
    gradient = cp.ones(8, cp.float32)
    with pytest.raises(TypeError):
        multi_tensor_sgd([param], [gradient.astype(cp.float64)], .001)
    with pytest.raises(TypeError):
        multi_tensor_sgd([param], [np.ones(8, np.float32)], .001)
    with pytest.raises(ValueError):
        multi_tensor_sgd([param[::2]], [gradient[::2]], .001)
    with pytest.raises(ValueError):
        multi_tensor_sgd([param], [gradient.reshape(2, 4)], .001)
    with pytest.raises(ValueError):
        multi_tensor_sgd([param], [param], .001)
    with pytest.raises(ValueError):
        multi_tensor_sgd([param, param], [gradient, gradient], .001)
    with pytest.raises(ValueError):
        multi_tensor_sgd([param], [], .001)
    with pytest.raises(ValueError):
        multi_tensor_sgd([cp.ones(1, cp.float32) for _ in range(65)],
                         [cp.ones(1, cp.float32) for _ in range(65)], .001)
    cp.cuda.get_current_stream().synchronize()
    np.testing.assert_array_equal(cp.asnumpy(param), np.ones(8, np.float32))


def test_donkeynet_loss_gradients_and_updates(cp):
    import torch
    import hytorch.functions as F
    from benchmarks.bench_cuda_training import initialize, parameters, torch_model
    from donkeycar_hy.model import DonkeyNet

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    baseline, fused = DonkeyNet(dropout_rate=0.), DonkeyNet(dropout_rate=0.)
    state = initialize(baseline, (3, 66, 200), 42)
    initialize(fused, (3, 66, 200), 42)
    baseline.to_gpu()
    fused.to_gpu()
    reference = torch_model("donkey", state).to("cuda")
    optimizers = [SGD(lr=.001).setup(baseline), SGD(lr=.001, fused=True).setup(fused)]
    torch_optimizer = torch.optim.SGD(reference.parameters(), lr=.001, foreach=True)
    rng = np.random.default_rng(43)
    for step in range(3):
        x = rng.normal(size=(4, 3, 66, 200)).astype(np.float32)
        target = rng.normal(size=(4, 1)).astype(np.float32)
        losses = []
        for model, optimizer in zip((baseline, fused), optimizers):
            model.cleargrads()
            with backend(matmul="shared", conv="im2col", pool="cupy", tile=16):
                loss = F.mean_square_error(model(cp.asarray(x)), cp.asarray(target))
                loss.backward()
            losses.append(cp.asnumpy(loss.data))
            optimizer.update()
        torch_optimizer.zero_grad(set_to_none=True)
        torch_loss = torch.nn.functional.mse_loss(reference(torch.tensor(x, device="cuda")),
                                                   torch.tensor(target, device="cuda"))
        torch_loss.backward()
        torch_optimizer.step()
        save_case(f"donkey_loss_step_{step}",
                  dict(baseline=losses[0], fused=losses[1], torch=np.array(torch_loss.item()), x=x, target=target),
                  [("fused", "baseline", 0., 0.), ("fused", "torch", ATOL, RTOL)])
        np.testing.assert_array_equal(losses[0], losses[1], err_msg=f"loss step {step}")
        np.testing.assert_allclose(losses[1], torch_loss.item(), atol=ATOL, rtol=RTOL)
        bp = parameters(baseline)
        for key, param in parameters(fused).items():
            layer, name = key.split("/")
            torch_param = getattr(getattr(reference, layer), "weight" if name == "W" else "bias")
            torch_weight = torch_param.detach().cpu().numpy()
            torch_grad = torch_param.grad.cpu().numpy()
            if torch_weight.ndim == 2:
                torch_weight, torch_grad = torch_weight.T, torch_grad.T
            save_case(f"donkey_step_{step}_"+key.replace("/", "_"),
                      dict(initial=state[key], fused=cp.asnumpy(param.data), baseline=cp.asnumpy(bp[key].data),
                           torch=torch_weight, gradient=cp.asnumpy(param.grad.data),
                           baseline_gradient=cp.asnumpy(bp[key].grad.data), torch_gradient=torch_grad),
                      [("fused", "baseline", 0., 0.), ("fused", "torch", ATOL, RTOL),
                       ("gradient", "baseline_gradient", 0., 0.), ("gradient", "torch_gradient", 1e-5, 1e-4)])
            np.testing.assert_array_equal(cp.asnumpy(param.data), cp.asnumpy(bp[key].data), err_msg=key)
            np.testing.assert_array_equal(cp.asnumpy(param.grad.data), cp.asnumpy(bp[key].grad.data), err_msg=key)
            np.testing.assert_allclose(cp.asnumpy(param.data), torch_weight, atol=ATOL, rtol=RTOL, err_msg=key)
            np.testing.assert_allclose(cp.asnumpy(param.grad.data), torch_grad, atol=1e-5, rtol=1e-4, err_msg=key)
