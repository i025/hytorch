"""Real GPU numerical equivalence and NVTX overhead; never changes production math.

Run this script only through the project's idle-GPU guard. NPZ files contain raw
inputs, initial parameters and per-tensor numerical results; keep them on server.
Timing uses resident synthetic batches and whole-block synchronization, so it is
not the benchmark's data-loader-inclusive end-to-end throughput measurement.
"""
import argparse
from contextlib import ExitStack
import csv
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

from profile_training import OperatorRanges, load_benchmark, memory_hook


SHAPE = (3, 66, 200)
BACKENDS = {
    "cupy": ("cupy", "cupy", "cupy", 16),
    "combined": ("shared", "im2col", "cupy", 16),
    "torch": ("cupy", "cupy", "cupy", 16),
}
# Unchanged tolerances from test_torch_parity_forward_backward_update in
# tests/test_cuda_host.py:128-145, applied here as a GPU diagnostic comparison.
TOLERANCES = {
    "prediction": (1e-6, 1e-5),
    "loss": (1e-6, 1e-5),
    "gradient": (1e-5, 1e-4),
    "updated": (1e-6, 1e-5),
}


def dump_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compare_tensor(actual, reference, comparison, tensor, exact=False):
    """Compare saved CPU values in FP64 without changing their measured values."""
    a, b = np.asarray(actual), np.asarray(reference)
    if a.shape != b.shape:
        raise ValueError("Shape mismatch for {}: {} != {}".format(tensor, a.shape, b.shape))
    category = tensor.split(":", 1)[0]
    atol, rtol = (0., 0.) if exact else TOLERANCES[category]
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    difference = np.abs(a.astype(np.float64) - b.astype(np.float64))
    denominator = np.abs(b.astype(np.float64))
    nonzero = denominator != 0
    relative = difference[nonzero] / denominator[nonzero]
    allowed = atol + rtol * denominator
    failures = int(np.count_nonzero(~np.isfinite(difference) | (difference > allowed)))
    equality = bool(a.dtype == b.dtype and np.array_equal(a, b))
    norm = float(np.linalg.norm(b.astype(np.float64).ravel()))
    error_norm = float(np.linalg.norm((a.astype(np.float64) - b.astype(np.float64)).ravel()))
    return dict(comparison=comparison, tensor=tensor, shape="x".join(map(str, a.shape)) or "scalar",
                elements=int(a.size), actual_dtype=str(a.dtype), reference_dtype=str(b.dtype),
                finite=finite, exact_equal=equality, atol=atol, rtol=rtol,
                failed_elements=failures, max_abs_error=float(difference.max()) if finite else None,
                max_rel_error_nonzero_reference=float(relative.max()) if finite and relative.size else None,
                zero_reference_elements=int(np.count_nonzero(~nonzero)),
                max_abs_error_zero_reference=float(difference[~nonzero].max()) if finite and np.any(~nonzero) else None,
                relative_l2_error=error_norm / norm if finite and norm else None,
                correct=finite and (equality if exact else failures == 0))


def make_hytorch(bench, initial, cp):
    from donkeycar_hy.model import DonkeyNet
    model = DonkeyNet(dropout_rate=0.)
    bench.initialize(model, SHAPE, 42)
    for name, parameter in bench.parameters(model).items():
        parameter.data = cp.asarray(initial[name])
    return model


def torch_arrays(model, initial, gradient=False):
    arrays = {}
    for name in initial:
        layer, parameter = name.split("/")
        value = getattr(getattr(model, layer), "weight" if parameter == "W" else "bias")
        value = value.grad if gradient else value
        host = value.detach().cpu().numpy()
        arrays[name] = host.T.copy() if host.ndim == 2 else host.copy()
    return arrays


def enter_instrumentation(stack, method, enabled, memory_stats, cp, torch):
    if not enabled:
        return None
    if method == "torch":
        # PyTorch synchronizes on entering/exiting emit_nvtx. Both boundaries
        # are outside the measured training block below.
        stack.enter_context(torch.autograd.profiler.emit_nvtx(record_shapes=True))
        return None
    recorder = stack.enter_context(OperatorRanges(cp.cuda.nvtx.RangePush, cp.cuda.nvtx.RangePop, cp))
    if memory_stats:
        stack.enter_context(memory_hook(cp, recorder))
    return recorder


def numerical_step(method, instrumented, bench, initial, dataset, cp, torch, memory_stats):
    import hytorch
    import hytorch.functions as F
    from hytorch.backends.cuda import backend, dispatch_stats
    from hytorch.optimizers import SGD

    if method == "torch":
        model = bench.torch_model("donkey", initial).to("cuda:0")
        optimizer = torch.optim.SGD(model.parameters(), lr=.001)
        x = torch.from_numpy(dataset.x).to("cuda:0")
        target = torch.from_numpy(dataset.t).to("cuda:0")
    else:
        model = make_hytorch(bench, initial, cp)
        optimizer = SGD(.001).setup(model)
        x, target = hytorch.Variable(cp.asarray(dataset.x)), hytorch.Variable(cp.asarray(dataset.t))
    actual_initial = torch_arrays(model, initial) if method == "torch" else bench.snapshot(model)
    actual_input = x.detach().cpu().numpy().copy() if method == "torch" else cp.asnumpy(x.data)
    actual_target = target.detach().cpu().numpy().copy() if method == "torch" else cp.asnumpy(target.data)
    dispatch_stats(reset=True)
    with backend(*BACKENDS[method]), ExitStack() as stack:
        recorder = enter_instrumentation(stack, method, instrumented, memory_stats, cp, torch)
        prediction = model(x)
        if method == "torch":
            loss = torch.nn.functional.mse_loss(prediction, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradients = torch_arrays(model, initial, gradient=True)
            result = dict(prediction=prediction.detach().cpu().numpy().copy(),
                          loss=loss.detach().cpu().numpy().copy())
            optimizer.step()
            updated = torch_arrays(model, initial)
            leaf_evidence = dict(input_requires_grad=x.requires_grad, target_requires_grad=target.requires_grad,
                                 input_grad_present=x.grad is not None, target_grad_present=target.grad is not None)
        else:
            loss = F.mean_square_error(prediction, target)
            model.cleargrads()
            loss.backward()
            gradients = {name: cp.asnumpy(parameter.grad.data) for name, parameter in bench.parameters(model).items()}
            result = dict(prediction=cp.asnumpy(prediction.data), loss=cp.asnumpy(loss.data),
                          input_gradient=cp.asnumpy(x.grad.data), target_gradient=cp.asnumpy(target.grad.data))
            optimizer.update()
            updated = bench.snapshot(model)
            leaf_evidence = dict(input_grad_present=x.grad is not None, target_grad_present=target.grad is not None,
                                 input_gradient_shape=list(x.grad.shape), target_gradient_shape=list(target.grad.shape),
                                 input_gradient_l2=float(np.linalg.norm(result["input_gradient"].astype(np.float64))),
                                 target_gradient_l2=float(np.linalg.norm(result["target_gradient"].astype(np.float64))))
    result.update({"gradient:" + name: value for name, value in gradients.items()})
    result.update({"updated:" + name: value for name, value in updated.items()})
    result.update({"initial:" + name: value for name, value in actual_initial.items()})
    result.update(input=actual_input, target=actual_target)
    metadata = dict(method=method, instrumented=instrumented, leaf_gradients=leaf_evidence,
                    dispatch_counts=dispatch_stats(),
                    initial_arrays_exact=all(np.array_equal(actual_initial[name], value) for name, value in initial.items()),
                    input_arrays_exact=bool(np.array_equal(actual_input, dataset.x) and np.array_equal(actual_target, dataset.t)),
                    all_float32=all(value.dtype == np.float32 for value in result.values()),
                    all_finite=all(bool(np.isfinite(value).all()) for value in result.values()))
    if recorder is not None:
        metadata["instrumentation"] = recorder.summary()
    return result, metadata


def timing_block(method, instrumented, bench, initial, dataset, cp, torch, args):
    import hytorch.functions as F
    from hytorch.backends.cuda import backend, dispatch_stats
    from hytorch.optimizers import SGD

    if method == "torch":
        model = bench.torch_model("donkey", initial).to("cuda:0")
        optimizer = torch.optim.SGD(model.parameters(), lr=.001)
        x, target = torch.from_numpy(dataset.x).to("cuda:0"), torch.from_numpy(dataset.t).to("cuda:0")
        saved = {name: value.detach().clone() for name, value in model.state_dict().items()}

        def step():
            loss = torch.nn.functional.mse_loss(model(x), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            return loss

        restore = lambda: model.load_state_dict(saved)
        sync = torch.cuda.synchronize
        scalar = lambda value: float(value.detach().cpu())
    else:
        model = make_hytorch(bench, initial, cp)
        optimizer = SGD(.001).setup(model)
        x, target = cp.asarray(dataset.x), cp.asarray(dataset.t)

        def step():
            loss = F.mean_square_error(model(x), target)
            model.cleargrads()
            loss.backward()
            optimizer.update()
            return loss

        restore = lambda: [setattr(parameter, "data", cp.asarray(initial[name]))
                           for name, parameter in bench.parameters(model).items()]
        sync = cp.cuda.get_current_stream().synchronize
        scalar = lambda value: float(cp.asnumpy(value.data))
    with backend(*BACKENDS[method]), ExitStack() as stack:
        recorder = enter_instrumentation(stack, method, instrumented, args.memory_stats, cp, torch)
        for _ in range(args.warmup):
            step()
        sync()
        restore()
        dispatch_stats(reset=True)
        sync()
        started = time.perf_counter()
        for _ in range(args.steps):
            loss = step()
        sync()
        wall_s = time.perf_counter() - started
        final_loss = scalar(loss)
    return dict(method=method, instrumented=instrumented, memory_stats=args.memory_stats and instrumented and method != "torch",
                steps=args.steps, batch=4, warmup=args.warmup, wall_s=wall_s,
                step_ms=1000 * wall_s / args.steps, samples_per_second=4 * args.steps / wall_s,
                final_loss=final_loss, finite_loss=bool(np.isfinite(final_loss)),
                dispatch_counts=json.dumps(dispatch_stats(), sort_keys=True)), recorder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, choices=[0], default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--memory-stats", action="store_true")
    parser.add_argument("--no-fingerprints", action="store_true", required=True)
    args = parser.parse_args()
    if min(args.steps, args.repeats, args.warmup) <= 0:
        parser.error("steps, repeats and warmup must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    bench = load_benchmark()
    import cupy as cp
    import torch
    from donkeycar_hy.model import DonkeyNet
    from hytorch.backends.cuda.runtime import load_library
    cp.cuda.Device(args.device).use()
    torch.cuda.set_device(args.device)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    config = dict(status="RUNNING", model="DonkeyNet", input_shape=[4, *SHAPE], dtype="float32",
                  seed=42, optimizer="SGD", lr=.001, dropout=0., backend_settings=BACKENDS,
                  logical_gpu=args.device, CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES"),
                  python=sys.version, platform=platform.platform(), numpy=np.__version__, cupy=cp.__version__,
                  torch=torch.__version__, cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
                  cudnn=torch.backends.cudnn.version(), tf32=False, cudnn_deterministic=True,
                  tolerances=TOLERANCES, tolerance_source="tests/test_cuda_host.py:128-145; unchanged host parity thresholds reused for this GPU diagnostic",
                  instrumentation_comparison="exact dtype, shape and elementwise equality",
                  timing_scope="resident fixed synthetic batch; warmup then restore; synchronization only before/after whole step block; excludes input transfer, DataLoader, scalar reads, file IO and NVTX context entry/exit",
                  timing_steps=args.steps, timing_warmup=args.warmup, timing_repeats=args.repeats,
                  memory_stats=args.memory_stats, fingerprints="DISABLED")
    dump_json(args.output / "config.json", config)
    load_library()
    initial = bench.initialize(DonkeyNet(dropout_rate=0.), SHAPE, 42)
    dataset = bench.SyntheticDataset(4, SHAPE, 42)
    np.savez(args.output / "initial-weights.npz", **initial)
    np.savez(args.output / "inputs.npz", x=dataset.x, target=dataset.t)
    saved, checks = {}, []
    for method in BACKENDS:
        for instrumented in (False, True):
            name = method + ("-instrumented" if instrumented else "-plain")
            values, metadata = numerical_step(method, instrumented, bench, initial, dataset, cp, torch, args.memory_stats)
            np.savez(args.output / (name + ".npz"), **values)
            dump_json(args.output / (name + ".json"), metadata)
            saved[name] = values
            checks.append(metadata)
            print(json.dumps(dict(event="numerical_step", run=name, loss=float(values["loss"]),
                                  all_finite=metadata["all_finite"], all_float32=metadata["all_float32"],
                                  leaf_gradients=metadata["leaf_gradients"])), flush=True)
    rows = []
    for method in BACKENDS:
        plain, observed = saved[method + "-plain"], saved[method + "-instrumented"]
        for tensor in plain:
            rows.append(compare_tensor(observed[tensor], plain[tensor], method + ":instrumented_vs_plain", tensor, exact=True))
    for method in ("cupy", "combined"):
        for tensor, reference in saved["torch-plain"].items():
            rows.append(compare_tensor(saved[method + "-plain"][tensor], reference,
                                       method + ":vs_torch", tensor,
                                       exact=tensor.split(":", 1)[0] in ("initial", "input", "target")))
    write_csv(args.output / "tensor-comparisons.csv", rows)
    correctness_passed = all(row["correct"] for row in rows) and all(
        row["all_finite"] and row["all_float32"] and row["initial_arrays_exact"] and row["input_arrays_exact"]
        for row in checks)
    numerical = dict(status="PASSED" if correctness_passed else "FAILED", comparisons=len(rows),
                     failed_comparisons=[row for row in rows if not row["correct"]],
                     exact_instrumentation_comparisons=sum(":instrumented_vs_plain" in row["comparison"] for row in rows),
                     torch_tolerance_comparisons=sum(":vs_torch" in row["comparison"] and row["atol"] > 0 for row in rows),
                     torch_initial_and_input_exact_comparisons=sum(":vs_torch" in row["comparison"] and row["atol"] == 0 for row in rows))
    dump_json(args.output / "numerical-summary.json", numerical)
    print(json.dumps(dict(event="numerical_summary", **numerical)), flush=True)
    if not correctness_passed:
        dump_json(args.output / "summary.json", dict(status="FAILED", numerical=numerical, timing_status="NOT_RUN_NUMERICAL_FAILURE"))
        raise SystemExit(1)
    timing = []
    methods = list(BACKENDS)
    for repeat in range(args.repeats):
        # Rotate method order and alternate condition order to expose order bias.
        order = methods[repeat % len(methods):] + methods[:repeat % len(methods)]
        for method in order:
            for instrumented in ((False, True) if repeat % 2 == 0 else (True, False)):
                row, recorder = timing_block(method, instrumented, bench, initial, dataset, cp, torch, args)
                row.update(repeat=repeat + 1, sequence=len(timing) + 1)
                timing.append(row)
                if recorder is not None:
                    dump_json(args.output / "timing-{}-repeat-{}-instrumentation.json".format(method, repeat + 1), recorder.summary())
                print(json.dumps(dict(event="timing_block", **row)), flush=True)
                write_csv(args.output / "instrumentation-timing.csv", timing)
    overhead = []
    for method in methods:
        plain = [row["step_ms"] for row in timing if row["method"] == method and not row["instrumented"]]
        observed = [row["step_ms"] for row in timing if row["method"] == method and row["instrumented"]]
        overhead.append(dict(method=method, plain_mean_step_ms=float(np.mean(plain)),
                             instrumented_mean_step_ms=float(np.mean(observed)),
                             plain_std_step_ms=float(np.std(plain, ddof=1)) if args.repeats > 1 else None,
                             instrumented_std_step_ms=float(np.std(observed, ddof=1)) if args.repeats > 1 else None,
                             overhead_percent=100 * (float(np.mean(observed)) / float(np.mean(plain)) - 1)))
    passed = all(row["finite_loss"] for row in timing)
    summary = dict(status="PASSED" if passed else "FAILED", gpu_status="EXECUTED", numerical=numerical,
                   timing_scope=config["timing_scope"], timing_blocks=len(timing), overhead=overhead,
                   raw_arrays="inputs.npz, initial-weights.npz and {cupy,combined,torch}-{plain,instrumented}.npz; server only")
    dump_json(args.output / "summary.json", summary)
    print(json.dumps(dict(event="completed", **summary)), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
