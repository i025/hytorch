"""Matched DonkeyNet-shape SGD update measurements (no forward/backward).

Full synchronized update wall time and unsynchronized host submission time are
separate runs. Pure GPU kernel execution time must come from Nsight correlation
analysis; no CUDA Event duration is mislabeled as pure kernel time here.
"""
import argparse
from contextlib import contextmanager, nullcontext
import csv
from datetime import datetime, timezone
import functools
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))


def load_training():
    spec = importlib.util.spec_from_file_location(
        "fused_sgd_training", ROOT / "benchmarks/bench_cuda_training.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def describe(values):
    a = np.asarray(values, dtype=np.float64)
    return dict(count=int(a.size), mean_us=float(a.mean()),
                std_us=float(a.std(ddof=1)) if a.size > 1 else 0.,
                median_us=float(np.median(a)), p95_us=float(np.percentile(a, 95)))


def compare(actual, expected):
    delta = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    nonzero = expected != 0
    relative = delta[nonzero] / np.abs(expected[nonzero].astype(np.float64))
    return dict(max_abs=float(delta.max()),
                max_rel_nonzero=float(relative.max()) if relative.size else 0.,
                reference_zero_count=int((~nonzero).sum()),
                passed=bool(np.allclose(actual, expected, atol=1e-6, rtol=1e-5)))


@contextmanager
def nvtx_range(cp, label):
    cp.cuda.nvtx.RangePush(label)
    try:
        yield
    finally:
        cp.cuda.nvtx.RangePop()


@contextmanager
def update_range(cp, mode, repeat, step):
    # The outer name is the existing stage 4A analyzer's phase vocabulary.
    with nvtx_range(cp, "update"):
        with nvtx_range(cp, f"SGD/update|mode={mode}|repeat={repeat}|step={step}"):
            yield


@contextmanager
def runtime_ranges(cp):
    """Trace-only host ranges; no instrumentation in ordinary timing runs."""
    import hytorch.backends.cuda.runtime as runtime
    patches = []
    for name, label in (("_sgd_metadata", "SGD/metadata_prepare"),
                        ("_submit_sgd", "SGD/native_submit"),
                        ("_call", "SGD/ctypes_submit"),
                        ("multi_tensor_sgd", "SGD/native_runtime")):
        original = getattr(runtime, name)

        def make_wrapper(function, range_label):
            @functools.wraps(function)
            def wrapper(*args, **kwargs):
                if range_label == "SGD/ctypes_submit" and args[0] != "hy_multi_tensor_sgd":
                    return function(*args, **kwargs)
                with nvtx_range(cp, range_label):
                    return function(*args, **kwargs)
            return wrapper

        patches.append((name, original))
        setattr(runtime, name, make_wrapper(original, label))
    try:
        yield
    finally:
        for name, original in reversed(patches):
            setattr(runtime, name, original)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["baseline", "fused", "torch"], required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--nvtx", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup < 0 or min(args.updates, args.repeats) < 1:
        parser.error("warmup must be nonnegative; updates/repeats must be positive")
    if not np.isfinite(args.lr):
        parser.error("learning rate must be finite")
    args.output.mkdir(parents=True, exist_ok=False)

    import cupy as cp
    import hytorch
    from hytorch import cuda
    from hytorch.backends.cuda.runtime import load_library, dispatch_stats
    from hytorch.optimizers import SGD
    from donkeycar_hy.model import DonkeyNet

    cp.cuda.Device(args.gpu).use()
    training = load_training()
    model = DonkeyNet(dropout_rate=0.)
    initial = training.initialize(model, (3, 66, 200), args.seed)
    rng = np.random.default_rng(args.seed)
    gradients = {name: (rng.standard_normal(a.shape).astype(np.float32) * np.float32(.01))
                 for name, a in initial.items()}
    expected = {name: a.copy() for name, a in initial.items()}
    for _ in range(args.updates):
        for name in initial:
            delta = np.multiply(np.float32(args.lr), gradients[name], dtype=np.float32)
            np.subtract(expected[name], delta, out=expected[name])
    np.savez(args.output / "initial_weights.npz", **initial)
    np.savez(args.output / "gradients.npz", **gradients)
    np.savez(args.output / "reference_weights.npz", **expected)

    if args.method == "torch":
        import torch
        torch.manual_seed(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.set_device(args.gpu)
        tensors = {name: torch.nn.Parameter(torch.from_numpy(a.copy()).cuda(args.gpu))
                   for name, a in initial.items()}
        for name, parameter in tensors.items():
            parameter.grad = torch.from_numpy(gradients[name].copy()).cuda(args.gpu)
        optimizer = torch.optim.SGD(tensors.values(), lr=args.lr, foreach=True)
        update = optimizer.step
        sync = lambda: torch.cuda.synchronize(args.gpu)

        def restore():
            with torch.no_grad():
                for name, parameter in tensors.items():
                    parameter.copy_(torch.from_numpy(initial[name]).cuda(args.gpu))

        def snapshot():
            return {name: p.detach().cpu().numpy().copy() for name, p in tensors.items()}

        def gradient_snapshot():
            return {name: p.grad.detach().cpu().numpy().copy() for name, p in tensors.items()}
    else:
        if args.method == "fused":
            load_library()
        model.to_gpu()
        tensors = training.parameters(model)
        for name, parameter in tensors.items():
            parameter.grad = hytorch.Variable(cp.asarray(gradients[name]))
        optimizer = SGD(args.lr, fused=args.method == "fused").setup(model)
        update = optimizer.update
        sync = cp.cuda.get_current_stream().synchronize

        def restore():
            for name, parameter in tensors.items():
                cp.copyto(parameter.data, cp.asarray(initial[name]))

        def snapshot():
            return training.snapshot(model)

        def gradient_snapshot():
            return {name: cuda.as_numpy(p.grad.data).copy() for name, p in tensors.items()}

    config = dict(vars(args), output=str(args.output), started_at=datetime.now(timezone.utc).isoformat(),
                  python=sys.version, platform=platform.platform(), numpy=np.__version__, cupy=cp.__version__,
                  cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                  runtime_version=cp.cuda.runtime.runtimeGetVersion(),
                  gpu_name=str(cp.cuda.runtime.getDeviceProperties(args.gpu)["name"]),
                  physical_pci_bus=cp.cuda.Device(args.gpu).pci_bus_id,
                  optimizer="SGD; no momentum, dampening, weight decay, or nesterov",
                  torch_foreach=True if args.method == "torch" else None,
                  dtype="float32", tensor_count=len(initial), elements=sum(a.size for a in initial.values()),
                  parameter_shapes={k: list(a.shape) for k, a in initial.items()},
                  atol=1e-6, rtol=1e-5, reference="CPU FP32 separate multiply and subtract, rounded each update",
                  modes={"sync_wall": "sync before timer; update; sync; stop timer: full host-to-completion wall",
                         "host_submission": "update host call only; no per-update synchronization; batch sync after loop"},
                  pure_gpu_kernel_time="NOT_MEASURED_BY_TIMER; use Nsight CUDA API correlation analysis",
                  instrumentation="NVTX host ranges enabled" if args.nvtx else "NONE",
                  gradient_policy="fixed identical gradients each update; actual training gradients change separately")
    if args.method == "torch":
        config["torch"] = torch.__version__
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    raw = []
    batches = []
    correctness = []
    instrumentation = runtime_ranges(cp) if args.nvtx and args.method == "fused" else nullcontext()
    with instrumentation:
        for repeat in range(1, args.repeats + 1):
            for mode in ("sync_wall", "host_submission"):
                restore()
                for _ in range(args.warmup):
                    update()
                sync()
                restore()
                sync()
                dispatch_stats(reset=True)
                batch_start = time.perf_counter_ns()
                for step in range(1, args.updates + 1):
                    if mode == "sync_wall":
                        sync()
                    range_context = update_range(cp, mode, repeat, step) if args.nvtx else nullcontext()
                    with range_context:
                        started = time.perf_counter_ns()
                        update()
                        if mode == "sync_wall":
                            sync()
                        elapsed_us = (time.perf_counter_ns() - started) / 1000.
                    raw.append(dict(method=args.method, repeat=repeat, mode=mode, step=step,
                                    elapsed_us=elapsed_us, instrumented=args.nvtx))
                sync()
                batch_wall_s = (time.perf_counter_ns() - batch_start) / 1e9
                counts = dispatch_stats()
                final = snapshot()
                final_gradients = gradient_snapshot()
                np.savez(args.output / f"final-{mode}-r{repeat}.npz", **final)
                np.savez(args.output / f"final-gradients-{mode}-r{repeat}.npz", **final_gradients)
                for name in initial:
                    correctness.append(dict(method=args.method, repeat=repeat, mode=mode, parameter=name,
                                            **compare(final[name], expected[name]),
                                            gradient_unchanged=bool(np.array_equal(final_gradients[name], gradients[name]))))
                row = dict(method=args.method, repeat=repeat, mode=mode, updates=args.updates,
                           batch_wall_s=batch_wall_s, dispatch_counts=counts,
                           **describe([r["elapsed_us"] for r in raw if r["repeat"] == repeat and r["mode"] == mode]))
                batches.append(row)
                print(json.dumps(row), flush=True)
    for filename, rows in (("update-benchmark-raw.csv", raw), ("correctness.csv", correctness)):
        with (args.output / filename).open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    passed = all(row["passed"] and row["gradient_unchanged"] for row in correctness)
    summary = dict(status="PASSED" if passed else "FAILED", method=args.method,
                   instrumentation=config["instrumentation"], per_repeat=batches,
                   aggregate={mode: describe([r["elapsed_us"] for r in raw if r["mode"] == mode])
                              for mode in ("sync_wall", "host_submission")},
                   numerical_checks=len(correctness), max_abs=max(r["max_abs"] for r in correctness),
                   max_rel_nonzero=max(r["max_rel_nonzero"] for r in correctness),
                   gradients_unchanged=all(r["gradient_unchanged"] for r in correctness))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    if not passed:
        raise SystemExit("Numerical comparison failed; see correctness.csv")


if __name__ == "__main__":
    main()
