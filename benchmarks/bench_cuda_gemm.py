"""Five-way GEMM comparison. Run from repository root; never simulates CUDA."""
import argparse
import csv
import json
import os
from pathlib import Path
import platform
import sys
import time
import statistics
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hytorch.backends.cuda.runtime import gemm, load_library, library_path


def errors(actual, reference):
    delta = np.asarray(actual, dtype=np.float64)-np.asarray(reference, dtype=np.float64)
    return float(np.max(np.abs(delta))), float(np.linalg.norm(delta)/max(np.linalg.norm(reference), 1e-30))


def cpu_time(fn, repeats, warmup=1):
    times = []
    for _ in range(warmup):
        fn()
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        times.append((time.perf_counter()-start)*1000)
    return result, times


def gpu_time(fn, cp, repeats, warmup=1):
    for _ in range(warmup):
        fn()  # Context/JIT/library warmup excluded.
    cp.cuda.get_current_stream().synchronize()
    times = []
    for _ in range(repeats):
        start, end = cp.cuda.Event(), cp.cuda.Event()
        start.record()
        result = fn()
        end.record()
        end.synchronize()
        times.append(float(cp.cuda.get_elapsed_time(start, end)))
    return result, times


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sizes", nargs="+", type=int, default=[1024, 2048, 4096])
    p.add_argument("--tiles", nargs="+", type=int, choices=[16, 32], default=[16, 32])
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--cpu-only", action="store_true")
    p.add_argument("--output", default="runs/cuda_gemm")
    p.add_argument("--atol", type=float, default=1e-4)
    p.add_argument("--rtol", type=float, default=1e-4)
    p.add_argument("--benefit-speedup", type=float, default=1.5)
    args = p.parse_args()
    if min(args.sizes) <= 0 or args.repeats < 3 or args.warmup < 1:
        p.error("sizes must be positive; at least 3 repeats and 1 warmup required")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    cp = None
    metadata = dict(vars(args), python=sys.version, platform=platform.platform(), numpy=np.__version__,
                    dtype="float32", relative_error="L2 norm(error) / L2 norm(reference)",
                    timing="preallocated device outputs; CUDA events; pageable transfer wall time separately",
                    environment={key: os.environ.get(key) for key in (
                        "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "OMP_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS", "CUPY_TF32", "NVIDIA_TF32_OVERRIDE")},
                    p95_method="numpy.percentile linear interpolation",
                    std_method="sample standard deviation (ddof=1)",
                    transfer_scope="pageable synchronous wall time; H2D copies both inputs once per size",
                    transfer_plus_compute_scope="sum of separate means; NOT measured end-to-end latency",
                    gpu_status="NOT_EXECUTED" if args.cpu_only else "RUNNING")
    if not args.cpu_only:
        import cupy as cp
        cp.cuda.Device(args.device).use()
        load_library()
        metadata.update(cupy=cp.__version__, cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
                        cuda_driver=cp.cuda.runtime.driverGetVersion(),
                        gpu_name=str(cp.cuda.runtime.getDeviceProperties(args.device)["name"]),
                        library=str(library_path()),
                        cublas_version=cp.cuda.cublas.getVersion(cp.cuda.device.get_cublas_handle()),
                        cublas_math_mode_before=cp.cuda.cublas.getMathMode(cp.cuda.device.get_cublas_handle()),
                        cupy_fp32_compute_type=int(cp._core.get_compute_type(cp.float32)),
                        precision_note="Observed handle mode and CuPy policy; kernel-level math not profiled")
    (out/"summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    rows, raw_rows = [], []
    for size in args.sizes:
        rng = np.random.default_rng(args.seed + size)
        a = rng.standard_normal((size, size), dtype=np.float32)
        b = rng.standard_normal((size, size), dtype=np.float32)
        host_out = np.empty((size, size), dtype=np.float32)
        reference, baseline = cpu_time(lambda: np.matmul(a, b, out=host_out), args.repeats, args.warmup)
        entries = [("numpy", None, reference, baseline, [], [])]
        if cp is not None:
            da, db, dy = cp.empty_like(a), cp.empty_like(b), cp.empty_like(host_out)
            def copy_inputs():
                cp.cuda.get_current_stream().synchronize()
                da.set(a); db.set(b)
                cp.cuda.get_current_stream().synchronize()
            _, transfer = cpu_time(copy_inputs, args.repeats, args.warmup)
            raw_rows.extend(dict(size=size, implementation="shared_inputs", tile=None,
                                 phase="h2d", repeat=i, latency_ms=value)
                            for i, value in enumerate(transfer, 1))
            for method, tile in [("cupy", None), ("basic", 16)] + [
                    (m, t) for m in ("blocked", "shared") for t in args.tiles]:
                fn = (lambda: cp.matmul(da, db, out=dy)) if method == "cupy" else (
                    lambda method=method, tile=tile: gemm(da, db, method, tile, out=dy))
                result, times = gpu_time(fn, cp, args.repeats, args.warmup)
                target = np.empty_like(reference)
                _, copies = cpu_time(lambda: result.get(out=target, blocking=True),
                                     args.repeats, args.warmup)
                entries.append((method, None if method == "basic" else tile,
                                target, times, transfer, copies))
            metadata["cublas_math_mode_after"] = cp.cuda.cublas.getMathMode(cp.cuda.device.get_cublas_handle())
        cupy_ms = next((statistics.mean(e[3]) for e in entries if e[0] == "cupy"), None)
        for method, tile, result, times, transfer, copies in entries:
            h2d = statistics.mean(transfer) if transfer else 0.
            d2h = statistics.mean(copies) if copies else 0.
            for phase, values in (("compute", times), ("d2h", copies)):
                raw_rows.extend(dict(size=size, implementation=method, tile=tile, phase=phase,
                                     repeat=i, latency_ms=value) for i, value in enumerate(values, 1))
            absolute, relative = errors(result, reference)
            mean = statistics.mean(times)
            rows.append(dict(implementation=method, size=size, tile=tile,
                             hardware="cpu" if method == "numpy" else "gpu",
                             atol=args.atol, rtol=args.rtol, warmup=args.warmup, repeats=args.repeats,
                             p95_ms=float(np.percentile(times, 95)), gflops=2*size**3/(mean*1e6),
                             mean_ms=mean, std_ms=statistics.stdev(times), median_ms=statistics.median(times),
                             repeats_ms=json.dumps(times), speedup_numpy=statistics.mean(baseline)/mean,
                             speedup_cupy=cupy_ms/mean if cupy_ms else None,
                             max_abs_error=absolute, relative_l2_error=relative,
                             correct=bool(np.allclose(result, reference, atol=args.atol, rtol=args.rtol)),
                             h2d_ms=h2d, d2h_ms=d2h, h2d_repeats_ms=json.dumps(transfer),
                             d2h_repeats_ms=json.dumps(copies),
                             transfer_plus_compute_ms=h2d+mean+d2h))
        # Checkpoint every completed size, retaining measurements even on later failure.
        for filename, data in (("gemm.csv", rows), ("raw-measurements.csv", raw_rows)):
            with (out/filename).open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(data[0]))
                writer.writeheader(); writer.writerows(data)
        del entries, a, b, host_out, reference
        print(f"Finished size {size}", flush=True)
    winners = {}
    for size in args.sizes:
        eligible = [r for r in rows if r["size"] == size and r["correct"] and
                    r["implementation"] in ("cupy", "basic", "blocked", "shared")]
        winners[str(size)] = min(eligible, key=lambda r: r["mean_ms"]) if eligible else None
    qualifying = [r["size"] for r in rows if r["implementation"] in ("basic", "blocked", "shared")
                  and r["correct"] and r["speedup_cupy"] >= args.benefit_speedup]
    metadata.update(gpu_status="NOT_EXECUTED" if args.cpu_only else "EXECUTED",
                    best_per_size=winners, first_measured_beneficial_size=min(qualifying) if qualifying else None,
                    threshold_scope="tested sizes only, speedup versus CuPy, not a universal crossover",
                    all_correct=all(r["correct"] for r in rows))
    (out/"summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if not metadata["all_correct"]:
        raise SystemExit("Numerical check failed; inspect gemm.csv before using timings")


if __name__ == "__main__":
    main()
