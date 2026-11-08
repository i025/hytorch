"""Paired shared-GEMM measurements of two CUDA libraries on identical inputs.

Both libraries must implement the current ABI. Strict NumPy allclose results
remain visible; revision acceptance requires bitwise-preserved baseline output
and relative L2 error <= 1e-5 against FP64. This is not a replacement for the
repository's stricter original benchmark gate.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hytorch.backends.cuda import runtime


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--sizes", nargs="+", type=int, default=[1024, 2048, 4096])
    p.add_argument("--tiles", nargs="+", type=int, choices=[16, 32], default=[16, 32])
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    if args.repeats < 3 or min(args.sizes) <= 0:
        p.error("Use positive sizes and at least three repeats")
    import cupy as cp
    cp.cuda.Device(args.device).use()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    libs, hashes = {}, {}
    for name in ("baseline", "candidate"):
        path = Path(getattr(args, name)).resolve()
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        os.environ["HYTORCH_CUDA_LIBRARY"] = str(path)
        runtime._library = None
        libs[name] = runtime.load_library()
    metadata = dict(vars(args), library_sha256=hashes, numpy=np.__version__,
                    cupy=cp.__version__, gpu_name=str(cp.cuda.runtime.getDeviceProperties(args.device)["name"]),
                    pci_bus_id=cp.cuda.Device(args.device).pci_bus_id,
                    cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
                    cuda_driver=cp.cuda.runtime.driverGetVersion(), seed="42+size",
                    cpu_threads={k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
                    timing="preallocated outputs, CUDA events, alternating revision order, 3 warmups",
                    acceptance="bitwise baseline equality AND FP64 relative L2 <= 1e-5",
                    original_gate="NumPy FP32 reference, atol=1e-4, rtol=1e-4; reported separately")
    (out / "config.json").write_text(json.dumps(metadata, indent=2))
    rows = []
    for size in args.sizes:
        rng = np.random.default_rng(42 + size)
        a = rng.standard_normal((size, size), dtype=np.float32)
        b = rng.standard_normal((size, size), dtype=np.float32)
        ref32 = a @ b
        ref64 = a.astype(np.float64) @ b.astype(np.float64)
        da, db = cp.asarray(a), cp.asarray(b)
        outputs = {name: cp.empty_like(da) for name in libs}
        for tile in args.tiles:
            times = {name: [] for name in libs}
            for name, lib in libs.items():
                runtime._library = lib
                for _ in range(3):
                    runtime.gemm(da, db, "shared", tile, out=outputs[name])
            cp.cuda.get_current_stream().synchronize()
            for repeat in range(args.repeats):
                order = list(libs) if repeat % 2 == 0 else list(reversed(libs))
                for name in order:
                    runtime._library = libs[name]
                    start, end = cp.cuda.Event(), cp.cuda.Event()
                    start.record()
                    runtime.gemm(da, db, "shared", tile, out=outputs[name])
                    end.record()
                    end.synchronize()
                    times[name].append(float(cp.cuda.get_elapsed_time(start, end)))
            old, new = (cp.asnumpy(outputs[name]) for name in ("baseline", "candidate"))
            relative = float(np.linalg.norm(new.astype(np.float64)-ref64)/np.linalg.norm(ref64))
            equal = bool(np.array_equal(old, new))
            row = dict(size=size, tile=tile,
                       baseline_median_ms=statistics.median(times["baseline"]),
                       candidate_median_ms=statistics.median(times["candidate"]),
                       speedup=statistics.median(times["baseline"])/statistics.median(times["candidate"]),
                       baseline_repeats_ms=json.dumps(times["baseline"]),
                       candidate_repeats_ms=json.dumps(times["candidate"]),
                       bitwise_equal=equal, fp64_relative_l2_error=relative,
                       fp64_max_abs_error=float(np.max(np.abs(new-ref64))),
                       baseline_original_gate=bool(np.allclose(old, ref32, atol=1e-4, rtol=1e-4)),
                       candidate_original_gate=bool(np.allclose(new, ref32, atol=1e-4, rtol=1e-4)),
                       revision_accepted=equal and relative <= 1e-5)
            rows.append(row)
            print(json.dumps(row), flush=True)
        with (out / "comparison.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if not all(row["revision_accepted"] for row in rows):
        raise SystemExit("Revision check failed; inspect comparison.csv")


if __name__ == "__main__":
    main()
