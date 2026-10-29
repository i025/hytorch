"""Split completed traces into first and remaining steps using API ownership.

GPU ownership follows the submitting CUDA API's step, never GPU temporal overlap.
Activity coverage is separately clipped to wall-time windows. No GPU is loaded.
"""
import argparse
from bisect import bisect_right
from collections import defaultdict
import csv
import json
from pathlib import Path
import re

from analyze_history import clipped, duration, write_csv


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    scope_rows, phase_rows, allocations, symbol_rows = [], [], [], []
    shapes = {(4, 24, 31, 98): "conv1", (4, 36, 14, 47): "conv2",
              (4, 48, 5, 22): "conv3", (4, 64, 3, 20): "conv4", (4, 64, 1, 18): "conv5"}
    names = ["trace-" + detail + "-" + method
             for detail in ("phase", "detail") for method in ("cupy", "combined", "torch")]
    for name in names:
        source = args.input / name
        summary = json.loads((source / "profiling-summary.json").read_text())
        ranges = read(source / "nvtx-events.csv")
        apis = read(source / "cuda-api-events.csv")
        kernels = read(source / "kernel-events.csv")
        copies = read(source / "memcpy-events.csv")
        phases = [r for r in ranges if r["name"] in ("data", "transfer", "forward", "backward", "update")]
        starts = sorted(int(r["start"]) for r in phases if r["name"] == "data")
        hi = summary["training_envelope_end_ns"]
        assert len(starts) == 16, (name, len(starts))
        by_api = {a["api_event_id"]: a for a in apis}
        for api in apis:
            start = int(api["start"])
            api["step"] = bisect_right(starts, start) - 1 if starts[0] <= start < hi else -1
            if api["step"] >= 0 and api["kind"] == "allocation_or_free":
                allocations.append(dict(trace=name, step=api["step"], phase=api["phase"],
                                        name=api["name"], start_ns=start, duration_ms=float(api["duration_ms"]),
                                        api_event_id=api["api_event_id"], operator_path=api["operator_path"]))
        for event in kernels + copies:
            event["step"] = by_api[event["api_event_id"]]["step"] if event["api_event_id"] in by_api else -1

        for scope, first, last in (("full16", 0, 16), ("first1", 0, 1), ("remaining15", 1, 16)):
            lo, end = starts[first], starts[last] if last < len(starts) else hi
            aa = [a for a in apis if first <= a["step"] < last]
            kk = [k for k in kernels if first <= k["step"] < last]
            cc = [c for c in copies if first <= c["step"] < last]
            activity = clipped([(int(e["start"]), int(e["end"])) for e in kernels + copies], lo, end)
            window_ms = (end - lo) / 1e6
            scope_rows.append(dict(trace=name, scope=scope, steps=last-first,
                                   wall_window_ms=window_ms, wall_window_per_step_ms=window_ms/(last-first),
                                   gpu_kernel_count=len(kk), gpu_kernel_sum_ms=sum(float(k["duration_ms"]) for k in kk),
                                   gpu_kernel_union_ms=duration([(int(k["start"]), int(k["end"])) for k in kk])/1e6,
                                   kernel_memcpy_clipped_union_ms=duration(activity)/1e6,
                                   kernel_memcpy_coverage_fraction=duration(activity)/(end-lo),
                                   cuda_api_sum_ms=sum(float(a["duration_ms"]) for a in aa),
                                   cuda_api_union_ms=duration([(int(a["start"]), int(a["end"])) for a in aa])/1e6,
                                   launch_api_count=sum(a["kind"] == "launch" for a in aa),
                                   sync_api_count=sum(a["kind"] == "synchronization" for a in aa),
                                   sync_api_sum_ms=sum(float(a["duration_ms"]) for a in aa if a["kind"] == "synchronization"),
                                   allocation_api_count=sum(a["kind"] == "allocation_or_free" for a in aa),
                                   allocation_api_sum_ms=sum(float(a["duration_ms"]) for a in aa if a["kind"] == "allocation_or_free")))
            for phase in ("data", "transfer", "forward", "backward", "update", "outside_phase"):
                rr = [r for r in phases if r["name"] == phase and lo <= int(r["start"]) < end]
                pp = [a for a in aa if a["phase"] == phase]
                gg = [k for k in kk if k["phase"] == phase]
                host = sum(float(r["duration_ms"]) for r in rr)
                gpu = sum(float(k["duration_ms"]) for k in gg)
                phase_rows.append(dict(trace=name, scope=scope, phase=phase, steps=last-first,
                                       host_nvtx_sum_ms=host, host_nvtx_per_step_ms=host/(last-first),
                                       gpu_kernel_count=len(gg), gpu_kernel_sum_ms=gpu,
                                       gpu_kernel_sum_per_step_ms=gpu/(last-first),
                                       gpu_kernel_union_ms=duration([(int(k["start"]), int(k["end"])) for k in gg])/1e6,
                                       cuda_api_count=len(pp), cuda_api_sum_ms=sum(float(a["duration_ms"]) for a in pp),
                                       cuda_api_union_ms=duration([(int(a["start"]), int(a["end"])) for a in pp])/1e6,
                                       sync_api_count=sum(a["kind"] == "synchronization" for a in pp),
                                       sync_api_sum_ms=sum(float(a["duration_ms"]) for a in pp if a["kind"] == "synchronization"),
                                       allocation_api_count=sum(a["kind"] == "allocation_or_free" for a in pp),
                                       allocation_api_sum_ms=sum(float(a["duration_ms"]) for a in pp if a["kind"] == "allocation_or_free")))

        grouped = defaultdict(list)
        for k in kernels:
            if k["step"] < 0 or k["phase"] not in ("backward", "update"):
                continue
            layer = "unlabelled"
            if name == "trace-detail-torch" and k["phase"] == "backward":
                conv = next((op for op in json.loads(k["operator_path"])
                             if op.startswith("aten::convolution_backward,")), None)
                if conv:
                    shape = tuple(int(v) for v in re.search(r"sizes = \[\[([0-9, ]+)\]", conv).group(1).split(","))
                    layer = shapes[shape]
            grouped[k["phase"], layer, k["name"], k["demangled_name"]].append(k)
        for (phase, layer, kernel, demangled), events in grouped.items():
            symbol_rows.append(dict(trace=name, phase=phase, layer=layer, kernel=kernel,
                                    demangled_name=demangled, calls=len(events),
                                    gpu_kernel_sum_ms=sum(float(e["duration_ms"]) for e in events)))
    write_csv(args.output / "step-scope-summary.csv", scope_rows)
    write_csv(args.output / "phase-step-scopes.csv", phase_rows)
    write_csv(args.output / "training-allocation-apis.csv", allocations)
    write_csv(args.output / "backward-update-symbols.csv", symbol_rows)
    result = dict(status="COMPLETE", traces=names,
                  step_definition="Each step begins at its data-phase NVTX start; final boundary is final training phase end. First step is retained separately, not removed from full16 results.",
                  attribution="Kernel -> correlation-matched API event -> API submission step; phase from existing same-thread NVTX or explicitly tagged cross-thread unique phase window. Never assign owners by GPU temporal overlap.",
                  coverage="Kernel+Memcpy union clipped to each wall-time window; not SM utilization. First window includes loss logging until next data start; final window ends at the last update NVTX end, before the final StepProfiler.end synchronization and loss readback. API-owned kernel sums may include work that executes after this wall-time boundary.",
                  torch_layer_mapping="Exact DonkeyNet output-gradient shapes in aten::convolution_backward NVTX labels; raw paths remain in kernel-events.csv.",
                  caveat="remaining15 reduces the visible first-step allocation outlier; this is still a single short Nsight trace, not a replacement for the independent 3x256-step timing runs.")
    (args.output / "definitions.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
