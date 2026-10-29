"""Summarize stage 4B traces through existing correlation-based stage 4A exports.

CPU-only. Raw GPU events are attributed by submitting API and NVTX ancestry,
never by overlap with the asynchronous GPU execution interval. Host ranges,
CUDA API time, and GPU event time remain separate measurements.
"""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np


def read_csv(path):
    if not path.exists():
        return []
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if not rows:
        return
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def intervals_ns(events):
    return [(int(event["start"]), int(event["end"])) for event in events]


def union_ns(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum(end-start for start, end in merged)


def event_metrics(kernels, copies, apis):
    launch = [api for api in apis if api["kind"] == "launch"]
    sync = [api for api in apis if api["kind"] == "synchronization"]
    gpu_span = max(int(k["end"]) for k in kernels)-min(int(k["start"]) for k in kernels) if kernels else 0
    gpu_union = union_ns(intervals_ns(kernels))
    return dict(gpu_kernel_count=len(kernels),
                gpu_kernel_sum_us=sum(float(k["duration_ms"])*1000 for k in kernels),
                gpu_kernel_union_us=gpu_union/1000,
                gpu_kernel_span_us=gpu_span/1000,
                gpu_kernel_gap_within_span_us=(gpu_span-gpu_union)/1000,
                launch_api_count=len(launch),
                launch_api_sum_us=sum(float(a["duration_ms"])*1000 for a in launch),
                launch_api_union_us=union_ns(intervals_ns(launch))/1000,
                sync_api_count=len(sync),
                sync_api_sum_us=sum(float(a["duration_ms"])*1000 for a in sync),
                sync_api_union_us=union_ns(intervals_ns(sync))/1000,
                cuda_api_union_us=union_ns(intervals_ns(apis))/1000,
                explicit_memcpy_count=len(copies),
                explicit_memcpy_sum_us=sum(float(c["duration_ms"])*1000 for c in copies),
                explicit_memcpy_bytes=sum(int(c["bytes"]) for c in copies))


def describe(values):
    a = np.array(values, dtype=np.float64)
    return dict(count=len(a), mean=float(a.mean()), std=float(a.std(ddof=1)) if len(a)>1 else 0.,
                median=float(np.median(a)), p95=float(np.percentile(a, 95)),
                min=float(a.min()), max=float(a.max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    update_rows, phase_rows, step_rows, metadata_rows, launch_rows, trace_rows = [], [], [], [], [], []
    for directory in sorted(args.analysis_root.glob("trace-*")):
        if not directory.is_dir():
            continue
        trace = directory.name
        _, workload, method = trace.split("-", 2)
        summary = json.loads((directory / "profiling-summary.json").read_text())
        ranges = read_csv(directory / "nvtx-events.csv")
        kernels = read_csv(directory / "kernel-events.csv")
        copies = read_csv(directory / "memcpy-events.csv")
        apis = read_csv(directory / "cuda-api-events.csv")
        api_map = {api["api_event_id"]: api for api in apis}
        for event in kernels+copies+apis:
            event["ids"] = set(json.loads(event["nvtx_ids"]))

        def in_range(event, nvtx):
            if int(nvtx["sqlite_rowid"]) in event["ids"]:
                return True
            # Match the old parser's explicit worker-thread submission rule;
            # this uses the API timestamp, never the GPU event timestamp.
            if event["phase_rule"].startswith("unique_cross_thread_phase_window"):
                api = event if "api_table" in event else api_map[event["api_event_id"]]
                return int(nvtx["start"]) <= int(api["start"]) < int(nvtx["end"]) and nvtx["name"] == event["phase"]
            return False

        by_phase = defaultdict(list)
        for nvtx in ranges:
            if nvtx["name"] in ("data", "transfer", "forward", "backward", "update"):
                by_phase[nvtx["name"]].append(nvtx)
        per_step = defaultdict(list)
        for phase, entries in by_phase.items():
            for index, nvtx in enumerate(sorted(entries, key=lambda r:int(r["start"])), 1):
                kk = [k for k in kernels if in_range(k, nvtx)]
                cc = [c for c in copies if in_range(c, nvtx)]
                aa = [a for a in apis if in_range(a, nvtx)]
                row = dict(trace=trace, workload=workload, method=method, phase=phase, index=index,
                           nvtx_rowid=nvtx["sqlite_rowid"], start_ns=nvtx["start"], end_ns=nvtx["end"],
                           host_nvtx_us=float(nvtx["duration_ms"])*1000, **event_metrics(kk, cc, aa))
                phase_rows.append(row)
                if workload == "training":
                    per_step[index].append((row, kk, cc, aa))
                if phase == "update":
                    mode, repeat, step = "training", 1, index
                    if workload == "update":
                        inner = [r for r in ranges if r["name"].startswith("SGD/update|")
                                 and r["globalTid_label"] == nvtx["globalTid_label"]
                                 and int(nvtx["start"]) <= int(r["start"]) < int(nvtx["end"])]
                        assert len(inner) == 1, (trace, nvtx["sqlite_rowid"], len(inner))
                        fields = dict(field.split("=", 1) for field in inner[0]["name"].split("|")[1:])
                        mode, repeat, step = fields["mode"], int(fields["repeat"]), int(fields["step"])
                    update_rows.append(dict(row, mode=mode, repeat=repeat, step=step))
            selected = [row for row in phase_rows if row["trace"] == trace and row["phase"] == phase]
            kernel_count = sum(row["gpu_kernel_count"] for row in selected)
            launch_rows.append(dict(trace=trace, workload=workload, method=method, phase=phase,
                                    nvtx_count=len(selected), gpu_kernel_count=kernel_count,
                                    kernels_per_nvtx=kernel_count/len(selected),
                                    launch_api_count=sum(row["launch_api_count"] for row in selected),
                                    gpu_kernel_sum_us=sum(row["gpu_kernel_sum_us"] for row in selected),
                                    gpu_memcpy_count=sum(row["explicit_memcpy_count"] for row in selected),
                                    gpu_memcpy_bytes=sum(row["explicit_memcpy_bytes"] for row in selected)))
        for step, records in sorted(per_step.items()):
            assert len(records) == 5, (trace, step, len(records))
            kk = [k for _, keys, _, _ in records for k in keys]
            cc = [c for _, _, copy, _ in records for c in copy]
            aa = [a for _, _, _, api in records for a in api]
            lower = min(int(row["start_ns"]) for row, _, _, _ in records)
            upper = max(int(row["end_ns"]) for row, _, _, _ in records)
            step_rows.append(dict(trace=trace, method=method, step=step,
                                  phase_envelope_us=(upper-lower)/1000,
                                  host_phase_sum_us=sum(row["host_nvtx_us"] for row, _, _, _ in records),
                                  **event_metrics(kk, cc, aa)))
        for nvtx in ranges:
            if nvtx["name"] in ("SGD/metadata_prepare", "SGD/native_submit", "SGD/ctypes_submit", "SGD/native_runtime") and nvtx["phase"] == "update":
                enclosing = [r for r in update_rows if r["trace"] == trace
                             and int(r["start_ns"]) <= int(nvtx["start"]) < int(r["end_ns"])]
                assert len(enclosing) == 1
                parent = enclosing[0]
                metadata_rows.append(dict(trace=trace, workload=workload, method=method,
                                          mode=parent["mode"], repeat=parent["repeat"], step=parent["step"],
                                          range=nvtx["name"], nvtx_rowid=nvtx["sqlite_rowid"],
                                          start_ns=nvtx["start"], end_ns=nvtx["end"],
                                          host_inclusive_us=float(nvtx["duration_ms"])*1000,
                                          host_self_us=float(nvtx["host_self_ms"])*1000))
        trace_rows.append(dict(trace=trace, workload=workload, method=method,
                               whole_trace_kernel_count=summary["whole_trace_kernel_count"],
                               measured_phase_kernel_count=summary["training_phase_submitted_kernel_count"],
                               unmatched_kernel_api_count=summary["unmatched_kernel_api_count"],
                               cross_thread_phase_kernel_count=summary["kernel_cross_thread_phase_attributions"],
                               measured_envelope_ms=summary["training_envelope_ms"],
                               kernel_memcpy_interval_coverage_fraction=summary["activity_coverage_fraction"]))
    assert len(trace_rows) == 6, [row["trace"] for row in trace_rows]
    assert all(row["unmatched_kernel_api_count"] == 0 for row in trace_rows)
    aggregate = []
    groups = defaultdict(list)
    for row in update_rows:
        groups[(row["trace"], row["workload"], row["method"], row["mode"])].append(row)
    metrics = ["host_nvtx_us", "gpu_kernel_count", "gpu_kernel_sum_us", "gpu_kernel_union_us",
               "gpu_kernel_span_us", "gpu_kernel_gap_within_span_us", "launch_api_count",
               "launch_api_sum_us", "launch_api_union_us", "sync_api_count", "sync_api_sum_us",
               "sync_api_union_us", "cuda_api_union_us", "explicit_memcpy_count", "explicit_memcpy_sum_us", "explicit_memcpy_bytes"]
    for (trace, workload, method, mode), rows in groups.items():
        for metric in metrics:
            aggregate.append(dict(trace=trace, workload=workload, method=method, mode=mode,
                                  metric=metric, **describe([row[metric] for row in rows])))
    metadata_aggregate = []
    groups = defaultdict(list)
    for row in metadata_rows:
        groups[(row["trace"], row["workload"], row["method"], row["mode"], row["range"])].append(row)
    for (trace, workload, method, mode, label), rows in groups.items():
        for metric in ("host_inclusive_us", "host_self_us"):
            metadata_aggregate.append(dict(trace=trace, workload=workload, method=method, mode=mode,
                                           range=label, metric=metric, **describe([row[metric] for row in rows])))
    for name, rows in (("trace-update-raw", update_rows), ("trace-update-summary", aggregate),
                       ("trace-phase-raw", phase_rows), ("trace-training-step-raw", step_rows),
                       ("metadata-host-raw", metadata_rows), ("metadata-host-summary", metadata_aggregate),
                       ("launch-counts", launch_rows)):
        write_csv(args.output/(name+".csv"), rows)
    report = dict(status="OFFLINE_ANALYSIS_COMPLETE", traces=trace_rows,
                  update_ranges=len(update_rows), training_steps=len(step_rows),
                  attribution="GPU correlation ID -> submitting CUDA API -> enclosing same-thread NVTX; explicitly flagged PyTorch worker API timestamps may select a unique phase window. Never match GPU time overlap.",
                  units="All *_us fields are microseconds. std is sample standard deviation; p95 uses NumPy percentile.",
                  scopes="Whole-trace counts include warmup/setup; measured phase counts exclude warmup/setup. Update averages separately group sync_wall, host_submission and training.",
                  timing_caveats=["Host NVTX time and CUDA API time are not GPU kernel execution time.",
                                  "Nested native_runtime/native_submit/ctypes_submit inclusive times overlap and must not be added.",
                                  "Zero explicit CUDA memcpy events does not make descriptor transport free: CUDA launch argument transport is included in submission and cannot be separated here.",
                                  "Kernel+Memcpy envelope coverage is not GPU/SM utilization.",
                                  "A gap within a submitted update's kernel span may include unrelated GPU work; it is not proof of whole-device idle time.",
                                  "Nsight figures are instrumented observations and must not be used as uninstrumented throughput."])
    (args.output/"trace-summary.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
