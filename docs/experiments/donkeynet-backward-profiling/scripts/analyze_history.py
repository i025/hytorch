"""Read-only Nsight Systems attribution, preserving each raw event and its join.

GPU work is attributed through CUDA correlation IDs to the NVTX range that
contains the submitting API, not through temporal overlap with asynchronous
GPU execution. Times in SQLite are nanoseconds; exported totals are milliseconds.
This is an instrumented trace analysis, not a training throughput benchmark.
"""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sqlite3


def union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return merged


def duration(intervals):
    return sum(end - start for start, end in union(intervals))


def clipped(intervals, lo, hi):
    return [(max(start, lo), min(end, hi)) for start, end in intervals
            if start < hi and end > lo]


def write_csv(path, rows):
    if not rows:
        return
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def public_rows(rows, thread_ids, process_ids):
    """Preserve attribution IDs while replacing host process/thread identifiers."""
    result = []
    for source in rows:
        row = dict(source)
        for key in ("globalTid", "endGlobalTid"):
            if key in row:
                value = row.pop(key)
                row[key + "_label"] = thread_ids.setdefault(value, "thread_" + str(len(thread_ids))) if value is not None else ""
        if "globalPid" in row:
            value = row.pop("globalPid")
            row["globalPid_label"] = process_ids.setdefault(value, "process_" + str(len(process_ids))) if value is not None else ""
        result.append(row)
    return result


def family(name):
    if name == "col2im":
        return "CuPy col2im"
    if name == "im2col":
        return "CuPy im2col"
    if name in ("im2col_kernel", "conv_pack", "gemm_shared"):
        return "HyTorch CUDA " + name
    if name.startswith("cupy_copy"):
        return "CuPy copy/layout"
    if name in ("cupy_sum", "DeviceReduceSingleTileKernel"):
        return "CuPy/CUB reduction"
    if name.startswith(("ampere_", "gemm", "gemv", "splitK")):
        return "cuBLAS GEMM/GEMV/reduction"
    return "CuPy elementwise"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    connection = sqlite3.connect("file:" + str(args.input.resolve()) + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    tables = [row[0] for row in connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")]
    names = dict(connection.execute("SELECT id,value FROM StringIds"))

    def rows(table):
        return [dict(row) for row in connection.execute("SELECT rowid AS sqlite_rowid,* FROM " + table + " ORDER BY start")]

    ranges = [row for row in rows("NVTX_EVENTS") if row["end"] is not None]
    for row in ranges:
        row["name"] = row["text"] or names.get(row["textId"], "")
        row["duration_ms"] = (row["end"] - row["start"]) / 1e6
    lo = min(row["start"] for row in ranges)
    hi = max(row["end"] for row in ranges)

    def enclosing(api):
        matches = [r for r in ranges if r["globalTid"] == api["globalTid"]
                   and r["start"] <= api["start"] < r["end"]]
        return min(matches, key=lambda r: r["end"] - r["start"]) if matches else None

    apis = rows("CUPTI_ACTIVITY_KIND_RUNTIME")
    by_correlation = {}
    for api in apis:
        api["name"] = names[api["nameId"]]
        api["duration_ms"] = (api["end"] - api["start"]) / 1e6
        active = enclosing(api)
        api["phase"] = active["name"] if active else "outside_nvtx"
        api["nvtx_rowid"] = active["sqlite_rowid"] if active else ""
        api["relative_start_ms"] = (api["start"] - lo) / 1e6
        assert api["correlationId"] not in by_correlation
        by_correlation[api["correlationId"]] = api

    kernels = rows("CUPTI_ACTIVITY_KIND_KERNEL")
    copies = rows("CUPTI_ACTIVITY_KIND_MEMCPY")
    copy_names = {r[0]: r[2] for r in connection.execute("SELECT * FROM ENUM_CUDA_MEMCPY_OPER")}
    for kind, activity in (("kernel", kernels), ("memcpy", copies)):
        for event in activity:
            event["kind"] = kind
            event["name"] = (names[event["shortName"]] if kind == "kernel" else copy_names[event["copyKind"]])
            if kind == "kernel":
                event["demangled_name"] = names[event["demangledName"]]
                event["family"] = family(event["name"])
            event["duration_ms"] = (event["end"] - event["start"]) / 1e6
            event["relative_start_ms"] = (event["start"] - lo) / 1e6
            event["overlaps_training_envelope"] = int(event["start"] < hi and event["end"] > lo)
            api = by_correlation.get(event["correlationId"])
            event["api_name"] = api["name"] if api else "unmatched"
            event["api_rowid"] = api["sqlite_rowid"] if api else ""
            event["phase"] = api["phase"] if api else "unmatched"
            event["nvtx_rowid"] = api["nvtx_rowid"] if api else ""
            event["api_start_to_gpu_start_ms"] = (event["start"] - api["start"]) / 1e6 if api else ""
            event["api_end_to_gpu_start_ms"] = (event["start"] - api["end"]) / 1e6 if api else ""
            active = enclosing(api) if api else None
            event["gpu_exec_fully_within_submitting_nvtx"] = int(bool(active) and event["start"] >= active["start"] and event["end"] <= active["end"])

    # Preserve timestamps/correlation values; raw PID/TID stay in server SQLite.
    thread_ids, process_ids = {}, {}
    for name, data in (("nvtx-events", ranges), ("cuda-api-events", apis),
                       ("kernel-events", kernels), ("memcpy-events", copies)):
        write_csv(args.output / (name + ".csv"), public_rows(data, thread_ids, process_ids))
    if "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION" in tables:
        write_csv(args.output / "synchronization-events.csv", public_rows(rows("CUPTI_ACTIVITY_KIND_SYNCHRONIZATION"), thread_ids, process_ids))
    schema = {table: [dict(row) for row in connection.execute("PRAGMA table_info(" + table + ")")]
              for table in ("NVTX_EVENTS", "CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION")}
    (args.output / "schema.json").write_text(json.dumps(schema, indent=2) + "\n")

    phase_rows = []
    for phase in ["data", "transfer", "forward", "backward", "update", "outside_nvtx"]:
        rs = [r for r in ranges if r["name"] == phase]
        aa = [a for a in apis if a["phase"] == phase]
        kk = [k for k in kernels if k["phase"] == phase]
        cc = [c for c in copies if c["phase"] == phase]
        phase_rows.append(dict(
            phase=phase, nvtx_count=len(rs), host_nvtx_sum_ms=sum(r["duration_ms"] for r in rs),
            cuda_api_count=len(aa), cuda_api_sum_ms=sum(a["duration_ms"] for a in aa),
            cuda_api_union_ms=duration([(a["start"], a["end"]) for a in aa]) / 1e6,
            launch_api_count=sum("LaunchKernel" in a["name"] for a in aa),
            launch_api_ms=sum(a["duration_ms"] for a in aa if "LaunchKernel" in a["name"]),
            sync_api_count=sum("Synchronize" in a["name"] for a in aa),
            sync_api_ms=sum(a["duration_ms"] for a in aa if "Synchronize" in a["name"]),
            allocation_api_count=sum("Malloc" in a["name"] or "HostAlloc" in a["name"] or "Free" in a["name"] for a in aa),
            allocation_api_ms=sum(a["duration_ms"] for a in aa if "Malloc" in a["name"] or "HostAlloc" in a["name"] or "Free" in a["name"]),
            gpu_kernel_count=len(kk), gpu_kernel_sum_ms=sum(k["duration_ms"] for k in kk),
            gpu_kernel_mean_us=sum(k["duration_ms"] for k in kk) * 1000 / len(kk) if kk else 0,
            gpu_kernel_under_10us_count=sum(k["duration_ms"] < .01 for k in kk),
            gpu_kernels_outside_submitting_nvtx=sum(not k["gpu_exec_fully_within_submitting_nvtx"] for k in kk),
            gpu_memcpy_count=len(cc), gpu_memcpy_sum_ms=sum(c["duration_ms"] for c in cc),
            gpu_memcpy_bytes=sum(c["bytes"] for c in cc)))
    write_csv(args.output / "phase-summary.csv", phase_rows)

    kernel_summary = []
    api_summary = []
    copy_summary = []
    for scope in ("whole_trace", "training_envelope", "submitted_in_training_nvtx"):
        def included(event):
            if scope == "whole_trace":
                return True
            if scope == "training_envelope":
                return event["start"] < hi and event["end"] > lo
            return event["phase"] not in ("outside_nvtx", "unmatched")

        for events, destination, group_key in ((kernels, kernel_summary, "name"), (apis, api_summary, "name"), (copies, copy_summary, "name")):
            grouped = defaultdict(list)
            for event in events:
                if included(event):
                    grouped[(event["phase"], event[group_key])].append(event)
            for (phase, name), values in sorted(grouped.items(), key=lambda kv: -sum(e["duration_ms"] for e in kv[1])):
                total = sum(e["duration_ms"] for e in values)
                record = dict(scope=scope, phase=phase, name=name, count=len(values), total_ms=total,
                              mean_us=total * 1000 / len(values), max_us=max(e["duration_ms"] for e in values) * 1000)
                if events is kernels:
                    record["family"] = family(name)
                if events is copies:
                    record["bytes"] = sum(e["bytes"] for e in values)
                destination.append(record)
    write_csv(args.output / "kernel-summary.csv", kernel_summary)
    write_csv(args.output / "api-summary.csv", api_summary)
    write_csv(args.output / "memcpy-summary.csv", copy_summary)

    intervals = [(e["start"], e["end"]) for e in kernels + copies]
    active_intervals = union(clipped(intervals, lo, hi))
    gaps = []
    position = lo
    api_intervals = [(a["start"], a["end"]) for a in apis]
    launch_intervals = [(a["start"], a["end"]) for a in apis if "LaunchKernel" in a["name"]]
    sync_intervals = [(a["start"], a["end"]) for a in apis if "Synchronize" in a["name"]]
    for start, end in active_intervals + [[hi, hi]]:
        if start > position:
            gaps.append(dict(start_ns=position, end_ns=start, relative_start_ms=(position-lo)/1e6,
                             duration_ms=(start-position)/1e6,
                             cuda_api_overlap_ms=duration(clipped(api_intervals, position, start))/1e6,
                             launch_api_overlap_ms=duration(clipped(launch_intervals, position, start))/1e6,
                             sync_api_overlap_ms=duration(clipped(sync_intervals, position, start))/1e6))
        position = end
    write_csv(args.output / "gpu-activity-gaps.csv", gaps)
    api_union = duration(clipped(api_intervals, lo, hi))
    summary = dict(
        input=str(args.input.resolve()), attribution="correlationId -> submitting CUDA API -> same-thread enclosing NVTX range; GPU timestamps do not determine phase",
        privacy="Published process/thread IDs replaced by stable labels; original identifiers remain in the read-only SQLite on server.",
        scope="Instrumented historical combined HyTorch DonkeyNet, 16 measured steps plus 2 warmup steps; not non-instrumented throughput",
        training_nvtx_envelope_start_ns=lo, training_nvtx_envelope_end_ns=hi,
        training_nvtx_envelope_ms=(hi-lo)/1e6,
        activity_covered_ms=duration(active_intervals)/1e6,
        activity_coverage_fraction=duration(active_intervals)/(hi-lo),
        coverage_definition="Union of measured CUDA kernel and memcpy intervals clipped to first-to-last training NVTX envelope. NOT GPU/SM utilization; profiled process only.",
        whole_trace_kernel_count=len(kernels), training_envelope_kernel_count=sum(k["overlaps_training_envelope"] for k in kernels),
        submitted_in_training_nvtx_kernel_count=sum(k["phase"] != "outside_nvtx" for k in kernels),
        whole_trace_kernel_duration_sum_ms=sum(k["duration_ms"] for k in kernels),
        unmatched_kernel_api_count=sum(k["phase"] == "unmatched" for k in kernels),
        whole_trace_kernel_names_with_fp64=[n for n in sorted({k["name"] for k in kernels}) if any(t in n.lower() for t in ("float64", "f64", "d884"))],
        cuda_api_union_within_envelope_ms=api_union/1e6,
        envelope_outside_recorded_cuda_api_ms=((hi-lo)-api_union)/1e6,
        gpu_activity_gap_sum_ms=sum(g["duration_ms"] for g in gaps),
        gpu_activity_gap_cuda_api_overlap_ms=sum(g["cuda_api_overlap_ms"] for g in gaps),
        gpu_activity_gap_launch_api_overlap_ms=sum(g["launch_api_overlap_ms"] for g in gaps),
        gpu_activity_gap_sync_api_overlap_ms=sum(g["sync_api_overlap_ms"] for g in gaps),
        caveat="CPU range/API duration is host elapsed time; API may submit, block, or include profiler overhead. Gaps outside APIs do not prove Python CPU computation. No CPU samples/context-switch trace, achieved occupancy or SM counters were captured.",
        phase_summary=phase_rows)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
