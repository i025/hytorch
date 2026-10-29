"""Attribute Nsight GPU events through CUDA API correlations and nested NVTX.

Reads SQLite in read-only mode. It does not import CUDA, run a GPU experiment,
or calculate hashes. Exclusive operator attribution assigns each event once;
inclusive attribution may count an event in several ancestors and must not be
summed across those ancestors. Raw timestamps and attribution are exported.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import sqlite3

from analyze_history import clipped, duration, public_rows, union, write_csv


PHASES = {"data", "transfer", "forward", "backward", "update"}


def phase_name(label):
    name = label[6:] if label.startswith("PHASE/") else label
    return name if name in PHASES else None


def operator_name(label):
    # PyTorch emits execution sequence numbers as part of names, not op identity.
    return re.sub(r",?\s*(?:stashed )?seq\s*=\s*\d+", "", label).strip()


def api_kind(name):
    if "LaunchKernel" in name:
        return "launch"
    if "Synchronize" in name:
        return "synchronization"
    if any(word in name for word in ("Malloc", "Alloc", "Free")):
        return "allocation_or_free"
    if "Memcpy" in name or "Memset" in name:
        return "copy_or_memset_api"
    if "Event" in name:
        return "event_api"
    return "other"


def ranges_for_api(api, ranges):
    """Return same-thread NVTX ancestors ordered outermost to innermost."""
    return sorted((r for r in ranges if r["globalTid"] == api["globalTid"]
                   and r["start"] <= api["start"] < r["end"]),
                  key=lambda r: (-(r["end"] - r["start"]), r["sqlite_rowid"]))


def op_ancestors(stack):
    # Wrapper domains other than a known phase remain visible; no op is invented.
    return [r for r in stack if phase_name(r["name"]) is None]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True, help="e.g. combined-profile-01; preserved in all summaries")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    db = sqlite3.connect("file:" + str(args.input.resolve()) + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    names = dict(db.execute("SELECT id,value FROM StringIds"))

    def read(table):
        return [dict(r) for r in db.execute("SELECT rowid AS sqlite_rowid,* FROM " + table + " ORDER BY start")]

    ranges = [r for r in read("NVTX_EVENTS") if r["end"] is not None]
    for r in ranges:
        r["name"] = r.get("text") or names.get(r.get("textId"), "")
        r["duration_ms"] = (r["end"] - r["start"]) / 1e6
        r["operator"] = operator_name(r["name"])
        children = [(child["start"], child["end"]) for child in ranges
                    if child["globalTid"] == r["globalTid"]
                    and child["sqlite_rowid"] != r["sqlite_rowid"]
                    and r["start"] <= child["start"] and child["end"] <= r["end"]
                    and (r["start"] < child["start"] or child["end"] < r["end"])]
        r["host_self_ms"] = r["duration_ms"] - duration(children) / 1e6
    phase_ranges = [r for r in ranges if phase_name(r["name"])]
    if not phase_ranges:
        raise ValueError("No data/transfer/forward/backward/update phase NVTX ranges found; cannot define training envelope")
    lo, hi = min(r["start"] for r in phase_ranges), max(r["end"] for r in phase_ranges)
    range_by_id = {r["sqlite_rowid"]: r for r in ranges}
    range_phase = {}
    for r in ranges:
        candidates = [p for p in phase_ranges if p["start"] <= r["start"] < p["end"]
                      and p["globalTid"] == r["globalTid"]]
        r["phase"] = phase_name(min(candidates, key=lambda p: p["end"]-p["start"])["name"]) if candidates else "outside_phase"
        r["phase_rule"] = "same_thread_nvtx" if candidates else "no_enclosing_phase"
        if not candidates:
            temporal = {phase_name(p["name"]) for p in phase_ranges if p["start"] <= r["start"] < p["end"]}
            if len(temporal) == 1:
                r["phase"] = next(iter(temporal))
                r["phase_rule"] = "unique_cross_thread_phase_window_at_nvtx_start"
        range_phase[r["sqlite_rowid"]] = r["phase"]

    apis = []
    for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
        if table in tables:
            for api in read(table):
                api["api_table"] = table
                api["api_event_id"] = table + ":" + str(api["sqlite_rowid"])
                api["name"] = names[api["nameId"]]
                api["kind"] = api_kind(api["name"])
                api["duration_ms"] = (api["end"] - api["start"]) / 1e6
                api["relative_start_ms"] = (api["start"] - lo) / 1e6
                stack = ranges_for_api(api, ranges)
                phases = [r for r in stack if phase_name(r["name"])]
                api["phase"] = phase_name(phases[-1]["name"]) if phases else "outside_phase"
                api["phase_rule"] = "same_thread_nvtx" if phases else "no_enclosing_phase"
                # PyTorch autograd can submit on a worker thread while the main
                # thread waits inside backward. Expose this distinct rule.
                if not phases:
                    temporal = [r for r in phase_ranges if r["start"] <= api["start"] < r["end"]]
                    candidates = {phase_name(r["name"]) for r in temporal}
                    if len(candidates) == 1:
                        api["phase"] = next(iter(candidates))
                        api["phase_rule"] = "unique_cross_thread_phase_window_at_api_submission"
                ops = op_ancestors(stack)
                api["exclusive_operator"] = ops[-1]["operator"] if ops else "unlabeled:" + api["phase"]
                api["nvtx_ids"] = json.dumps([r["sqlite_rowid"] for r in stack])
                api["operator_ids"] = json.dumps([r["sqlite_rowid"] for r in ops])
                api["operator_path"] = json.dumps([r["name"] for r in ops])
                apis.append(api)
    apis.sort(key=lambda a: a["start"])
    api_candidates = defaultdict(list)
    for api in apis:
        api_candidates[api["correlationId"]].append(api)
    # If a Runtime API wraps a Driver API with the same correlation, choose the
    # inner API while retaining both original rows and a candidate-count field.
    api_by_correlation = {}
    for key, values in api_candidates.items():
        innermost = min(values, key=lambda a: a["end"]-a["start"])
        if any(a["globalTid"] != innermost["globalTid"] or a["start"] > innermost["start"] or a["end"] < innermost["end"] for a in values):
            raise ValueError("Non-nested or cross-thread duplicate CUDA correlation ID: " + str(key))
        api_by_correlation[key] = innermost

    kernels = read("CUPTI_ACTIVITY_KIND_KERNEL")
    copies = read("CUPTI_ACTIVITY_KIND_MEMCPY") if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables else []
    memsets = read("CUPTI_ACTIVITY_KIND_MEMSET") if "CUPTI_ACTIVITY_KIND_MEMSET" in tables else []
    copy_names = {r[0]: r[2] for r in db.execute("SELECT * FROM ENUM_CUDA_MEMCPY_OPER")}
    for kind, events in (("kernel", kernels), ("memcpy", copies), ("memset", memsets)):
        for event in events:
            event["event_id"] = kind + ":" + str(event["sqlite_rowid"])
            event["kind"] = kind
            event["name"] = names[event["shortName"]] if kind == "kernel" else copy_names[event["copyKind"]] if kind == "memcpy" else "GPU Memset"
            if kind == "kernel":
                event["demangled_name"] = names[event["demangledName"]]
            event["duration_ms"] = (event["end"] - event["start"]) / 1e6
            event["relative_start_ms"] = (event["start"] - lo) / 1e6
            event["overlaps_training_envelope"] = int(event["start"] < hi and event["end"] > lo)
            api = api_by_correlation.get(event["correlationId"])
            event["api_candidate_count"] = len(api_candidates[event["correlationId"]])
            event["api_event_id"] = api["api_event_id"] if api else "unmatched"
            event["api_name"] = api["name"] if api else "unmatched"
            event["phase"] = api["phase"] if api else "unmatched"
            event["phase_rule"] = api["phase_rule"] if api else "unmatched"
            event["exclusive_operator"] = api["exclusive_operator"] if api else "unmatched"
            event["nvtx_ids"] = api["nvtx_ids"] if api else "[]"
            event["operator_ids"] = api["operator_ids"] if api else "[]"
            event["operator_path"] = api["operator_path"] if api else "[]"
            event["api_start_to_gpu_start_ms"] = (event["start"]-api["start"])/1e6 if api else ""
            event["api_end_to_gpu_start_ms"] = (event["start"]-api["end"])/1e6 if api else ""

    thread_ids, process_ids = {}, {}
    for basename, events in (("nvtx-events", ranges), ("cuda-api-events", apis), ("kernel-events", kernels),
                             ("memcpy-events", copies), ("memset-events", memsets)):
        write_csv(args.output/(basename+".csv"), public_rows(events, thread_ids, process_ids))
    for table in ("CUPTI_ACTIVITY_KIND_SYNCHRONIZATION",):
        if table in tables:
            write_csv(args.output/(table.lower()+".csv"), public_rows(read(table), thread_ids, process_ids))
    selected_tables = [table for table in ("NVTX_EVENTS", "CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER", "CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET", "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION") if table in tables]
    (args.output/"schema.json").write_text(json.dumps({table:[dict(r) for r in db.execute("PRAGMA table_info("+table+")")] for table in selected_tables}, indent=2)+"\n")

    phase_rows = []
    for phase in ["data", "transfer", "forward", "backward", "update", "outside_phase", "unmatched"]:
        rs = [r for r in phase_ranges if phase_name(r["name"]) == phase]
        aa = [a for a in apis if a["phase"] == phase]
        kk = [k for k in kernels if k["phase"] == phase]
        cc = [c for c in copies if c["phase"] == phase]
        mm = [m for m in memsets if m["phase"] == phase]
        phase_rows.append(dict(label=args.label, phase=phase, nvtx_count=len(rs),
                               host_nvtx_sum_ms=sum(r["duration_ms"] for r in rs),
                               host_nvtx_union_ms=duration([(r["start"],r["end"]) for r in rs])/1e6,
                               cuda_api_count=len(aa), cuda_api_sum_ms=sum(a["duration_ms"] for a in aa),
                               cuda_api_union_ms=duration([(a["start"],a["end"]) for a in aa])/1e6,
                               launch_api_count=sum(a["kind"] == "launch" for a in aa),
                               launch_api_ms=sum(a["duration_ms"] for a in aa if a["kind"] == "launch"),
                               sync_api_count=sum(a["kind"] == "synchronization" for a in aa),
                               sync_api_ms=sum(a["duration_ms"] for a in aa if a["kind"] == "synchronization"),
                               allocation_api_count=sum(a["kind"] == "allocation_or_free" for a in aa),
                               allocation_api_ms=sum(a["duration_ms"] for a in aa if a["kind"] == "allocation_or_free"),
                               gpu_kernel_count=len(kk), gpu_kernel_sum_ms=sum(k["duration_ms"] for k in kk),
                               gpu_kernel_union_ms=duration([(k["start"],k["end"]) for k in kk])/1e6,
                               gpu_kernel_under_10us_count=sum(k["duration_ms"] < .01 for k in kk),
                               gpu_memcpy_count=len(cc), gpu_memcpy_sum_ms=sum(c["duration_ms"] for c in cc),
                               gpu_memcpy_bytes=sum(c["bytes"] for c in cc),
                               gpu_memset_count=len(mm), gpu_memset_sum_ms=sum(m["duration_ms"] for m in mm)))
    write_csv(args.output/"phase-summary.csv", phase_rows)

    # Each GPU event belongs to exactly one exclusive bucket. Inclusive buckets
    # expose ancestry and intentionally overlap, with per-bucket de-duplication.
    ops = {}
    def op_group(scope, phase, name):
        return ops.setdefault((scope,phase,name), {"range_ids":set(),"apis":{},"kernels":{},"copies":{},"memsets":{}})

    for r in ranges:
        if not phase_name(r["name"]):
            for scope in ("exclusive", "inclusive"):
                op_group(scope, range_phase[r["sqlite_rowid"]], r["operator"])["range_ids"].add(r["sqlite_rowid"])
    for collection, events in (("apis", apis), ("kernels", kernels), ("copies", copies), ("memsets", memsets)):
        for event in events:
            event_id = event.get("event_id", event.get("api_event_id"))
            op_group("exclusive",event["phase"],event["exclusive_operator"])[collection][event_id] = event
            ids = json.loads(event["operator_ids"])
            for rid in ids:
                op_group("inclusive",event["phase"],range_by_id[rid]["operator"])[collection][event_id] = event
    operator_rows = []
    for (scope,phase,name), values in ops.items():
        rs = [range_by_id[rid] for rid in values["range_ids"]]
        aa,kk,cc,mm = [list(values[key].values()) for key in ("apis","kernels","copies","memsets")]
        total = sum(k["duration_ms"] for k in kk)
        operator_rows.append(dict(label=args.label,attribution=scope,phase=phase,operator=name,
                                  operator_nvtx_call_count=len(rs), host_nvtx_inclusive_ms=sum(r["duration_ms"] for r in rs),
                                  host_nvtx_exclusive_ms=sum(r["host_self_ms"] for r in rs),
                                  cuda_api_count=len(aa),cuda_api_sum_ms=sum(a["duration_ms"] for a in aa),
                                  cuda_api_union_ms=duration([(a["start"],a["end"]) for a in aa])/1e6,
                                  launch_api_count=sum(a["kind"] == "launch" for a in aa),
                                  launch_api_ms=sum(a["duration_ms"] for a in aa if a["kind"] == "launch"),
                                  sync_api_count=sum(a["kind"] == "synchronization" for a in aa),
                                  sync_api_ms=sum(a["duration_ms"] for a in aa if a["kind"] == "synchronization"),
                                  allocation_api_count=sum(a["kind"] == "allocation_or_free" for a in aa),
                                  allocation_api_ms=sum(a["duration_ms"] for a in aa if a["kind"] == "allocation_or_free"),
                                  gpu_kernel_count=len(kk),gpu_kernel_sum_ms=total,
                                  gpu_kernel_mean_us=total*1000/len(kk) if kk else 0,
                                  gpu_kernel_union_ms=duration([(k["start"],k["end"]) for k in kk])/1e6,
                                  gpu_kernel_under_10us_count=sum(k["duration_ms"] < .01 for k in kk),
                                  gpu_memcpy_count=len(cc),gpu_memcpy_sum_ms=sum(c["duration_ms"] for c in cc),
                                  gpu_memcpy_bytes=sum(c["bytes"] for c in cc),
                                  gpu_memset_count=len(mm),gpu_memset_sum_ms=sum(m["duration_ms"] for m in mm)))
    operator_rows.sort(key=lambda r:(r["attribution"],-r["gpu_kernel_sum_ms"]))
    write_csv(args.output/"operator-hotspots.csv",operator_rows)

    for basename, events, keys in (("kernel-launches",kernels,["phase","exclusive_operator","name"]),
                                   ("cuda-api-summary",apis,["phase","exclusive_operator","kind","name"]),
                                   ("memcpy-summary",copies,["phase","exclusive_operator","name"])):
        grouped = defaultdict(list)
        for event in events:
            grouped[tuple(event[k] for k in keys)].append(event)
        rows = []
        for key, values in grouped.items():
            total = sum(e["duration_ms"] for e in values)
            row = dict(label=args.label, **dict(zip(keys,key)))
            row.update(count=len(values),total_ms=total,mean_us=total*1000/len(values),max_us=max(e["duration_ms"] for e in values)*1000)
            if basename == "memcpy-summary":
                row["bytes"] = sum(e["bytes"] for e in values)
            rows.append(row)
        rows.sort(key=lambda r:-r["total_ms"])
        write_csv(args.output/(basename+".csv"), rows)

    active = union(clipped([(e["start"],e["end"]) for e in kernels+copies],lo,hi))
    gaps = []
    cursor = lo
    api_intervals = [(a["start"],a["end"]) for a in apis]
    launch_intervals = [(a["start"],a["end"]) for a in apis if a["kind"] == "launch"]
    for start,end in active+[[hi,hi]]:
        if start > cursor:
            gaps.append(dict(start_ns=cursor,end_ns=start,relative_start_ms=(cursor-lo)/1e6,
                             duration_ms=(start-cursor)/1e6,cuda_api_overlap_ms=duration(clipped(api_intervals,cursor,start))/1e6,
                             launch_api_overlap_ms=duration(clipped(launch_intervals,cursor,start))/1e6))
        cursor=end
    write_csv(args.output/"gpu-activity-gaps.csv",gaps)
    exclusive_count = sum(r["gpu_kernel_count"] for r in operator_rows if r["attribution"] == "exclusive")
    assert exclusive_count == len(kernels), (exclusive_count,len(kernels))
    summary = dict(label=args.label,input=str(args.input.resolve()),
                   training_envelope_start_ns=lo,training_envelope_end_ns=hi,training_envelope_ms=(hi-lo)/1e6,
                   whole_trace_kernel_count=len(kernels),training_envelope_kernel_count=sum(e["overlaps_training_envelope"] for e in kernels),
                   training_phase_submitted_kernel_count=sum(e["phase"] in PHASES for e in kernels),
                   activity_covered_ms=duration(active)/1e6,activity_coverage_fraction=duration(active)/(hi-lo),
                   coverage_definition="Kernel + Memcpy interval union clipped to training NVTX envelope; not GPU/SM utilization; excludes Memset for consistency with historical definition",
                   unmatched_kernel_api_count=sum(e["phase"] == "unmatched" for e in kernels),
                   kernels_with_multiple_api_candidates=sum(e["api_candidate_count"] > 1 for e in kernels),
                   kernel_cross_thread_phase_attributions=sum(e["phase_rule"] == "unique_cross_thread_phase_window_at_api_submission" for e in kernels),
                   exclusive_kernel_count_check=exclusive_count,
                   operator_nvtx_range_count=sum(phase_name(r["name"]) is None for r in ranges),
                   attribution="CUDA correlation -> submitting API -> same-thread nested NVTX; worker API phase may use unique main-thread phase window, explicitly tagged. Innermost non-phase NVTX gets exclusive attribution; all operator ancestors get inclusive attribution.",
                   timing_caveat="Host API/NVTX elapsed time is not GPU execution time. Inclusive operator totals overlap; only exclusive GPU attribution sums without duplicate events. Per-op host self time excludes nested same-thread NVTX intervals, not arbitrary Python or profiler overhead.",
                   api_count_caveat="Runtime and Driver API rows are both retained. Nested API wrappers can make API sums/counts exceed API interval union/unique GPU events; kernel count comes from CUPTI GPU activity rows.",
                   privacy="Published global process/thread identifiers replaced with stable labels. Original values remain in server SQLite.",
                   output_status="OFFLINE_ANALYSIS_COMPLETE")
    (args.output/"profiling-summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
