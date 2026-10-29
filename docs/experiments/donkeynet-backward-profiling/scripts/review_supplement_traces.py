"""Independent read-only SQLite audit of phase attribution and kernel overlap."""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sqlite3
import statistics


PHASES = {"data", "transfer", "forward", "backward", "update"}


def union_ns(events):
    intervals = sorted((row["start"], row["end"]) for row in events)
    total, end = 0, -1
    for start, stop in intervals:
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total


def review(raw, analysis, name):
    db = sqlite3.connect("file:" + str((raw / (name + ".sqlite")).resolve()) + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    names = {row[0]: row[1] for row in db.execute("SELECT id,value FROM StringIds")}
    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}

    def read(table):
        return [dict(row) for row in db.execute("SELECT rowid AS rowid,* FROM " + table + " ORDER BY start")]

    ranges = [row for row in read("NVTX_EVENTS") if row["end"] is not None]
    for row in ranges:
        row["name"] = row.get("text") or names.get(row.get("textId"), "")
    phases = [row for row in ranges if row["name"] in PHASES]
    starts = [row["start"] for row in phases if row["name"] == "data"]
    lo, hi = min(row["start"] for row in phases), max(row["end"] for row in phases)
    api_candidates = defaultdict(list)
    for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
        if table in tables:
            for row in read(table):
                row["id"] = table + ":" + str(row["rowid"])
                api_candidates[row["correlationId"]].append(row)
    apis = {key: min(rows, key=lambda r: r["end"] - r["start"]) for key, rows in api_candidates.items()}
    csv_kernels = {int(row["sqlite_rowid"]): row for row in csv.DictReader((analysis / name / "kernel-events.csv").open())}
    raw_kernels = read("CUPTI_ACTIVITY_KIND_KERNEL")
    kinds = {"kernel": raw_kernels}
    for kind, table in (("memcpy", "CUPTI_ACTIVITY_KIND_MEMCPY"), ("memset", "CUPTI_ACTIVITY_KIND_MEMSET")):
        kinds[kind] = read(table) if table in tables else []
    missing, mismatches = [], []
    cross = Counter()
    cross_api_crosses_phase_end = 0
    cross_kernel_worker_nvtx = 0
    cross_kernel_autograd_nvtx = 0
    training, perstep = [], defaultdict(list)
    outside_before, outside_after = 0, 0
    for kind, events in kinds.items():
        for event in events:
            api = apis.get(event["correlationId"])
            if api is None:
                missing.append(dict(kind=kind, rowid=event["rowid"]))
                continue
            same = [p for p in phases if p["globalTid"] == api["globalTid"] and p["start"] <= api["start"] < p["end"]]
            temporal = [p for p in phases if p["start"] <= api["start"] < p["end"]]
            assigned = min(same, key=lambda p: p["end"] - p["start"]) if same else temporal[0] if len(temporal) == 1 else None
            phase = assigned["name"] if assigned else "outside_phase"
            if assigned and not same:
                cross[(kind, phase)] += 1
                cross_api_crosses_phase_end += int(api["end"] > assigned["end"])
                if kind == "kernel":
                    own = [r for r in ranges if r["globalTid"] == api["globalTid"] and r["start"] <= api["start"] < r["end"]]
                    cross_kernel_worker_nvtx += bool(own)
                    cross_kernel_autograd_nvtx += any("Backward" in r["name"] or "autograd::engine" in r["name"] for r in own)
            if kind == "kernel":
                exported = csv_kernels[event["rowid"]]
                if exported["phase"] != phase or exported["api_event_id"] != api["id"]:
                    mismatches.append(dict(rowid=event["rowid"], expected_phase=phase, exported_phase=exported["phase"]))
                if assigned:
                    enriched = dict(event, phase=phase, step=bisect_right(starts, api["start"]), name=names[event["shortName"]])
                    training.append(enriched)
                    perstep[enriched["step"]].append(enriched)
                else:
                    outside_before += api["start"] < lo
                    outside_after += api["start"] >= hi
    rows = []
    for step in range(1, 17):
        events = perstep[step]
        row = dict(trace=name, step=step, kernel_count=len(events),
                   kernel_sum_ms=sum(e["end"] - e["start"] for e in events) / 1e6,
                   kernel_union_ms=union_ns(events) / 1e6,
                   streams=",".join(str(v) for v in sorted({e["streamId"] for e in events})))
        for phase in ("forward", "backward", "update"):
            selected = [e for e in events if e["phase"] == phase]
            row[phase + "_kernel_count"] = len(selected)
            row[phase + "_kernel_sum_ms"] = sum(e["end"] - e["start"] for e in selected) / 1e6
            row[phase + "_kernel_union_ms"] = union_ns(selected) / 1e6
        rows.append(row)
    window_rows = []
    for label, selected in (("full_16_steps", rows), ("first_step", rows[:1]), ("remaining_15_steps", rows[1:])):
        window_rows.append(dict(window=label, steps=len(selected),
                               kernel_count_per_step=sorted({row["kernel_count"] for row in selected}),
                               kernel_sum_mean_ms=statistics.mean(row["kernel_sum_ms"] for row in selected),
                               kernel_sum_min_ms=min(row["kernel_sum_ms"] for row in selected),
                               kernel_sum_max_ms=max(row["kernel_sum_ms"] for row in selected),
                               kernel_union_mean_ms=statistics.mean(row["kernel_union_ms"] for row in selected),
                               kernel_union_min_ms=min(row["kernel_union_ms"] for row in selected),
                               kernel_union_max_ms=max(row["kernel_union_ms"] for row in selected)))
    submitted_before_but_gpu_overlaps = sum(e["start"] < hi and e["end"] > lo and apis[e["correlationId"]]["start"] < lo for e in raw_kernels)
    update_names = Counter(e["name"] for e in training if e["phase"] == "update")
    summary = dict(trace=name, phase_counts=dict(Counter(p["name"] for p in phases)),
                   full_trace_kernels=len(raw_kernels), training_kernels=len(training),
                   outside_phase_kernels_submitted_before_training=outside_before,
                   outside_phase_kernels_submitted_after_training=outside_after,
                   warmup_kernel_overlap_with_training_envelope=submitted_before_but_gpu_overlaps,
                   unmatched_events=missing, parser_kernel_phase_or_api_mismatches=mismatches,
                   kernels_with_multiple_correlated_apis=sum(len(api_candidates[e["correlationId"]]) != 1 for e in raw_kernels),
                   cross_thread_event_phases=[dict(kind=kind, phase=phase, count=count) for (kind, phase), count in sorted(cross.items())],
                   cross_thread_api_crosses_phase_end=cross_api_crosses_phase_end,
                   cross_thread_kernels_with_own_worker_nvtx=cross_kernel_worker_nvtx,
                   cross_thread_kernels_with_own_backward_autograd_nvtx=cross_kernel_autograd_nvtx,
                   training_stream_ids=sorted({e["streamId"] for e in training}),
                   training_context_count=len({e["contextId"] for e in training}),
                   training_device_count=len({e["deviceId"] for e in training}),
                   training_kernel_sum_ms=sum(e["end"] - e["start"] for e in training) / 1e6,
                   training_kernel_union_ms=union_ns(training) / 1e6,
                   windows=window_rows, update_kernel_names=dict(update_names),
                   limitations="Same-process unique main-thread phase window identifies worker submission phase; detailed worker NVTX independently corroborates backward. GPU execution timestamps are never used to pick phase. Kernel union measures traced execution intervals, not SM utilization or achieved FLOP efficiency.")
    db.close()
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summaries, rows = [], []
    for detail in ("phase", "detail"):
        for method in ("cupy", "combined", "torch"):
            summary, step_rows = review(args.raw, args.analysis, "trace-" + detail + "-" + method)
            summaries.append(summary)
            rows.extend(step_rows)
    passed = all(not row["unmatched_events"] and not row["parser_kernel_phase_or_api_mismatches"]
                 and row["warmup_kernel_overlap_with_training_envelope"] == 0
                 and row["cross_thread_api_crosses_phase_end"] == 0 for row in summaries)
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "per-step-kernel-review.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = dict(status="PASSED" if passed else "FAILED", raw_root=str(args.raw.resolve()),
                  parsed_root=str(args.analysis.resolve()), trace_reviews=summaries,
                  scope="Independent read-only original SQLite correlation and phase audit; no CUDA imported or GPU work")
    (args.output / "review.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
