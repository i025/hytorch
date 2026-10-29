"""Produce disjoint backward groups and envelope-limited API/transfer summaries."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

from analyze_history import duration, family, write_csv


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    group_rows, operation_rows, transfer_rows, api_rows = [], [], [], []
    names = ["trace-" + detail + "-" + method
             for detail in ("phase", "detail") for method in ("cupy", "combined", "torch")]
    for name in names:
        source = args.input / name
        summary = json.loads((source / "profiling-summary.json").read_text())
        lo, hi = summary["training_envelope_start_ns"], summary["training_envelope_end_ns"]
        apis = read(source / "cuda-api-events.csv")
        by_api = {a["api_event_id"]: a for a in apis}
        kernels = read(source / "kernel-events.csv")
        copies = read(source / "memcpy-events.csv")
        memsets_path = source / "memset-events.csv"
        memsets = read(memsets_path) if memsets_path.exists() else []

        def submitted_in_window(event):
            api = by_api.get(event["api_event_id"])
            return api is not None and lo <= int(api["start"]) < hi

        selected_apis = [a for a in apis if lo <= int(a["start"]) < hi]
        for kind, events in (("memcpy", copies), ("memset", memsets)):
            grouped = defaultdict(list)
            for event in events:
                if submitted_in_window(event):
                    grouped[event["phase"], event["name"]].append(event)
            for (phase, event_name), values in grouped.items():
                transfer_rows.append(dict(trace=name, activity_kind=kind, phase=phase, name=event_name,
                                          count=len(values), bytes=sum(int(e["bytes"]) for e in values),
                                          gpu_sum_ms=sum(float(e["duration_ms"]) for e in values),
                                          gpu_union_ms=duration([(int(e["start"]), int(e["end"])) for e in values])/1e6))
        grouped = defaultdict(list)
        for api in selected_apis:
            grouped[api["phase"], api["kind"], api["name"]].append(api)
        for (phase, kind, api_name), values in grouped.items():
            api_rows.append(dict(trace=name, phase=phase, kind=kind, name=api_name, count=len(values),
                                 host_sum_ms=sum(float(a["duration_ms"]) for a in values),
                                 host_union_ms=duration([(int(a["start"]), int(a["end"])) for a in values])/1e6))

        if name not in ("trace-detail-cupy", "trace-detail-combined"):
            continue
        backward = [k for k in kernels if k["phase"] == "backward" and submitted_in_window(k)]
        total = sum(float(k["duration_ms"]) for k in backward)
        by_family, by_operation = defaultdict(list), defaultdict(list)
        for kernel in backward:
            by_family[family(kernel["name"])].append(kernel)
            path = json.loads(kernel["operator_path"])
            node = next((p for p in path if p.startswith("OP/bwd:")), "unlabelled")
            by_operation[node.split("|", 1)[0]].append(kernel)
        for groups, destination, field in ((by_family, group_rows, "kernel_family"),
                                            (by_operation, operation_rows, "backward_node")):
            ranked = sorted(groups.items(), key=lambda item: -sum(float(k["duration_ms"]) for k in item[1]))
            for rank, (key, values) in enumerate(ranked, 1):
                elapsed = sum(float(k["duration_ms"]) for k in values)
                destination.append(dict(trace=name, rank=rank, **{field:key}, kernel_count=len(values),
                                        gpu_kernel_sum_ms=elapsed, gpu_kernel_mean_us=elapsed*1000/len(values),
                                        fraction_of_backward_gpu_kernel_sum=elapsed/total))
            assert sum(len(values) for values in groups.values()) == len(backward)
    write_csv(args.output / "backward-kernel-families.csv", group_rows)
    write_csv(args.output / "backward-function-groups.csv", operation_rows)
    write_csv(args.output / "envelope-submitted-transfer-summary.csv", transfer_rows)
    write_csv(args.output / "envelope-cuda-api-summary.csv", api_rows)
    result = dict(status="COMPLETE", traces=names,
                  scope="GPU events belong to CUDA APIs submitted within the measured training NVTX envelope. Initialization/warmup/exit APIs and their GPU activities are excluded even when their phase is outside_phase.",
                  disjointness="Each HyTorch backward GPU kernel has exactly one kernel-family group and one outer backward Function group. These are two alternative partitions, not additive tables.",
                  timing="API host elapsed sum/union, GPU event sum/union and transfer bytes are distinct. Copy kernels belong to kernel-family tables, not CUDA Memcpy tables. Memset is separately preserved.",
                  caveat="Function GPU totals omit host-only autograd traversal and do not measure each Function's wall-time contribution. They cannot be used as end-to-end speedup percentages.")
    (args.output / "definitions.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
