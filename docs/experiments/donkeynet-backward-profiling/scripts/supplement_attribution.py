"""Summarize layer-labelled trace exports; no CUDA operations or timing inference.

Input is analyze_trace.py output. GPU events have already been joined through
correlationId -> submitting API -> same-thread NVTX ancestry. This script only
groups those joins; asynchronous GPU/host overlap is never used for ownership.
"""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import re

from analyze_history import write_csv


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def metadata(path):
    layer = "unlabelled"
    part = "unlabelled"
    node = "unlabelled"
    internal = "unlabelled"
    for label in path:
        match = re.search(r"(?:^|\|)layer=([^|]+)", label)
        if match:
            layer = match.group(1)
        if label.startswith("OP/part:"):
            part = label.split("|", 1)[0][len("OP/part:"):]
        if label.startswith("OP/bwd:"):
            node = label.split("|", 1)[0]
        if label.startswith("OP/internal:"):
            internal = label.split("|", 1)[0]
    return layer, part, node, internal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    phases = {row["phase"]: row for row in read_csv(args.input / "phase-summary.csv")}
    operators = read_csv(args.input / "operator-hotspots.csv")
    kernels = read_csv(args.input / "kernel-events.csv")
    trace = json.loads((args.input / "profiling-summary.json").read_text())
    backward_gpu_ms = float(phases["backward"]["gpu_kernel_sum_ms"])
    backward_host_ms = float(phases["backward"]["host_nvtx_sum_ms"])

    selected = []
    for row in operators:
        op = row["operator"]
        if row["attribution"] != "inclusive" or row["phase"] != "backward":
            continue
        if not op.startswith(("OP/bwd:Conv2d|", "OP/bwd:MatMul|", "OP/part:")):
            continue
        layer, part, node, _ = metadata([op])
        result = dict(row, layer=layer, part=part, backward_node=node)
        result["fraction_of_backward_gpu_kernel_sum"] = float(row["gpu_kernel_sum_ms"]) / backward_gpu_ms if backward_gpu_ms else 0
        result["fraction_of_backward_host_nvtx_sum"] = float(row["host_nvtx_inclusive_ms"]) / backward_host_ms if backward_host_ms else 0
        result["kernel_sum_fraction_of_training_envelope"] = float(row["gpu_kernel_sum_ms"]) / trace["training_envelope_ms"]
        selected.append(result)
    write_csv(args.output / "layer-backward.csv", selected)

    groups = defaultdict(list)
    copies = defaultdict(list)
    for kernel in kernels:
        if kernel["phase"] != "backward":
            continue
        path = json.loads(kernel["operator_path"])
        layer, part, node, internal = metadata(path)
        key = layer, part, node, internal, kernel["name"]
        groups[key].append(kernel)
        if kernel["name"].startswith("cupy_copy"):
            copies[key].append(kernel)

    def rows_for(grouped):
        rows = []
        for key, events in grouped.items():
            total = sum(float(event["duration_ms"]) for event in events)
            rows.append(dict(layer=key[0], part=key[1], backward_node=key[2],
                             innermost_internal=key[3], kernel=key[4],
                             calls=len(events), gpu_kernel_sum_ms=total,
                             gpu_kernel_mean_us=total * 1000 / len(events),
                             fraction_of_backward_gpu_kernel_sum=total / backward_gpu_ms if backward_gpu_ms else 0,
                             operator_paths=json.dumps(sorted({event["operator_path"] for event in events}))))
        return sorted(rows, key=lambda row: -row["gpu_kernel_sum_ms"])

    grouped_rows = rows_for(groups)
    copy_rows = rows_for(copies)
    write_csv(args.output / "backward-kernel-layer.csv", grouped_rows)
    write_csv(args.output / "copy-sources.csv", copy_rows)
    conv = [row for row in selected if row["operator"].startswith("OP/bwd:Conv2d|")]
    first_input = [row for row in selected if row["part"] == "Conv2d.dInput" and row["layer"] == "conv1"]
    target = [row for row in selected if row["part"] == "MeanSquareError.dTarget"]
    conv_gpu = sum(float(row["gpu_kernel_sum_ms"]) for row in conv)
    conv_host = sum(float(row["host_nvtx_inclusive_ms"]) for row in conv)
    first_gpu = sum(float(row["gpu_kernel_sum_ms"]) for row in first_input)
    result = dict(
        label=trace["label"], input=str(args.input.resolve()),
        status="LAYER_ATTRIBUTION_COMPLETE" if first_input else "NO_LABELLED_FIRST_LAYER_DINPUT",
        denominator_definitions={
            "backward_gpu_kernel_sum_ms": "Sum of GPU kernel durations attributed to backward by submitting API; not backward wall time",
            "backward_host_nvtx_sum_ms": "Sum of outer backward NVTX host ranges; includes host execution/submission/waits inside those ranges",
            "training_envelope_ms": "Elapsed from first training phase start to last phase end; instrumented trace, not uninstrumented throughput"},
        backward_gpu_kernel_sum_ms=backward_gpu_ms,
        backward_host_nvtx_sum_ms=backward_host_ms,
        training_envelope_ms=trace["training_envelope_ms"],
        convolution_backward_gpu_kernel_sum_ms=conv_gpu if conv else None,
        convolution_backward_host_nvtx_sum_ms=conv_host if conv else None,
        convolution_fraction_of_backward_gpu_kernel_sum=conv_gpu / backward_gpu_ms if conv and backward_gpu_ms else None,
        convolution_fraction_of_backward_host_nvtx_sum=conv_host / backward_host_ms if conv and backward_host_ms else None,
        first_input_gradient_gpu_kernel_sum_ms=first_gpu if first_input else None,
        first_input_gradient_fraction_of_backward_gpu_kernel_sum=first_gpu / backward_gpu_ms if first_input and backward_gpu_ms else None,
        first_input_gradient_kernel_sum_fraction_of_training_envelope=first_gpu / trace["training_envelope_ms"] if first_input else None,
        first_input_gradient_rows=first_input,
        mse_extra_target_negation_rows=target,
        backward_cupy_copy_kernel_count=sum(row["calls"] for row in copy_rows),
        backward_cupy_copy_kernel_sum_ms=sum(row["gpu_kernel_sum_ms"] for row in copy_rows),
        caveats=["Layer/part rows are inclusive and overlap their children; never sum all rows as disjoint work.",
                 "First-layer dInput is executed but unused by this training objective; its measured kernel sum is not demonstrated end-to-end speedup from eliminating it.",
                 "MSE target scope measures only the additional negation; shared gradient arithmetic also serves prediction gradients.",
                 "Copy kernels are distinct from CUPTI Memcpy events; recorded call ancestry identifies their trigger, not a complete allocation-byte model.",
                 "NVTX host time outside CUDA APIs is not a direct measurement of Python execution; it can include profiler costs and OS scheduling."])
    (args.output / "layer-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
