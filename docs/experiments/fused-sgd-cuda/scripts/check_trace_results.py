"""CPU-only consistency checks for stage 4B correlation-derived trace results."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--summary-root", type=Path, required=True)
    parser.add_argument("--correctness-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    launch_rows = rows(args.summary_root/"launch-counts.csv")
    updates = rows(args.summary_root/"trace-update-raw.csv")
    steps = rows(args.summary_root/"trace-training-step-raw.csv")
    metadata = rows(args.summary_root/"metadata-host-raw.csv")
    summary = json.loads((args.summary_root/"trace-summary.json").read_text())
    checked = []
    for trace in summary["traces"]:
        name = trace["trace"]
        selected = [r for r in launch_rows if r["trace"] == name]
        assert sum(int(r["gpu_kernel_count"]) for r in selected) == trace["measured_phase_kernel_count"]
        kernels = rows(args.analysis_root/name/"kernel-events.csv")
        assert len(kernels) == trace["whole_trace_kernel_count"]
        assert len({k["event_id"] for k in kernels}) == len(kernels)
        assert all(k["api_event_id"] != "unmatched" for k in kernels)
        phase_updates = [r for r in updates if r["trace"] == name]
        update_kernels = [k for k in kernels if k["phase"] == "update"]
        assert sum(int(r["gpu_kernel_count"]) for r in phase_updates) == len(update_kernels)
        assert sum(int(r["explicit_memcpy_count"]) for r in phase_updates) == 0
        expected_count = 32 if trace["method"] == "baseline" else 1
        assert {int(r["gpu_kernel_count"]) for r in phase_updates} == {expected_count}
        assert len(phase_updates) == (200 if trace["workload"] == "update" else 16)
        expected_metadata = 4*len(phase_updates) if trace["method"] == "fused" else 0
        assert len([r for r in metadata if r["trace"] == name]) == expected_metadata
        checked.append(dict(trace=name, phase_kernel_count=trace["measured_phase_kernel_count"],
                            update_ranges=len(phase_updates), kernels_per_update=expected_count,
                            cross_thread_api_phase_attributions=trace["cross_thread_phase_kernel_count"]))
    for method, expected in (("baseline", 148), ("fused", 117), ("torch", 94)):
        selected = [row for row in steps if row["method"] == method]
        assert len(selected) == 16
        assert {int(row["gpu_kernel_count"]) for row in selected} == {expected}
    pointer_counts = []
    for tensor in (0, 1):
        pointers = []
        for step in range(10):
            with np.load(args.correctness_root/f"fresh_gradients_step_{step}_tensor_{tensor}.npz") as data:
                pointers.append(int(data["gradient_pointer"]))
                np.testing.assert_array_equal(data["gradient"], data["gradient_after"])
                np.testing.assert_allclose(data["fused"], data["baseline"], atol=1e-6, rtol=1e-5)
        assert len(set(pointers)) == 10
        pointer_counts.append(dict(tensor=tensor, updates=10, distinct_gradient_pointers=10))
    result = dict(status="PASSED", scope="CPU verification of recorded GPU artifacts; no GPU work launched",
                  trace_checks=checked, fresh_gradient_storage_checks=pointer_counts,
                  metadata_count_rule="Four nested/subranges per fused update; none in baseline or Torch updates",
                  limitations="Checks establish internal attribution consistency, not absence of Nsight overhead or general performance significance.")
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
