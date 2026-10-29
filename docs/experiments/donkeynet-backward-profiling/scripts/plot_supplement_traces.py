"""Plot real Stage 4A supplemental Nsight CSV exports; no GPU work.

Every plotted value comes from analyze_trace.py/supplement_attribution.py.
Kernel sums, NVTX host intervals, and uninstrumented throughput are distinct.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


METHODS = ("cupy", "combined", "torch")
LABELS = {"cupy": "HyTorch / CuPy", "combined": "HyTorch / Combined", "torch": "PyTorch CUDA"}
PHASES = ("forward", "backward", "update")
COLORS = {"data": "#9ba7ad", "transfer": "#45a6a0", "forward": "#397cac",
          "backward": "#d77845", "update": "#9864ac", "outside_phase": "#777777"}
PARTS = ("dInput", "dWeight", "dBias")
PART_COLORS = ("#d77845", "#397cac", "#66a182")
LAYERS = ["conv" + str(i) for i in range(1, 6)]


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def phase_rows(root, detail, method):
    return {row["phase"]: row for row in read_csv(root / ("trace-{}-{}".format(detail, method)) / "phase-summary.csv")}


def save(fig, output, name, footer):
    path = output / name
    if path.exists():
        raise FileExistsError(path)
    fig.text(.025, .025, footer, fontsize=9, color="#46515a", va="bottom")
    fig.savefig(path, dpi=175, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(str(path), flush=True)


def layer_hotspots(root, output):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("DonkeyNet backward: layout kernels attributed to actual convolution layers", fontsize=15)
    x = np.arange(5)
    for i, method in enumerate(METHODS[:2]):
        rows = read_csv(root / ("layer-trace-detail-" + method) / "backward-kernel-layer.csv")
        steps = int(phase_rows(root, "detail", method)["backward"]["nvtx_count"])
        for j, category in enumerate(("Col2im", "Im2col", "CuPy copy")):
            values = []
            counts = []
            for layer in LAYERS:
                selected = [r for r in rows if r["layer"] == layer and (
                    (category == "Col2im" and "col2im" in r["kernel"].lower()) or
                    (category == "Im2col" and "im2col" in r["kernel"].lower()) or
                    (category == "CuPy copy" and r["kernel"].startswith("cupy_copy")))]
                values.append(sum(float(r["gpu_kernel_sum_ms"]) for r in selected) / steps)
                counts.append(sum(int(r["calls"]) for r in selected) / steps)
            ax = axes[i, j]
            bars = ax.bar(x, values, color=PART_COLORS[j], width=.65)
            ax.bar_label(bars, labels=["{:.3f}".format(v) for v in values], fontsize=9, padding=3)
            ax.set_xticks(x, ["{}\n{:.0f} / step".format(layer, count) for layer, count in zip(LAYERS, counts)], fontsize=9)
            ax.set_title(LABELS[method] + " — " + category, fontsize=11)
            ax.set_ylabel("GPU kernel time (ms / step)")
            ax.margins(y=.22)
            ax.grid(axis="y", alpha=.18)
            ax.set_axisbelow(True)
    fig.tight_layout(rect=(0, .14, 1, .94))
    save(fig, output, "trace-layer-layout-hotspots.png",
         "Source: layer-trace-detail-{cupy,combined}/backward-kernel-layer.csv; sums divided by recorded backward phase count.\n"
         "Detailed Nsight + NVTX traces; GPU attribution uses CUDA API correlation IDs and named parent nodes.\n"
         "Only Conv1–5 backward shown; x labels include launches per step. Copy kernels differ from Memcpy events. Log visualization.")


def conv_components(root, output):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("DonkeyNet convolution backward: dInput, dWeight and dBias", fontsize=15)
    for i, method in enumerate(METHODS[:2]):
        rows = read_csv(root / ("layer-trace-detail-" + method) / "layer-backward.csv")
        steps = int(phase_rows(root, "detail", method)["backward"]["nvtx_count"])
        for j, field in enumerate(("gpu_kernel_sum_ms", "host_nvtx_inclusive_ms")):
            ax = axes[i, j]
            bottom = np.zeros(5)
            for part, color in zip(PARTS, PART_COLORS):
                values = np.array([sum(float(r[field]) for r in rows
                    if r["layer"] == layer and r["part"] == "Conv2d." + part) / steps for layer in LAYERS])
                ax.bar(np.arange(5), values, bottom=bottom, label=part, color=color, width=.65)
                if part == "dInput":
                    ax.text(0, values[0] / 2, "{:.3f}\ndInput".format(values[0]),
                            ha="center", va="center", fontsize=9, color="white", fontweight="bold")
                bottom += values
            for k, total in enumerate(bottom):
                ax.text(k, total, "{:.3f}".format(total), ha="center", va="bottom", fontsize=9)
            ax.set_xticks(np.arange(5), LAYERS)
            ax.set_title(LABELS[method] + (" — GPU kernels" if j == 0 else " — host component NVTX"), fontsize=11)
            ax.set_ylabel("Milliseconds per training step")
            ax.margins(y=.2)
            ax.grid(axis="y", alpha=.18)
            ax.set_axisbelow(True)
    axes[0, 1].legend(frameon=False, ncol=3, loc="upper right", fontsize=9)
    fig.tight_layout(rect=(0, .14, 1, .94))
    save(fig, output, "trace-convolution-gradient-components.png",
         "Source: layer-trace-detail-{cupy,combined}/layer-backward.csv; GPU sums and host NVTX elapsed time are separate measures.\n"
         "Component scopes wrap unchanged original forward calls; host scopes omit surrounding Function bookkeeping.\n"
         "Conv1 dInput is unused by this training objective. Its kernel time is not measured end-to-end savings. Log visualization.")


def kernel_counts(root, output):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("DonkeyNet: launch counts and GPU execution under Nsight Systems", fontsize=15)
    for i, detail in enumerate(("phase", "detail")):
        for j, field in enumerate(("gpu_kernel_count", "gpu_kernel_sum_ms")):
            ax = axes[i, j]
            bottom = np.zeros(3)
            for phase in PHASES:
                values = []
                for method in METHODS:
                    rows = phase_rows(root, detail, method)
                    steps = int(rows["backward"]["nvtx_count"])
                    values.append(float(rows[phase][field]) / steps)
                ax.bar(np.arange(3), values, bottom=bottom, label=phase.capitalize(), color=COLORS[phase], width=.65)
                bottom += values
            for k, total in enumerate(bottom):
                ax.text(k, total, ("{:.0f}" if j == 0 else "{:.3f}").format(total), ha="center", va="bottom", fontsize=10)
            ax.set_xticks(np.arange(3), [LABELS[m].replace(" / ", "\n") for m in METHODS], fontsize=10)
            ax.set_ylabel("Kernel launches / step" if j == 0 else "GPU kernel sum (ms / step)")
            ax.set_title("Phase labels only" if detail == "phase" else "Detailed operator labels + CuPy memory hooks")
            ax.margins(y=.2)
            ax.grid(axis="y", alpha=.18)
            ax.set_axisbelow(True)
    axes[0, 1].legend(frameon=False, ncol=3, fontsize=9)
    fig.tight_layout(rect=(0, .13, 1, .94))
    save(fig, output, "trace-kernel-count-and-time.png",
         "Source: trace-{phase,detail}-{cupy,combined,torch}/phase-summary.csv; forward + backward + update, per recorded step.\n"
         "Warmup and out-of-phase events excluded. GPU kernel sums are not training wall time or device utilization.\n"
         "Separate traced executions; use uninstrumented measurements for training speed comparisons. Log visualization.")


def timeline_rows(root, detail, method, displayed_steps, first_step):
    directory = root / "trace-{}-{}".format(detail, method)
    summary = json.loads((directory / "profiling-summary.json").read_text())
    origin = summary["training_envelope_start_ns"]
    nvtx = [r for r in read_csv(directory / "nvtx-events.csv") if r["name"] in COLORS]
    data = [r for r in nvtx if r["name"] == "data"]
    window_start = (int(data[first_step - 1]["start"]) - origin) / 1e6
    update = [r for r in nvtx if r["name"] == "update"]
    limit = (int(update[first_step + displayed_steps - 2]["end"]) - origin) / 1e6 - window_start
    rows = [("Host phases", [{"relative_start_ms": (int(r["start"]) - origin) / 1e6,
             "duration_ms": r["duration_ms"], "phase": r["name"]} for r in nvtx])]
    for label, file in (("CUDA APIs", "cuda-api-events.csv"), ("GPU kernels", "kernel-events.csv"),
                        ("GPU memcpy", "memcpy-events.csv")):
        rows.append((label, read_csv(directory / file)))
    for _, events in rows:
        for event in events:
            event["relative_start_ms"] = float(event["relative_start_ms"]) - window_start
    return rows, limit


def timelines(root, output, displayed_steps, first_step=1):
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    fig.suptitle("Host submission and device execution — measured steps {}–{}".format(
        first_step, first_step + displayed_steps - 1), fontsize=15)
    for i, method in enumerate(METHODS):
        inputs = [timeline_rows(root, detail, method, displayed_steps, first_step) for detail in ("phase", "detail")]
        row_limit = max(limit for _, limit in inputs)
        for j, (rows, limit) in enumerate(inputs):
            ax = axes[i, j]
            for lane, (_, events) in enumerate(rows):
                for phase, color in COLORS.items():
                    intervals = []
                    for event in events:
                        start, length = float(event["relative_start_ms"]), float(event["duration_ms"])
                        if event["phase"] == phase and start < limit and start + length > 0:
                            left, right = max(0., start), min(limit, start + length)
                            intervals.append((left, right - left))
                    ax.broken_barh(intervals, (3 - lane - .32, .64), facecolors=color, edgecolors="none")
            ax.axvline(limit, color="#999999", linestyle=":", linewidth=.8)
            ax.set_xlim(0, row_limit * 1.02)
            ax.set_ylim(-.65, 3.65)
            ax.set_yticks([3, 2, 1, 0], [name for name, _ in rows], fontsize=9)
            ax.set_title(LABELS[method] + (" — phase labels only" if j == 0 else " — detailed labels"), fontsize=11)
            ax.set_xlabel("Milliseconds since step {} data phase".format(first_step))
            ax.grid(axis="x", alpha=.15)
            ax.set_axisbelow(True)
    handles = [Patch(color=COLORS[p], label=p.capitalize().replace("_", " ")) for p in COLORS]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .11), ncol=6, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, .16, 1, .94))
    name = "trace-host-device-timelines.png" if first_step == 1 else "trace-host-device-timelines-steps2-3.png"
    save(fig, output, name,
         "Source: trace-{phase,detail}-{cupy,combined,torch}/{nvtx,cuda-api,kernel,memcpy}-events.csv. Device colors follow API correlations.\n"
         "Each row uses the same x scale across its two separate executions; dotted lines end the displayed step window.\n"
         "Blank device intervals are gaps in recorded activity, not direct GPU-utilization or Python-execution measurements. Log visualization.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeline-steps", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    layer_hotspots(args.input, args.output)
    conv_components(args.input, args.output)
    kernel_counts(args.input, args.output)
    timelines(args.input, args.output, args.timeline_steps)
    timelines(args.input, args.output, args.timeline_steps, first_step=2)


if __name__ == "__main__":
    main()
