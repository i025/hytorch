"""Replot existing DonkeyNet evidence only; no GPU, training, or fingerprints."""
import csv
from collections import defaultdict
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "docs/experiments/donkeynet-backward-profiling"
RESULTS = OUT / "results"
FIGURES = OUT / "figures"
HISTORY = ROOT / "docs/experiments/a100-gpu-window"
TRAIN = HISTORY / "training-comparison"
NSYS = RESULTS / "historical-nsys-fixed-01"
ORDER = ["cupy", "combined", "torch"]
LABELS = {"cupy": "HyTorch / CuPy", "combined": "HyTorch / Combined", "torch": "PyTorch CUDA"}
COLORS = {"cupy": "#4279a5", "combined": "#d58b32", "torch": "#4e986d"}
PHASE_COLORS = {"data": "#9aabb4", "transfer": "#48a1a7", "forward": "#4279a5",
                "backward": "#c7654b", "update": "#a077b6", "other": "#c6ccd0",
                "outside_nvtx": "#7d8489"}
DATE = "2026-10-29 Asia/Shanghai"


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save(fig, name, footer):
    fig.text(.02, .025, footer, fontsize=9, color="#42505a", va="bottom")
    fig.savefig(FIGURES / name, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"figure: {FIGURES / name}", flush=True)


def selected_training_rows():
    phase_source = TRAIN / "donkey-phases-fixed-details-01/phase-means.csv"
    throughput_source = TRAIN / "donkey-throughput-fixed-details-01/epoch-results.csv"
    phase_rows, throughput_rows = [], []
    for source, mode, target in ((phase_source, "phases", phase_rows),
                                  (throughput_source, "throughput", throughput_rows)):
        for row in read_csv(source):
            variant, repeat = row["run"].rsplit("_r", 1)
            if variant not in ORDER:
                continue
            directory = TRAIN / f"donkey-{mode}-fixed-01" / row["run"]
            config = json.loads((directory / "config.json").read_text())
            steps = [json.loads(line) for line in (directory / "steps.jsonl").read_text().splitlines()]
            assert len(steps) == 16 and config["batch"] == 4 and config["seed"] == 42
            assert config["no_fingerprints"] and not config["nvtx"]
            timestamp = datetime.fromtimestamp(steps[0]["timestamp"], timezone.utc)
            metadata = dict(variant=variant, repeat=int(repeat), evidence="HISTORICAL_NOT_NEW_MEASUREMENT",
                            mode=mode, nsight_instrumented=False, steps=16, batch=4, seed=42,
                            started_utc=timestamp.isoformat(),
                            started_asia_shanghai=timestamp.astimezone(timezone(timedelta(hours=8))).isoformat(),
                            source=str(source.relative_to(ROOT)))
            target.append(dict(**metadata, **row))
    assert len(phase_rows) == 3 and len(throughput_rows) == 9
    write_csv(RESULTS / "historical-phase-timing.csv", phase_rows)
    write_csv(RESULTS / "historical-training-comparison.csv", throughput_rows)
    return phase_rows, throughput_rows


def phase_figure(rows):
    values = {row["variant"]: row for row in rows}
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.6), gridspec_kw={"width_ratios": [1.1, 1]})
    fig.suptitle("Historical DonkeyNet phase timing — synchronized wall time", fontsize=16, y=.98)
    x = np.arange(3)
    phases = ["forward", "backward", "update"]
    for offset, phase in enumerate(phases):
        y = [float(values[v][phase + "_mean_ms"]) for v in ORDER]
        bars = axes[0].bar(x + (offset-1)*.23, y, width=.23, label=phase.capitalize(), color=PHASE_COLORS[phase])
        axes[0].bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    axes[0].set_xticks(x, [LABELS[v].replace(" / ", "\n") for v in ORDER])
    axes[0].set_ylabel("Mean milliseconds per step")
    axes[0].set_ylim(0, 5.6)
    axes[0].legend(frameon=False, fontsize=9)
    left = np.zeros(3)
    for phase in ["data", "transfer", "forward", "backward", "update", "other"]:
        y = np.array([float(values[v][phase + "_mean_ms"]) for v in ORDER])
        axes[1].barh(x, y, left=left, label=phase.capitalize(), color=PHASE_COLORS[phase])
        left += y
    axes[1].set_yticks(x, [LABELS[v] for v in ORDER])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Mean complete step time (ms)")
    axes[1].set_xlim(0, 9.7)
    for i, total in enumerate(left):
        axes[1].text(total+.12, i, f"{total:.3f}", va="center", fontsize=10)
    axes[1].legend(ncol=3, loc="upper right", bbox_to_anchor=(1, 1.14), fontsize=8, frameon=False)
    for ax in axes:
        ax.grid(axis="y" if ax is axes[0] else "x", alpha=.18)
        ax.set_axisbelow(True)
    fig.tight_layout(rect=(0, .16, 1, .93))
    save(fig, "phase-timing.png",
         f"Source: historical-phase-timing.csv, originally donkey-phases-fixed-details-01/phase-means.csv | {DATE}\n"
         "Non-Nsight run; 1 repeat × 16 steps, batch 4. Phase boundaries synchronize CUDA; these are not pure GPU kernel durations.\n"
         "Historical log visualization, not a screenshot or a new Stage 4A GPU measurement.")


def throughput_figure(rows):
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.6))
    fig.suptitle("Historical DonkeyNet training — three independent throughput runs", fontsize=15, y=.98)
    stats = []
    for i, variant in enumerate(ORDER):
        selected = sorted((r for r in rows if r["variant"] == variant), key=lambda r: int(r["repeat"]))
        wall_ms = np.array([float(r["train_wall_s"])*1000 for r in selected])
        rates = np.array([float(r["train_samples_per_sec"]) for r in selected])
        stats.append(dict(variant=variant, repeats=3, steps_per_repeat=16,
                          epoch_wall_mean_ms=float(wall_ms.mean()), epoch_wall_std_ms=float(wall_ms.std(ddof=1)),
                          mean_individual_samples_per_sec=float(rates.mean()),
                          aggregate_samples_per_sec=float(192/(wall_ms.sum()/1000))))
        for ax, measurements in zip(axes, [wall_ms, rates]):
            mean, std = measurements.mean(), measurements.std(ddof=1)
            ax.bar(i, mean, .6, color=COLORS[variant], alpha=.7,
                   yerr=std, capsize=5, error_kw={"linewidth": 1})
            ax.scatter(i+np.array([-.14, 0, .14]), measurements, s=28, color="#29333a", zorder=3)
            ax.text(i, mean+std+(.06*mean), f"{mean:.2f}", ha="center", fontsize=10)
    axes[0].set_ylabel("64-sample training-loop wall time (ms)")
    axes[0].set_ylim(0, 150)
    axes[1].set_ylabel("Samples/s (mean of individual run rates)")
    axes[1].set_ylim(0, 1000)
    for ax in axes:
        ax.set_xticks(range(3), [LABELS[v].replace(" / ", "\n") for v in ORDER])
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    fig.tight_layout(rect=(0, .17, 1, .93))
    save(fig, "training-comparison.png",
         f"Source: historical-training-comparison.csv, originally donkey-throughput-fixed-details-01/epoch-results.csv | {DATE}\n"
         "No Nsight / no operator NVTX. Each repeat: 16 steps, batch 4, 64 synthetic samples; 2 warmup steps precede timing.\n"
         "Dots: all three repeats. Error bars: sample standard deviation. Epoch time includes Loss D2H/logging; per-step boundary sync remains.\n"
         "Historical log visualization. This short experiment does not establish sustained training throughput.")
    return stats


def family_rows():
    grouped = defaultdict(lambda: {"count": 0, "total_ms": 0.})
    backward = defaultdict(lambda: {"count": 0, "total_ms": 0.})
    for row in read_csv(NSYS / "kernel-summary.csv"):
        if row["scope"] != "submitted_in_training_nvtx":
            continue
        for target in [grouped] + ([backward] if row["phase"] == "backward" else []):
            target[row["family"]]["count"] += int(row["count"])
            target[row["family"]]["total_ms"] += float(row["total_ms"])
    def finish(table, scope):
        total = sum(r["total_ms"] for r in table.values())
        rows = [dict(scope=scope, family=family, count=value["count"], total_ms=value["total_ms"],
                     mean_us=value["total_ms"]*1000/value["count"], fraction_gpu_duration=value["total_ms"]/total,
                     source="historical-nsys-fixed-01/kernel-summary.csv") for family, value in table.items()]
        return sorted(rows, key=lambda r: -r["total_ms"])
    all_rows = finish(grouped, "submitted_in_training_nvtx")
    back_rows = finish(backward, "backward_submitted_in_training_nvtx")
    assert sum(r["count"] for r in all_rows) == 2368
    assert sum(r["count"] for r in back_rows) == 1232
    write_csv(RESULTS / "historical-kernel-families.csv", all_rows)
    write_csv(RESULTS / "historical-backward-kernel-families.csv", back_rows)
    return all_rows, back_rows


def backward_figure(rows):
    fig, ax = plt.subplots(figsize=(11.8, 5.6))
    y = np.arange(len(rows))
    vals = [r["total_ms"] for r in rows]
    ax.barh(y, vals, color=["#c7654b", "#d58b32", "#4279a5", "#4e986d", "#a077b6", "#9aabb4"][:len(rows)])
    for i, r in enumerate(rows):
        ax.text(r["total_ms"]+.05, i, f'{r["total_ms"]:.3f} ms | {r["fraction_gpu_duration"]:.1%} | {r["count"]} calls',
                va="center", fontsize=10)
    ax.set_yticks(y, [r["family"] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, max(vals)*1.65)
    ax.set_xlabel("Cumulative actual GPU kernel duration (ms), 16 training steps")
    ax.set_title("Historical Backward: GPU kernel families, not Python-operator timing", fontsize=15, pad=16)
    ax.grid(axis="x", alpha=.18)
    ax.set_axisbelow(True)
    fig.tight_layout(rect=(0, .18, 1, 1))
    save(fig, "backward-kernel-distribution.png",
         f"Source: historical-backward-kernel-families.csv ← historical-nsys-fixed-01/kernel-summary.csv | {DATE}\n"
         "Nsight-instrumented Combined trace. Backward: 1,232 kernels / 12.940981 ms GPU duration vs 95.977064 ms host NVTX time.\n"
         "Attribution: kernel correlationId → submitting API → enclosing host phase. No per-layer NVTX existed in this historical trace.\n"
         "Percentages divide GPU kernel-duration sum only; they are not fractions of end-to-end wall time or measured GPU utilization.")


def launch_figure(rows):
    fig, axes = plt.subplots(1, 2, figsize=(14.3, 6.7), sharey=True)
    y = np.arange(len(rows))
    for ax, key, label in [(axes[0], "count", "Kernel launch count"), (axes[1], "total_ms", "Cumulative GPU execution (ms)")]:
        val = [r[key] for r in rows]
        bars = ax.barh(y, val, color=["#d58b32" if r["family"].startswith("HyTorch") else "#4279a5" for r in rows])
        ax.bar_label(bars, fmt="%d" if key == "count" else "%.3f", padding=4, fontsize=9)
        ax.set_xlim(0, max(val)*1.25)
        ax.set_xlabel(label)
        ax.grid(axis="x", alpha=.18)
        ax.set_axisbelow(True)
    axes[0].set_yticks(y, [r["family"] for r in rows])
    axes[0].invert_yaxis()
    fig.suptitle("Historical training trace: 2,368 kernels across 16 steps", fontsize=16)
    axes[1].legend(handles=[Patch(color="#d58b32", label="HyTorch compiled CUDA C/C++"),
                            Patch(color="#4279a5", label="CuPy / library kernels")], loc="lower right", fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, .14, 1, .96))
    save(fig, "kernel-launches.png",
         f"Source: historical-kernel-families.csv ← historical-nsys-fixed-01/kernel-summary.csv | {DATE}\n"
         "Nsight-instrumented Combined trace; kernels submitted inside training NVTX phases. Full trace has 2,664 kernels including warmup/setup.\n"
         "Count and GPU-duration sum are separate quantities; neither is CUDA API submission time. Historical log visualization.")


def timeline_figure(summary):
    start = int(summary["training_nvtx_envelope_start_ns"])
    span = float(summary["training_nvtx_envelope_ms"])
    nvtx = read_csv(NSYS / "nvtx-events.csv")
    apis = read_csv(NSYS / "cuda-api-events.csv")
    kernels = read_csv(NSYS / "kernel-events.csv")
    copies = read_csv(NSYS / "memcpy-events.csv")
    zoom_end = next((int(r["end"])-start)/1e6 for r in nvtx if r["name"] == "update") + .2
    fig, axes = plt.subplots(2, 1, figsize=(15.3, 7.2))
    fig.suptitle("Historical Nsight timeline — host phases, API calls, and GPU activity", fontsize=16)
    def interval(row):
        left = (int(row["start"])-start)/1e6
        return left, (int(row["end"])-int(row["start"]))/1e6
    for ax, end in zip(axes, [span, zoom_end]):
        for rows, lane in [(nvtx, 2), (apis, 1), (kernels, 0), (copies, 0)]:
            for row in rows:
                left, width = interval(row)
                if left+width < 0 or left > end:
                    continue
                phase = row.get("phase", row.get("name", "outside_nvtx"))
                color = "#48a1a7" if rows is copies else PHASE_COLORS.get(phase, "#7d8489")
                ax.broken_barh([(left, width)], (lane-.28, .56), facecolors=color)
        ax.set_xlim(0, end)
        ax.set_ylim(-.5, 2.65)
        ax.set_yticks([0, 1, 2], ["GPU kernels + memcpy", "CUDA API host calls", "Host NVTX phases"])
        ax.set_xlabel("Milliseconds relative to first training NVTX range")
        ax.grid(axis="x", alpha=.18)
        ax.set_axisbelow(True)
    axes[0].set_title(f"Entire training envelope: {span:.6f} ms; GPU activity union {summary['activity_covered_ms']:.6f} ms", fontsize=11)
    axes[1].set_title("First training-step window (same absolute time origin)", fontsize=11)
    axes[0].legend(handles=[Patch(color=PHASE_COLORS[p], label=p.replace("_", " "))
                            for p in ["data", "transfer", "forward", "backward", "update", "outside_nvtx"]],
                   ncol=6, loc="upper center", bbox_to_anchor=(.5, 1.35), frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, .15, 1, .92))
    save(fig, "cpu-gpu-timeline.png",
         f"Source: historical-nsys-fixed-01/{{nvtx-events,cuda-api-events,kernel-events,memcpy-events}}.csv | {DATE}\n"
         "Historical Nsight-instrumented Combined trace. GPU bars use submitting-phase colors; GPU execution can cross host NVTX boundaries.\n"
         "14.657% is the profiled process's kernel+memcpy interval-union coverage, NOT GPU/SM utilization.\n"
         "Uncovered host/API intervals do not by themselves prove Python computation: CPU sampling/context-switch capture was disabled.")


def accounting_figure():
    rows = {r["phase"]: r for r in read_csv(NSYS / "phase-summary.csv")}
    phases = ["forward", "backward", "update"]
    metrics = [("host_nvtx_sum_ms", "Host NVTX elapsed", "#9aabb4"),
               ("gpu_kernel_sum_ms", "GPU kernel execution", "#4279a5"),
               ("cuda_api_sum_ms", "CUDA API host duration", "#d58b32")]
    fig, ax = plt.subplots(figsize=(11.7, 5.5))
    x = np.arange(3)
    for i, (key, label, color) in enumerate(metrics):
        values = [float(rows[p][key]) for p in phases]
        bars = ax.bar(x+(i-1)*.24, values, width=.24, color=color, label=label)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_xticks(x, [p.capitalize() for p in phases])
    ax.set_ylabel("Cumulative milliseconds across 16 steps")
    ax.set_ylim(0, 114)
    ax.set_title("Historical profiling time domains — overlapping, not additive", fontsize=15, pad=15)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=.18)
    ax.set_axisbelow(True)
    fig.tight_layout(rect=(0, .16, 1, 1))
    save(fig, "host-gpu-time-accounting.png",
         f"Source: historical-nsys-fixed-01/phase-summary.csv | {DATE} | Nsight-instrumented Combined\n"
         "These three time domains overlap: do not stack/add them, or call CPU API duration GPU execution time.\n"
         "The historical trace used mode=throughput; phase NVTX records host submission, not per-phase synchronized wall timing.")


def main():
    FIGURES.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
    phases, throughput = selected_training_rows()
    phase_figure(phases)
    stats = throughput_figure(throughput)
    all_families, backward_families = family_rows()
    backward_figure(backward_families)
    launch_figure(all_families)
    summary = json.loads((NSYS / "summary.json").read_text())
    timeline_figure(summary)
    accounting_figure()
    metadata = dict(evidence="HISTORICAL_REANALYSIS_ONLY_NEW_GPU_MEASUREMENTS_NOT_RUN",
                    source_date_timezone=DATE, phase_repeats=1, throughput_repeats=3, steps_per_repeat=16,
                    throughput_statistics=stats, figures=sorted(p.name for p in FIGURES.glob("*.png")),
                    backward_gpu_kernel_duration_ms=sum(r["total_ms"] for r in backward_families),
                    backward_family_count=sum(r["count"] for r in backward_families),
                    nvtx_activity_coverage_definition=summary["coverage_definition"])
    (RESULTS / "historical-figure-data.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
