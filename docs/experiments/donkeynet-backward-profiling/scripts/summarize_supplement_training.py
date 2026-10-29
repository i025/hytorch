"""CPU-only audit and plots of the 27 completed supplement training runs.

Keep synchronized phase timings, uninstrumented training-loop throughput, and
NVTX/memory-hook throughput separate. This script never reads Nsight timings.
"""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
METHODS = ("cupy", "combined", "torch")
LABELS = {"cupy": "HyTorch / CuPy", "combined": "HyTorch / Combined", "torch": "PyTorch CUDA"}
COLORS = {"cupy": "#4279a5", "combined": "#d58b32", "torch": "#4e986d"}
PHASES = ("data", "transfer", "forward", "backward", "update", "other")
ATOL, RTOL = 1e-4, 1e-3


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_csv(path, rows):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def relative(path):
    return str(path.resolve().relative_to(ROOT))


def numerical_row(run, reference_run, kind, name, actual, reference, exact):
    if actual.shape != reference.shape:
        raise ValueError("Mismatched shape: {} {} {}".format(run, kind, name))
    difference = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
    finite = bool(np.isfinite(actual).all() and np.isfinite(reference).all())
    equal = bool(actual.dtype == reference.dtype and np.array_equal(actual, reference))
    atol, rtol = (0., 0.) if exact else (ATOL, RTOL)
    failures = int(np.count_nonzero(~np.isfinite(difference) | (difference > atol + rtol * np.abs(reference.astype(np.float64)))))
    reference_norm = float(np.linalg.norm(reference.astype(np.float64).ravel()))
    return dict(run=run, reference_run=reference_run, kind=kind, tensor=name,
                shape="x".join(map(str, actual.shape)) or "scalar", elements=int(actual.size),
                dtype=str(actual.dtype), reference_dtype=str(reference.dtype), finite=finite,
                exact_equal=equal, atol=atol, rtol=rtol, failed_elements=failures,
                max_abs_error=float(difference.max()) if finite else None,
                relative_l2_error=float(np.linalg.norm(difference.ravel())) / reference_norm if finite and reference_norm else None,
                correct=finite and (equal if exact else failures == 0))


def load_runs(raw):
    runs = []
    for instrumented in (False, True):
        for mode in (("throughput",) if instrumented else ("throughput", "phases")):
            for method in METHODS:
                for repeat in range(1, 4):
                    path = raw / ("instrumented-{}-r{}".format(method, repeat)) if instrumented else raw / "measurements" / ("{}-{}-r{}".format(mode, method, repeat))
                    summary = read_json(path / "summary.json")
                    config = read_json(path / "config.json")
                    steps, losses = read_jsonl(path / "steps.jsonl"), read_jsonl(path / "losses.jsonl")
                    if summary["status"] != "COMPLETED" or summary["gpu_status"] != "EXECUTED":
                        raise ValueError("Run did not complete GPU training: " + str(path))
                    if len(steps) != 256 or len(losses) != 256 or summary["timing"]["steps"] != 256:
                        raise ValueError("Expected 256 measured steps: " + str(path))
                    for step, loss in zip(steps, losses):
                        if (step["epoch"], step["step"], step["samples"]) != (loss["epoch"], loss["step"], loss["samples"]):
                            raise ValueError("Step/loss alignment failure: " + str(path))
                    if config["nvtx"] != instrumented or config["mode"] != mode:
                        raise ValueError("Unexpected instrumentation or mode: " + str(path))
                    runs.append(dict(path=path, name=path.name, method=method, mode=mode,
                                     instrumented=instrumented, repeat=repeat, config=config,
                                     summary=summary, steps=steps, losses=losses))
    matched_keys = ("model", "samples", "batch", "epochs", "steps", "workers", "seed", "lr", "warmup",
                    "device", "gpu", "physical_monitor_id", "optimizer", "dropout", "dtype", "data_kind",
                    "no_monitor", "no_fingerprints", "save_comparison_inputs")
    canonical = runs[0]["config"]
    for run in runs:
        mismatch = [key for key in matched_keys if run["config"][key] != canonical[key]]
        if mismatch:
            raise ValueError("Unmatched config {}: {}".format(run["name"], mismatch))
    return runs


def collect_tables(runs):
    run_rows, step_rows, epoch_rows, loss_rows, dispatch_rows = [], [], [], [], []
    for run in runs:
        summary, steps, losses = run["summary"], run["steps"], run["losses"]
        metadata = dict(run=run["name"], method=run["method"], mode=run["mode"],
                        instrumented=run["instrumented"], repeat=run["repeat"], nsight=False)
        wall = sum(row["train_wall_s"] for row in summary["epochs"])
        samples = sum(row["samples"] for row in summary["epochs"])
        row = dict(**metadata, steps=len(steps), samples=samples, training_wall_s=wall,
                   training_samples_per_second=samples / wall,
                   measured_steps_wall_s=sum(row["step_s"] for row in steps),
                   measured_step_mean_ms=1000 * statistics.mean(row["step_s"] for row in steps),
                   loss_first=losses[0]["loss"], loss_last=losses[-1]["loss"],
                   loss_mean=statistics.mean(row["loss"] for row in losses),
                   loss_min=min(row["loss"] for row in losses), loss_max=max(row["loss"] for row in losses),
                   all_losses_finite=all(np.isfinite(row["loss"]) for row in losses),
                   config=relative(run["path"] / "config.json"), summary=relative(run["path"] / "summary.json"))
        for phase in PHASES:
            row[phase + "_mean_ms"] = 1000 * statistics.mean(step[phase + "_s"] for step in steps) if run["mode"] == "phases" else None
        run_rows.append(row)
        for index, (step, loss) in enumerate(zip(steps, losses), 1):
            step_rows.append({**metadata, "global_step": index, **step, "loss": loss["loss"],
                              "source": relative(run["path"] / "steps.jsonl")})
            loss_rows.append(dict(**metadata, global_step=index, **loss,
                                  source=relative(run["path"] / "losses.jsonl")))
        for epoch in summary["epochs"]:
            epoch_rows.append(dict(**metadata, **epoch, source=relative(run["path"] / "summary.json")))
        for operator, count in summary["dispatch_counts"].items():
            dispatch_rows.append(dict(**metadata, operator=operator, count=count,
                                      source=relative(run["path"] / "summary.json")))
    return run_rows, step_rows, epoch_rows, loss_rows, dispatch_rows


def audit_arrays(runs):
    baseline = runs[0]
    expected = {filename: arrays(baseline["path"] / filename)
                for filename in ("initial_weights.npz", "comparison_inputs.npz", "final_weights.npz")}
    input_rows, final_rows, exact_rows = [], [], []
    plain = {(run["method"], run["repeat"]): run for run in runs
             if run["mode"] == "throughput" and not run["instrumented"]}
    for run in runs:
        for filename in ("initial_weights.npz", "comparison_inputs.npz", "final_weights.npz"):
            actual = arrays(run["path"] / filename)
            if actual.keys() != expected[filename].keys():
                raise ValueError("Different tensor names: {} {}".format(run["name"], filename))
            exact = filename != "final_weights.npz"
            destination = input_rows if exact else final_rows
            for name, reference in expected[filename].items():
                destination.append(numerical_row(run["name"], baseline["name"], filename, name,
                                                 actual[name], reference, exact))
        if run["instrumented"]:
            ref = plain[(run["method"], run["repeat"])]
            ref_final = arrays(ref["path"] / "final_weights.npz")
            for name, reference in ref_final.items():
                exact_rows.append(numerical_row(run["name"], ref["name"], "instrumentation_final_weights", name,
                                                actual[name], reference, True))
            actual_loss = np.array([row["loss"] for row in run["losses"]])
            reference_loss = np.array([row["loss"] for row in ref["losses"]])
            exact_rows.append(numerical_row(run["name"], ref["name"], "instrumentation_loss_sequence", "loss_256_steps",
                                            actual_loss, reference_loss, True))
    return input_rows, final_rows, exact_rows


def aggregate(run_rows):
    groups = defaultdict(list)
    for row in run_rows:
        groups[(row["method"], row["mode"], row["instrumented"])].append(row)
    metrics = ("training_wall_s", "training_samples_per_second", "measured_steps_wall_s", "measured_step_mean_ms",
               "loss_first", "loss_last", "loss_mean", *(phase + "_mean_ms" for phase in PHASES))
    rows = []
    for (method, mode, instrumented), values in groups.items():
        for metric in metrics:
            numbers = [row[metric] for row in values if row[metric] is not None]
            if numbers:
                rows.append(dict(method=method, mode=mode, instrumented=instrumented, metric=metric,
                                 repeats=len(numbers), mean=statistics.mean(numbers), std=statistics.stdev(numbers),
                                 minimum=min(numbers), maximum=max(numbers),
                                 repeat_1=values[0][metric], repeat_2=values[1][metric], repeat_3=values[2][metric]))
    overhead = []
    for method in METHODS:
        plain, observed = groups[(method, "throughput", False)], groups[(method, "throughput", True)]
        mean_plain = statistics.mean(row["training_wall_s"] for row in plain)
        mean_observed = statistics.mean(row["training_wall_s"] for row in observed)
        overhead.append(dict(method=method, repeats=3, steps_per_repeat=256,
                             plain_mean_training_wall_s=mean_plain,
                             instrumented_mean_training_wall_s=mean_observed,
                             wall_overhead_percent=100 * (mean_observed / mean_plain - 1),
                             plain_aggregate_samples_per_second=sum(row["samples"] for row in plain) / sum(row["training_wall_s"] for row in plain),
                             instrumented_aggregate_samples_per_second=sum(row["samples"] for row in observed) / sum(row["training_wall_s"] for row in observed)))
    return rows, overhead


def figures(output, destination, run_rows, loss_rows, epoch_rows, overhead):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
    destination.mkdir(parents=True, exist_ok=True)

    def save(fig, name, footer):
        path = destination / name
        if path.exists():
            raise FileExistsError(path)
        fig.text(.02, .02, footer, fontsize=8, color="#42505a")
        fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    fig.suptitle("DonkeyNet supplement: synchronized phase wall time, no Nsight", fontsize=14)
    x = np.arange(3)
    phase_colors = {"data": "#a0aab0", "transfer": "#48a1a7", "forward": "#4279a5",
                    "backward": "#c7654b", "update": "#a077b6", "other": "#ccd0d3"}
    for offset, phase in enumerate(("forward", "backward", "update")):
        values = [[row[phase + "_mean_ms"] for row in run_rows if row["method"] == method and row["mode"] == "phases"] for method in METHODS]
        means = [statistics.mean(v) for v in values]
        bars = axes[0].bar(x + (offset - 1) * .24, means, .24, yerr=[statistics.stdev(v) for v in values],
                           capsize=3, label=phase.capitalize(), color=phase_colors[phase])
        axes[0].bar_label(bars, fmt="%.3f", fontsize=8, padding=4)
    axes[0].set_xticks(x, [LABELS[m].replace(" / ", "\n") for m in METHODS])
    axes[0].set_ylabel("Mean milliseconds per step")
    axes[0].legend(frameon=False)
    axes[0].margins(y=.22)
    left = np.zeros(3)
    for phase in PHASES:
        means = [statistics.mean(row[phase + "_mean_ms"] for row in run_rows if row["method"] == method and row["mode"] == "phases") for method in METHODS]
        axes[1].barh(x, means, left=left, color=phase_colors[phase], label=phase)
        left += means
    axes[1].set_yticks(x, [LABELS[m] for m in METHODS])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Mean measured step wall time (ms)")
    axes[1].legend(ncol=3, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(.5, 1.2))
    for index, value in enumerate(left):
        axes[1].text(value + .04, index, "{:.3f}".format(value), va="center", fontsize=9)
    axes[1].set_xlim(0, max(left) * 1.17)
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(alpha=.18)
    fig.tight_layout(rect=(0, .14, 1, .94))
    save(fig, "phase-comparison.png", "Source: results/training/{run-results,aggregate-statistics}.csv. Log visualization.\n"
         "3 independent runs x 256 steps; batch 4, FP32, seed 42, warmup 10. Error bars: sample standard deviation of run means.\n"
         "CUDA synchronization at phase boundaries: phase wall durations include host submission/waiting; these are not GPU kernel times.")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.3))
    fig.suptitle("DonkeyNet supplement: training-loop throughput and instrumentation cost", fontsize=14)
    for condition, offset, alpha, label in ((False, -.18, .9, "Uninstrumented"), (True, .18, .45, "NVTX + hooks")):
        for index, method in enumerate(METHODS):
            selected = [row for row in run_rows if row["method"] == method and row["mode"] == "throughput" and row["instrumented"] == condition]
            for axis, metric in zip(axes[:2], ("training_wall_s", "training_samples_per_second")):
                values = [row[metric] for row in selected]
                axis.bar(index + offset, statistics.mean(values), .32, yerr=statistics.stdev(values), capsize=3,
                         color=COLORS[method], alpha=alpha, label=label if index == 0 else None)
                axis.scatter(index + offset + np.array([-.06, 0., .06]), values, color="#26333b", s=12, zorder=3)
    axes[0].set_ylabel("Wall seconds per 256-step run")
    axes[1].set_ylabel("Samples/s, mean of run rates")
    for axis in axes[:2]:
        axis.legend(frameon=False, fontsize=8)
        axis.set_xticks(x, [LABELS[m].replace(" / ", "\n") for m in METHODS])
        axis.set_axisbelow(True)
        axis.grid(axis="y", alpha=.18)
    bars = axes[2].bar(x, [row["wall_overhead_percent"] for row in overhead], color=[COLORS[m] for m in METHODS])
    axes[2].bar_label(bars, fmt="%.1f%%", padding=4)
    axes[2].set_ylabel("Increase in mean training-loop wall time (%)")
    axes[2].set_xticks(x, [LABELS[m].replace(" / ", "\n") for m in METHODS])
    axes[2].margins(y=.2)
    fig.tight_layout(rect=(0, .16, 1, .93))
    save(fig, "throughput-and-overhead.png", "Source: results/training/{run-results,instrumentation-overhead}.csv. No Nsight process attached. Log visualization.\n"
         "3 runs per condition, each 256 steps: 4 epochs x 256 synthetic samples, batch 4. Training wall includes data transfer, Loss D2H and logs.\n"
         "HyTorch instrumentation: operator NVTX + CuPy memory hooks; PyTorch: native autograd NVTX. Dots: all runs; error bars: sample SD.\n"
         "Per-step boundary synchronization remains in the benchmark; these measurements differ from the resident-batch validation microtest.")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    fig.suptitle("Measured DonkeyNet Loss on synthetic data (not driving-task accuracy)", fontsize=14)
    for method in METHODS:
        selected = [row for row in loss_rows if row["method"] == method and row["mode"] == "throughput" and not row["instrumented"]]
        for repeat in range(1, 4):
            values = [row for row in selected if row["repeat"] == repeat]
            axes[0].plot([row["global_step"] for row in values], [row["loss"] for row in values],
                         color=COLORS[method], alpha=.45, linewidth=.8, label=LABELS[method] if repeat == 1 else None)
        for condition in (False, True):
            means = [statistics.mean(row["train_loss"] for row in epoch_rows
                                     if row["method"] == method and row["mode"] == "throughput"
                                     and row["instrumented"] == condition and row["epoch"] == epoch) for epoch in range(1, 5)]
            axes[1].plot(range(1, 5), means, "o-" if not condition else "x--", color=COLORS[method],
                         label=LABELS[method] + (" plain" if not condition else " instrumented"))
    axes[0].set(xlabel="Measured training step", ylabel="Per-batch MSE Loss", yscale="log")
    axes[1].set(xlabel="Epoch", ylabel="Mean epoch MSE Loss", xticks=[1, 2, 3, 4])
    for axis in axes:
        axis.legend(frameon=False, fontsize=8)
        axis.grid(alpha=.18)
    fig.tight_layout(rect=(0, .13, 1, .93))
    save(fig, "loss.png", "Source: results/training/{losses,epochs}.csv. All values originate in actual non-Nsight training logs.\n"
         "Left: all 3 uninstrumented runs per method, every batch shown. Right: mean of 3 runs; dashed lines include instrumentation.\n"
         "Identical/near-identical curves may overlap. Long-training numerical differences are retained in the CSV audits without relaxing tolerances.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    args = parser.parse_args()
    runs = load_runs(args.raw.resolve())
    run_rows, step_rows, epoch_rows, loss_rows, dispatch_rows = collect_tables(runs)
    input_rows, final_rows, exact_rows = audit_arrays(runs)
    aggregates, overhead = aggregate(run_rows)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, rows in (("run-results", run_rows), ("steps", step_rows), ("epochs", epoch_rows),
                       ("losses", loss_rows), ("dispatch-counts", dispatch_rows),
                       ("input-initial-exact-comparisons", input_rows), ("final-parameter-comparisons", final_rows),
                       ("instrumentation-exact-comparisons", exact_rows), ("aggregate-statistics", aggregates),
                       ("instrumentation-overhead", overhead)):
        write_csv(args.output / (name + ".csv"), rows)
    passed = all(row["correct"] for row in input_rows + final_rows + exact_rows) and all(row["all_losses_finite"] for row in run_rows)
    summary = dict(status="PASSED" if passed else "FAILED", runs=len(runs), steps_per_run=256, repeats=3,
                   raw_root=relative(args.raw.resolve()), nsight=False,
                   config_matching="All matched config fields in load_runs verified; backend, mode and NVTX differ intentionally",
                   initial_input_comparisons=len(input_rows), initial_input_exact=all(row["correct"] for row in input_rows),
                   final_parameter_comparisons=len(final_rows), final_parameters_passed=all(row["correct"] for row in final_rows),
                   final_parameter_reference=runs[0]["name"], final_parameter_atol=ATOL, final_parameter_rtol=RTOL,
                   final_parameter_tolerance_source="scripts/report_cuda.py compare_training; unchanged historical final-state tolerance",
                   final_parameter_max_abs_error=max(row["max_abs_error"] for row in final_rows),
                   instrumentation_exact_comparisons=len(exact_rows), instrumentation_exact=all(row["correct"] for row in exact_rows),
                   all_losses_finite=all(row["all_losses_finite"] for row in run_rows),
                   failed_comparisons=[row for row in input_rows + final_rows + exact_rows if not row["correct"]],
                   overhead=overhead,
                   timing_scopes=dict(training_wall="Sum of four epoch train_wall_s, includes DataLoader, H2D, step synchronization, loss D2H and logs",
                                      measured_step="Benchmark StepProfiler begin/end with per-step synchronization; excludes subsequent loss scalar read/logging",
                                      phases="Separate mode=phases runs, each phase boundary synchronizes CUDA; CPU submission and waiting included",
                                      instrumentation="Separate mode=throughput runs, HyTorch operator NVTX+CuPy MemoryHook, Torch autograd NVTX, no Nsight"))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    figures(args.output, args.figures, run_rows, loss_rows, epoch_rows, overhead)
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
