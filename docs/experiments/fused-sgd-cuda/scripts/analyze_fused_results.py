"""CPU-only, hash-free analysis of the complete stage 4B measurement set.

Independent processes are the repetition unit. Raw per-update/step values are
retained. Nsight data are used only for attribution, never training throughput.
"""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

METHODS = ("baseline", "fused", "torch")
LABELS = {"baseline": "HyTorch original SGD", "fused": "HyTorch fused SGD", "torch": "PyTorch foreach SGD"}
COLORS = {"baseline": "#4077b6", "fused": "#d77b21", "torch": "#36976c"}
TOLERANCE = dict(atol=1e-6, rtol=1e-5)


def json_file(path):
    return json.loads(Path(path).read_text())


def json_lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def csv_rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def identical(actual, expected):
    return actual.keys() == expected.keys() and all(np.array_equal(actual[k], expected[k]) for k in actual)


def statistics(values):
    values = np.asarray(values, dtype=np.float64)
    return dict(count=int(values.size), mean=float(values.mean()),
                std=float(values.std(ddof=1)) if values.size > 1 else 0.,
                median=float(np.median(values)), p95=float(np.percentile(values, 95)),
                minimum=float(values.min()), maximum=float(values.max()))


def difference(actual, reference):
    delta = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
    nonzero = reference != 0
    relative = delta[nonzero] / np.abs(reference[nonzero].astype(np.float64))
    allowed = TOLERANCE["atol"] + TOLERANCE["rtol"] * np.abs(reference.astype(np.float64))
    return dict(elements=int(actual.size), max_abs=float(delta.max()),
                max_rel_nonzero=float(relative.max()) if relative.size else 0.,
                reference_zero_count=int((~nonzero).sum()),
                mismatch_count=int((delta > allowed).sum()),
                exact_equal=bool(np.array_equal(actual, reference)),
                passed=bool(np.allclose(actual, reference, **TOLERANCE)))


def microbench(raw):
    rows, repeat_rows, comparisons, source_checks = [], [], [], []
    for repeat in range(1, 4):
        baseline_path = raw / f"update-baseline-r{repeat}"
        base_initial = arrays(baseline_path / "initial_weights.npz")
        base_gradients = arrays(baseline_path / "gradients.npz")
        for method in METHODS:
            path = raw / f"update-{method}-r{repeat}"
            config = json_file(path / "config.json")
            summary = json_file(path / "summary.json")
            if config["nvtx"]:
                raise ValueError("Instrumented run cannot be ordinary microbenchmark timing: " + str(path))
            source_checks.append(dict(experiment="update", method=method, repeat=repeat,
                                      initial_exact=identical(arrays(path / "initial_weights.npz"), base_initial),
                                      gradients_exact=identical(arrays(path / "gradients.npz"), base_gradients),
                                      gradients_unchanged=summary["gradients_unchanged"],
                                      standalone_status=summary["status"],
                                      reference_max_abs=summary["max_abs"], path=str(path)))
            local = csv_rows(path / "update-benchmark-raw.csv")
            for row in local:
                rows.append(dict(method=method, repeat=repeat, local_repeat=int(row["repeat"]),
                                 mode=row["mode"], step=int(row["step"]), elapsed_us=float(row["elapsed_us"]),
                                 source=str(path / "update-benchmark-raw.csv")))
            for mode in ("sync_wall", "host_submission"):
                selected = [r for r in rows if r["method"] == method and r["repeat"] == repeat and r["mode"] == mode]
                repeat_rows.append(dict(method=method, repeat=repeat, mode=mode, unit="us", scope="within independent process",
                                        **statistics([r["elapsed_us"] for r in selected])))
                for local_repeat in sorted({r["local_repeat"] for r in selected}):
                    final = arrays(path / f"final-{mode}-r{local_repeat}.npz")
                    base = arrays(baseline_path / f"final-{mode}-r{local_repeat}.npz")
                    for name in base:
                        comparisons.append(dict(experiment="standalone_update", method=method, repeat=repeat,
                                                mode=mode, local_repeat=local_repeat, parameter=name,
                                                reference="HyTorch original SGD", **difference(final[name], base[name])))
    aggregate = []
    for mode in ("sync_wall", "host_submission"):
        base = [r["mean"] for r in repeat_rows if r["method"] == "baseline" and r["mode"] == mode]
        for method in METHODS:
            selected = [r["mean"] for r in repeat_rows if r["method"] == method and r["mode"] == mode]
            aggregate.append(dict(method=method, mode=mode, unit="us", scope="distribution of independent process means",
                                  speedup_vs_baseline=float(np.mean(base) / np.mean(selected)), **statistics(selected)))
    return rows, repeat_rows, aggregate, comparisons, source_checks


def training(raw):
    runs, steps_out, losses_out, comparison, checks = [], [], [], [], []
    for mode in ("throughput", "phases"):
        for repeat in range(1, 4):
            baseline_path = raw / f"training-{mode}-baseline-r{repeat}"
            initial = arrays(baseline_path / "initial_weights.npz")
            inputs = arrays(baseline_path / "comparison_inputs.npz")
            final = arrays(baseline_path / "final_weights.npz")
            reference_loss = np.asarray([r["loss"] for r in json_lines(baseline_path / "losses.jsonl")])
            for method in METHODS:
                path = raw / f"training-{mode}-{method}-r{repeat}"
                config, summary = json_file(path / "config.json"), json_file(path / "summary.json")
                if config["nvtx"] or not config["no_fingerprints"]:
                    raise ValueError("Training comparison requires no NVTX and --no-fingerprints: " + str(path))
                check = dict(experiment="training", method=method, repeat=repeat, mode=mode,
                             initial_exact=identical(arrays(path / "initial_weights.npz"), initial),
                             inputs_exact=identical(arrays(path / "comparison_inputs.npz"), inputs),
                             status=summary["status"], path=str(path))
                checks.append(check)
                steps = json_lines(path / "steps.jsonl")
                losses = json_lines(path / "losses.jsonl")
                wall = sum(epoch["train_wall_s"] for epoch in summary["epochs"])
                samples = sum(epoch["samples"] for epoch in summary["epochs"])
                row = dict(method=method, repeat=repeat, mode=mode, steps=len(steps), samples=samples,
                           train_wall_s=wall, samples_per_sec=samples / wall,
                           measured_step_mean_ms=1000 * np.mean([s["step_s"] for s in steps]),
                           first_loss=losses[0]["loss"], final_loss=losses[-1]["loss"],
                           finite_loss=bool(all(np.isfinite(s["loss"]) for s in losses)),
                           phase_data_ms=None, phase_transfer_ms=None, phase_forward_ms=None,
                           phase_backward_ms=None, phase_update_ms=None, phase_other_ms=None,
                           monitor_sample_count=summary["hardware"]["sample_count"],
                           gpu_utilization_mean=summary["hardware"]["gpu_utilization_pct_mean"],
                           gpu_memory_sampled_peak_mib=summary["hardware"]["gpu_memory_used_mib_max"],
                           source=str(path))
                if mode == "phases":
                    for phase in ("data", "transfer", "forward", "backward", "update", "other"):
                        row["phase_" + phase + "_ms"] = 1000 * float(np.mean([s[phase + "_s"] for s in steps]))
                runs.append(row)
                for step_index, (step, loss) in enumerate(zip(steps, losses), 1):
                    steps_out.append(dict(step, method=method, repeat=repeat, mode=mode, measured_step=step_index))
                    losses_out.append(dict(loss, method=method, repeat=repeat, mode=mode, measured_step=step_index))
                actual = arrays(path / "final_weights.npz")
                for name in final:
                    comparison.append(dict(experiment="training_final_weights", method=method, repeat=repeat,
                                           mode=mode, parameter=name, reference="HyTorch original SGD",
                                           **difference(actual[name], final[name])))
                comparison.append(dict(experiment="training_loss", method=method, repeat=repeat, mode=mode,
                                       parameter="per_step_loss", reference="HyTorch original SGD",
                                       **difference(np.asarray([r["loss"] for r in losses]), reference_loss)))
    aggregate = []
    for mode in ("throughput", "phases"):
        baseline = [r["samples_per_sec"] for r in runs if r["mode"] == mode and r["method"] == "baseline"]
        for method in METHODS:
            selected = [r for r in runs if r["mode"] == mode and r["method"] == method]
            metrics = ["train_wall_s", "samples_per_sec", "measured_step_mean_ms"]
            if mode == "phases":
                metrics += ["phase_" + phase + "_ms" for phase in ("data", "transfer", "forward", "backward", "update", "other")]
            for metric in metrics:
                values = [r[metric] for r in selected]
                aggregate.append(dict(method=method, mode=mode, metric=metric,
                                      scope="distribution of three independent process measurements",
                                      **statistics(values)))
    return runs, steps_out, losses_out, aggregate, comparison, checks


def make_plots(output, figures, trace_root, update_summary, training_runs, losses):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/hytorch-fused-sgd-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figures.mkdir(parents=True, exist_ok=True)
    names = [LABELS[m].replace(" SGD", "\nSGD") for m in METHODS]
    colors = [COLORS[m] for m in METHODS]

    def save(fig, filename, source):
        fig.text(.01, .01, "Source: measured logs / Nsight export visualization; " + source, fontsize=8)
        fig.tight_layout(rect=(0, .06, 1, 1))
        path = figures / filename
        if path.exists():
            raise FileExistsError(path)
        fig.savefig(path, dpi=170)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, mode, title in zip(axes, ("sync_wall", "host_submission"),
                               ("Complete optimizer update, synchronized", "Host call submission, no per-update sync")):
        selected = [next(r for r in update_summary if r["method"] == m and r["mode"] == mode) for m in METHODS]
        ax.bar(names, [r["mean"] for r in selected], yerr=[r["std"] for r in selected], color=colors, capsize=4)
        ax.set_ylabel("Microseconds / update")
        ax.set_title(title)
        for i, row in enumerate(selected):
            ax.text(i, row["mean"] + row["std"], f'{row["mean"]:.2f}', ha="center", va="bottom")
    save(fig, "update-timing.png", "3 independent processes; bars mean ± sample SD of process means; no NVTX")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    selected = [[r for r in training_runs if r["method"] == m and r["mode"] == "throughput"] for m in METHODS]
    means = [np.mean([r["samples_per_sec"] for r in group]) for group in selected]
    stds = [np.std([r["samples_per_sec"] for r in group], ddof=1) for group in selected]
    ax.bar(names, means, yerr=stds, color=colors, capsize=4)
    for i, group in enumerate(selected):
        ax.scatter([i] * len(group), [r["samples_per_sec"] for r in group], color="black", s=18, zorder=4)
        ax.text(i, means[i] + stds[i], f"{means[i]:.1f}", ha="center", va="bottom")
    ax.set_ylabel("Training samples / second")
    ax.set_title("DonkeyNet training throughput: 256 steps per process")
    save(fig, "training-throughput.png", "un-instrumented training epoch wall time; 3 processes; error bars sample SD")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    bottom = np.zeros(len(METHODS))
    for phase, color in (("forward", "#4679b0"), ("backward", "#d78535"), ("update", "#399c74"),
                         ("data", "#ad88bd"), ("transfer", "#c95b64"), ("other", "#999999")):
        values = [np.mean([r["phase_" + phase + "_ms"] for r in training_runs if r["method"] == method and r["mode"] == "phases"])
                  for method in METHODS]
        ax.bar(names, values, bottom=bottom, label=phase, color=color)
        bottom += values
    ax.set_ylabel("Synchronized wall time, ms / training step")
    ax.set_title("DonkeyNet phase timing (separate diagnostic runs)")
    ax.set_ylim(0, float(bottom.max()) * 1.16)
    ax.legend(ncol=3, loc="upper center")
    save(fig, "training-phase-timing.png", "3 process means; phase synchronization changes timing; not GPU kernel duration")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for method in METHODS:
        groups = [[r["loss"] for r in losses if r["method"] == method and r["repeat"] == repeat and r["mode"] == "throughput"]
                  for repeat in range(1, 4)]
        avg = np.mean(groups, axis=0)
        axes[0].plot(np.arange(1, len(avg) + 1), avg, label=LABELS[method], color=COLORS[method], alpha=.8)
    base = np.asarray([r["loss"] for r in losses if r["method"] == "baseline" and r["repeat"] == 1 and r["mode"] == "throughput"])
    for method in ("fused", "torch"):
        actual = np.asarray([r["loss"] for r in losses if r["method"] == method and r["repeat"] == 1 and r["mode"] == "throughput"])
        axes[1].plot(np.arange(1, len(base) + 1), np.abs(actual - base), label=LABELS[method], color=COLORS[method])
    axes[0].set_ylabel("MSE loss (three repeat mean)")
    axes[0].set_title("Identical synthetic inputs and initial weights")
    axes[1].set_ylabel("Absolute loss difference vs original SGD")
    axes[1].set_title("First process repeat; all values retained")
    for ax in axes:
        ax.set_xlabel("Measured training step")
        ax.legend(fontsize=8)
    save(fig, "training-loss.png", "un-instrumented throughput runs; synthetic data, not task accuracy")

    phases = {}
    for kind in ("update", "training"):
        for method in METHODS:
            phases[(kind, method)] = next(r for r in csv_rows(trace_root / f"trace-{kind}-{method}" / "phase-summary.csv") if r["phase"] == "update")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(METHODS))
    for offset, kind, label in ((-.18, "update", "Standalone update trace"), (.18, "training", "Training update trace")):
        values = [float(phases[(kind, m)]["gpu_kernel_count"]) / float(phases[(kind, m)]["nvtx_count"]) for m in METHODS]
        bars = axes[0].bar(x + offset, values, width=.36, label=label)
        axes[0].bar_label(bars, fmt="%.0f")
    axes[0].set_xticks(x, names)
    axes[0].set_ylabel("GPU kernel launches / measured update")
    axes[0].set_title("CUDA API correlation → update NVTX range")
    axes[0].legend(fontsize=8)
    values = [1000 * float(phases[("training", m)]["gpu_kernel_sum_ms"]) / float(phases[("training", m)]["nvtx_count"]) for m in METHODS]
    bars = axes[1].bar(names, values, color=colors)
    axes[1].bar_label(bars, fmt="%.2f")
    axes[1].set_ylabel("Sum of GPU update kernel execution, μs / step")
    axes[1].set_title("Training trace; no host or launch gaps included")
    save(fig, "kernel-launches.png", "Nsight instrumented traces; timings do not replace ordinary wall / throughput measurements")

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    for ax, method in zip(axes, METHODS):
        trace = trace_root / f"trace-training-{method}"
        nvtx = [r for r in csv_rows(trace / "nvtx-events.csv") if r["name"] == "update"]
        chosen = nvtx[len(nvtx) // 2]
        rid, start = int(chosen["sqlite_rowid"]), float(chosen["start"])
        def intervals(filename, selector=lambda row: True):
            values = csv_rows(trace / filename)
            return [((float(r["start"]) - start) / 1000, (float(r["end"]) - float(r["start"])) / 1000)
                    for r in values if rid in json.loads(r["nvtx_ids"]) and selector(r)]
        host_us = (float(chosen["end"]) - start) / 1000
        ax.broken_barh([(0, host_us)], (2.1, .65), facecolors=COLORS[method], alpha=.65)
        ax.broken_barh(intervals("cuda-api-events.csv", lambda r: r["kind"] == "launch"), (1.1, .65), facecolors="#8059a5")
        ax.broken_barh(intervals("kernel-events.csv"), (.1, .65), facecolors="#222222")
        ax.set_yticks([.4, 1.4, 2.4], ["GPU kernel", "CUDA launch API", "Host update NVTX"])
        ax.set_title(f"{LABELS[method]} — measured update {len(nvtx)//2+1}/{len(nvtx)} (chronological midpoint)", fontsize=10)
        ax.grid(axis="x", alpha=.2)
    axes[-1].set_xlabel("Microseconds relative to selected update host start (same scale for all methods)")
    save(fig, "update-trace-timeline.png", "actual Nsight events; kernels attributed by CUDA API correlation IDs, not overlap")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--trace-analysis", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--figures", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    update_raw, update_repeat, update_summary, update_comparisons, update_checks = microbench(args.raw)
    training_runs, steps, losses, training_summary, training_comparisons, training_checks = training(args.raw)
    for name, rows in (("update-benchmark-raw.csv", update_raw), ("update-benchmark-per-repeat.csv", update_repeat),
                       ("update-benchmark-summary.csv", update_summary), ("update-cross-method-correctness.csv", update_comparisons),
                       ("training-raw.csv", training_runs), ("training-steps.csv", steps), ("training-losses.csv", losses),
                       ("training-summary.csv", training_summary), ("training-numerical-comparison.csv", training_comparisons)):
        write_csv(args.output / name, rows)
    initial_inputs_ok = all(r["initial_exact"] and r["gradients_exact"] for r in update_checks) and all(r["initial_exact"] and r["inputs_exact"] for r in training_checks)
    numerical = dict(tolerance=TOLERANCE, reference="same-mode same-repeat HyTorch original SGD",
                     initial_and_inputs_all_exact=initial_inputs_ok, source_checks=update_checks + training_checks,
                     standalone_all_passed=all(r["passed"] for r in update_comparisons),
                     standalone_reference_all_passed=all(r["standalone_status"] == "PASSED" and r["gradients_unchanged"] for r in update_checks),
                     training_all_passed=all(r["passed"] for r in training_comparisons),
                     training_losses_all_finite=all(r["finite_loss"] for r in training_runs),
                     standalone_max_abs=max(r["max_abs"] for r in update_comparisons),
                     training_max_abs=max(r["max_abs"] for r in training_comparisons),
                     training_failed_rows=[r for r in training_comparisons if not r["passed"]],
                     standalone_failed_rows=[r for r in update_comparisons if not r["passed"]])
    with (args.output / "numerical-comparison.json").open("x") as stream:
        json.dump(numerical, stream, indent=2, allow_nan=False)
        stream.write("\n")
    make_plots(args.output, args.figures, args.trace_analysis, update_summary, training_runs, losses)
    metrics = {m: [r["samples_per_sec"] for r in training_runs if r["method"] == m and r["mode"] == "throughput"] for m in METHODS}
    report = dict(status="PASSED" if initial_inputs_ok and numerical["standalone_all_passed"] and numerical["standalone_reference_all_passed"] and numerical["training_all_passed"] and numerical["training_losses_all_finite"] else "NUMERICAL_MISMATCH_RECORDED",
                  raw_root=str(args.raw.resolve()), trace_analysis_root=str(args.trace_analysis.resolve()),
                  throughput_mean={m: float(np.mean(v)) for m, v in metrics.items()},
                  fused_throughput_speedup_vs_baseline=float(np.mean(metrics["fused"]) / np.mean(metrics["baseline"])),
                  torch_throughput_speedup_vs_fused=float(np.mean(metrics["torch"]) / np.mean(metrics["fused"])),
                  counting="3 independent process runs per method/mode; 256 measured training steps each; microbench >=100 updates",
                  timing="Throughput=total samples/epoch training wall; phase runs synchronized separately; trace GPU kernel sum excludes host gaps",
                  figures=sorted(str(p) for p in args.figures.glob("*.png")))
    with (args.output / "analysis-summary.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
