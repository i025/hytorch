import json
from pathlib import Path


def read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def list_runs(root_dir="runs"):
    root = Path(root_dir)
    if not root.exists():
        return []
    runs = []
    for run_dir in sorted(root.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        metrics_path = run_dir / "metrics.jsonl"
        config_path = run_dir / "config.json"
        summary_path = run_dir / "summary.json"
        if not metrics_path.exists() and not summary_path.exists():
            continue
        records = read_jsonl(metrics_path)
        summary = read_json(summary_path, default={}) or {}
        config = read_json(config_path, default={}) or {}
        runs.append({
            "run_id": run_dir.name,
            "run_dir": str(run_dir),
            "has_metrics": metrics_path.exists(),
            "has_config": config_path.exists(),
            "has_summary": summary_path.exists(),
            "num_records": len(records),
            "metrics": available_metrics(records),
            "last_record": records[-1] if records else {},
            "summary": summary,
            "config": config,
        })
    return runs


def load_run(run_dir):
    run_dir = Path(run_dir)
    records = read_jsonl(run_dir / "metrics.jsonl")
    return {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "config": read_json(run_dir / "config.json", default={}) or {},
        "summary": read_json(run_dir / "summary.json", default={}) or {},
        "records": records,
        "metrics": available_metrics(records),
        "epoch_records": [r for r in records if r.get("type") == "epoch"],
        "batch_records": [r for r in records if r.get("type") == "batch"],
    }


def available_metrics(records):
    excluded = {"time", "elapsed", "type", "step", "epoch", "phase", "event", "message"}
    metrics = set()
    for record in records:
        for key, value in record.items():
            if key in excluded:
                continue
            if isinstance(value, (int, float)):
                metrics.add(key)
    return sorted(metrics)


def metric_series(records, metric, record_type=None):
    series = []
    for index, record in enumerate(records):
        if record_type is not None and record.get("type") != record_type:
            continue
        if metric not in record or not isinstance(record[metric], (int, float)):
            continue
        x = record.get("epoch", record.get("step", index))
        series.append({"x": x, "y": float(record[metric]), "record_type": record.get("type")})
    return series


def latest_record(records, record_type="epoch"):
    for record in reversed(records):
        if record_type is None or record.get("type") == record_type:
            return record
    return records[-1] if records else {}


def latest_diagnostics(records, record_type="epoch"):
    record = latest_record(records, record_type=record_type)
    diagnostics = {}
    for key in (
        "weight_histogram",
        "gradient_histogram",
        "weight_heatmap",
        "weight_mean",
        "weight_std",
        "weight_min",
        "weight_max",
        "grad_mean",
        "grad_std",
        "global_grad_norm",
        "zero_grad_ratio",
    ):
        if key in record:
            diagnostics[key] = record[key]
    return diagnostics


def diagnostics_payload(records, record_type="epoch"):
    diagnostics = latest_diagnostics(records, record_type=record_type)
    diagnostics["global_grad_norm_series"] = metric_series(
        records,
        "global_grad_norm",
        record_type=record_type,
    )
    diagnostics["weight_std_series"] = metric_series(records, "weight_std", record_type=record_type)
    diagnostics["grad_std_series"] = metric_series(records, "grad_std", record_type=record_type)
    return diagnostics


def summarize_run(run):
    records = run["records"] if "records" in run else read_jsonl(Path(run["run_dir"]) / "metrics.jsonl")
    epoch_records = [r for r in records if r.get("type") == "epoch"]
    last_epoch = epoch_records[-1] if epoch_records else (records[-1] if records else {})
    metrics = {}
    for key in available_metrics(records):
        values = [r[key] for r in epoch_records if key in r and isinstance(r[key], (int, float))]
        if not values:
            values = [r[key] for r in records if key in r and isinstance(r[key], (int, float))]
        if values:
            metrics[key] = {
                "last": float(values[-1]),
                "min": float(min(values)),
                "max": float(max(values)),
            }
    return {
        "run_id": run.get("run_id") or Path(run["run_dir"]).name,
        "num_records": len(records),
        "num_epochs": len(epoch_records),
        "last_epoch": last_epoch.get("epoch"),
        "metrics": metrics,
    }


def compare_runs(run_dirs, metrics=None):
    loaded = [load_run(run_dir) for run_dir in run_dirs]
    if metrics is None:
        metric_names = sorted({m for run in loaded for m in run["metrics"]})
    else:
        metric_names = list(metrics)
    runs = []
    for run in loaded:
        records = run["epoch_records"] or run["records"]
        runs.append({
            "run_id": run["run_id"],
            "summary": summarize_run(run),
            "series": {
                metric: metric_series(records, metric)
                for metric in metric_names
            },
        })
    return {"metrics": metric_names, "runs": runs}


def export_markdown_report(run_dirs, output_path, metrics=None):
    comparison = compare_runs(run_dirs, metrics=metrics)
    lines = ["# HyTorch Run Comparison Report", ""]
    lines.append("| Run | Records | Epochs | Metrics |")
    lines.append("| --- | ---: | ---: | --- |")
    for run in comparison["runs"]:
        summary = run["summary"]
        metric_text = ", ".join(
            f"{name}: last={values['last']:.6g}"
            for name, values in summary["metrics"].items()
        )
        lines.append(
            f"| {run['run_id']} | {summary['num_records']} | "
            f"{summary['num_epochs']} | {metric_text} |"
        )
    lines.append("")
    lines.append("## Compared Metrics")
    for metric in comparison["metrics"]:
        lines.append(f"- `{metric}`")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path
