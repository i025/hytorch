"""Common benchmark interface and measured-result analysis; never invent results."""
import csv
import json
import statistics
import uuid
from pathlib import Path

from .config import TrainingConfig
from .launcher import run_training
from .monitor import write_json

DEFAULT_CASES = [
    {"name": "single", "mode": "single", "num_workers": 1, "transport": "socket_json"},
    {"name": "ps-socket-2", "mode": "ps", "num_workers": 2, "transport": "socket_json"},
    {"name": "ps-grpc-2", "mode": "ps", "num_workers": 2, "transport": "grpc_protobuf"},
    {"name": "ps-socket-4", "mode": "ps", "num_workers": 4, "transport": "socket_json"},
    {"name": "ps-grpc-4", "mode": "ps", "num_workers": 4, "transport": "grpc_protobuf"},
]

RING_CASES = [
    {"name": "single", "mode": "single", "num_workers": 1, "transport": "socket_json"},
    {"name": "ps-grpc-2", "mode": "ps", "num_workers": 2, "transport": "grpc_protobuf"},
    {"name": "ps-grpc-4", "mode": "ps", "num_workers": 4, "transport": "grpc_protobuf"},
    {"name": "ring-grpc-2", "mode": "ring", "num_workers": 2, "transport": "grpc_protobuf"},
    {"name": "ring-grpc-4", "mode": "ring", "num_workers": 4, "transport": "grpc_protobuf"},
]


def analyze_results(results, destination):
    """Analyze completed summaries, returning comparison rows (all times in seconds)."""
    fingerprints = {r["summary"]["dataset_fingerprint"] for r in results}
    if len(fingerprints) != 1:
        raise ValueError("Benchmark runs used different datasets; comparison rejected")
    baselines = [r["summary"]["train_time_s"] / len(r["summary"]["epochs"])
                 for r in results if r["summary"]["mode"] == "single"]
    if not baselines:
        raise ValueError("At least one completed single-process baseline is required")
    baseline = statistics.median(baselines)
    rows = []
    for item in results:
        summary = item["summary"]
        epochs, workers = summary["epochs"], summary["num_workers"]
        time_s = summary["train_time_s"] / len(epochs)
        sums = [epoch["worker_time_sums"] for epoch in epochs]
        def mean_work(key):
            return sum(values.get(key, 0) for values in sums) / len(epochs)
        row = {"case": item["case"], "repeat": item["repeat"], "run_id": summary["run_id"],
               "mode": summary["mode"], "transport": summary["transport"], "num_workers": workers,
               "mean_epoch_train_s": time_s,
               "mean_worker_compute_s": mean_work("compute_s") / workers,
               "mean_worker_communication_s": mean_work("communication_s") / workers,
               "mean_worker_data_loading_s": mean_work("data_loading_s") / workers,
               "mean_worker_gradient_copy_s": mean_work("gradient_copy_s") / workers,
               "mean_worker_parameter_copy_s": mean_work("parameter_copy_s") / workers,
               "mean_epoch_optimizer_s": sum(e["optimizer_s"] for e in epochs) / len(epochs),
               "mean_epoch_aggregation_s": sum(e["aggregation_s"] for e in epochs) / len(epochs),
               "mean_epoch_message_bytes": mean_work("sent_bytes") + mean_work("received_bytes"),
               "mean_worker_ring_reduce_scatter_s": mean_work("ring_reduce_scatter_s") / workers,
               "mean_worker_ring_allgather_s": mean_work("ring_allgather_s") / workers,
               "mean_epoch_ring_gradient_sent_bytes": mean_work("ring_gradient_sent_bytes"),
               "speedup": baseline / max(time_s, 1e-12),
               "efficiency": baseline / max(time_s * workers, 1e-12),
               "samples_per_sec": sum(e["samples"] for e in epochs) / max(summary["train_time_s"], 1e-12),
               "val_mse": summary["validation"].get("mse"),
               "val_mae": summary["validation"].get("mae"),
               "val_rmse": summary["validation"].get("rmse"),
               "gpu_sampled_peak_mb": summary["hardware"].get("gpu_memory_used_mb_max")}
        rows.append(row)
    destination = Path(destination)
    with (destination / "benchmark.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(destination / "benchmark.json", {"schema_version": 1, "rows": rows,
                "baseline_median_epoch_s": baseline,
                "byte_scope": "outgoing_training_RPC_requests_and_responses_counted_once_per_exchange"})
    grouped = {}
    for row in rows:
        grouped.setdefault(row["case"], []).append(row)
    lines = ["# PS Benchmark 分析", "", "以下数值来自本次实际完成的运行。", "",
             "全局 batch、数据划分、种子、模型和 epoch 数保持一致。单 GPU 的多个 Worker 共享算力，",
             "线性 N 倍曲线只是独立计算资源条件下的参考，不是本机应达到的目标。", "",
             "计算/通信列是每个 epoch 各 Worker 耗时的平均值；Worker 之间存在重叠，不能相加得到总墙钟耗时。",
             "通信是编码、RPC 往返及解码耗时，上传 RPC 包含等待其他 Worker、聚合及更新，不等于纯网络时间。", ""]
    for name, group in grouped.items():
        median_time = statistics.median(r["mean_epoch_train_s"] for r in group)
        representative = min(group, key=lambda r: abs(r["mean_epoch_train_s"] - median_time))
        components = {"前向与反向计算": representative["mean_worker_compute_s"],
                      "RPC/同步等待": representative["mean_worker_communication_s"],
                      "数据准备": representative["mean_worker_data_loading_s"],
                      "主机与设备间复制": representative["mean_worker_gradient_copy_s"] + representative["mean_worker_parameter_copy_s"]}
        dominant = max(components, key=components.get)
        lines.append(f"- {name}：epoch 中位耗时 {median_time:.4f}s，加速比 {baseline / max(median_time, 1e-12):.3f}；"
                     f"代表运行中观测最大的阶段是{dominant}。验证 MAE={representative['val_mae']}。")
    lines += ["", "## 后续调优", "", "- RPC/同步等待较大：先比较 JSON 与 protobuf 的字节数和编码耗时，再结合各 Worker 计算耗时排查慢进程。",
              "- 设备复制较大：评估通信张量大小；当前 PS 在 CPU 更新，梯度与参数必须经过主机内存。",
              "- 多 Worker 变慢或显存高：减少 Worker 数或全局 batch，再用相同数据重新评测。",
              "- 数据准备较大：将异步 DataLoader 作为后续适配任务；本版本按统一全局批次顺序同步读取。",
              "- 比较验证 MSE/MAE，不能仅凭速度选择配置。单次短训练的结果不代表稳定收敛效果。",
              "", "suggested_config.json 按各配置重复运行的 epoch 中位耗时选择候选；",
              "它是速度候选，尚未通过精度门槛筛选。运行 retrain 命令即可再训练，结果会写入新的运行目录。"]
    (destination / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for mode, transport in sorted({(r["mode"], r["transport"]) for r in rows if r["mode"] != "single"}):
        points = {}
        for row in rows:
            if row["transport"] == transport and row['mode'] == mode:
                points.setdefault(row["num_workers"], []).append(row["mean_epoch_train_s"])
        xs = sorted(points)
        ax.plot([1] + xs, [1.0] + [baseline / statistics.median(points[x]) for x in xs], "o-", label=f'{mode}/{transport}')
    maximum = max(r["num_workers"] for r in rows)
    ax.plot([1, maximum], [1, maximum], "k--", label="N x reference (independent devices)")
    ax.set(xlabel="Training workers sharing one device", ylabel="Speedup vs single process", title="PS training benchmark")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination / "speedup.png", dpi=150)
    plt.close(fig)
    return rows


def run_benchmark(config, cases=None):
    if isinstance(config, dict):
        config = TrainingConfig.from_dict(config)
    else:
        config = TrainingConfig.from_dict(config.to_dict())
    config.validate(require_data=True)
    if config.dropout != 0 or config.augment:
        raise ValueError("For comparable worker-count benchmarks, use dropout=0 and augment=false")
    cases = (RING_CASES if config.mode == 'ring' else DEFAULT_CASES) if cases is None else cases
    if not cases or not any(case.get("mode") == "single" for case in cases):
        raise ValueError("Benchmark cases must include a single-process baseline")
    names = [case["name"] for case in cases]
    if len(set(names)) != len(names):
        raise ValueError("Benchmark case names must be unique")
    for case in cases:
        if set(case) != {"name", "mode", "num_workers", "transport"}:
            raise ValueError("Case fields must be name, mode, num_workers, transport")
        TrainingConfig.from_dict(dict(config.to_dict(), mode=case["mode"],
                                      transport=case["transport"], num_workers=case["num_workers"],
                                      run_name=case["name"]))
    destination = Path(config.output_dir).resolve() / f"benchmark-{uuid.uuid4().hex[:12]}"
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "benchmark_config.json", {"config": config.to_dict(), "cases": cases})
    results = []
    try:
        for repeat in range(1, config.benchmark_repeats + 1):
            for case in cases:
                print(f"[Benchmark] repeat {repeat}/{config.benchmark_repeats}: {case['name']}", flush=True)
                values = dict(config.to_dict(), mode=case["mode"], transport=case["transport"],
                              num_workers=case["num_workers"], run_name=case["name"], output_dir=str(destination))
                summary = run_training(values)
                results.append({"case": case["name"], "repeat": repeat, "summary": summary})
                write_json(destination / "runs.json", results)
        rows = analyze_results(results, destination)
        best_name = min(names, key=lambda name: statistics.median(
            r["mean_epoch_train_s"] for r in rows if r["case"] == name))
        selected = next(item for item in results if item["case"] == best_name)
        chosen = json.loads((Path(selected["summary"]["run_dir"]) / "config.json").read_text(encoding="utf-8"))
        chosen.update(run_name="donkey-tuned", output_dir=str(Path(config.output_dir).resolve()))
        write_json(destination / "suggested_config.json", chosen)
        result = {"schema_version": 1, "status": "completed", "benchmark_dir": str(destination),
                  "rows": rows, "suggested_config": str(destination / "suggested_config.json")}
        write_json(destination / "status.json", result)
        return result
    except BaseException as exc:
        write_json(destination / "status.json", {"status": "failed", "error": str(exc),
                                                "completed_runs": len(results)})
        raise
