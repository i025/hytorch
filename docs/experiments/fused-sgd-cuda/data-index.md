# 阶段 4B 数据索引

服务器仓库路径 `/home/xymo/projects/hytorch` 指向 `/srv/lab-disks/disk2/users/xymo/projects/hytorch`。本轮原始数据全部保留；大文件未上传 GitHub。所有文件索引仅记录路径和大小，没有哈希校验。

## 服务器原始目录

| 绝对路径 | 内容 |
|---|---|
| `/home/xymo/projects/hytorch/runs/fused_sgd_cuda_01/` | GPU 测试全部原数组；9 组 Update；18 组训练初始/最终权重、固定梯度、合成输入；40 条命令和日志；6 份 `.nsys-rep` 和 SQLite |
| `/home/xymo/projects/hytorch/runs/fused_sgd_cuda_analysis_01/` | 六 Trace 完整事件 CSV、NVTX、Kernel、API、Memcpy、统计、解析命令/日志 |
| `/home/xymo/projects/hytorch/runs/fused_sgd_cuda_monitor_01/` | 两次空闲检查、单卡映射、133 条显存/负载/功率/温度/时钟监控、全部任务终端输出、正常退出记录 |
| `/home/xymo/projects/hytorch/runs/fused_sgd_cuda_build_01/` | 新动态库、build.log、build.json；旧动态库未覆盖 |
| `/home/xymo/projects/hytorch/runs/fused_sgd_cuda_ptx_01/` | 离线 PTX/cubin、原编译命令、编译日志和资源汇总 |
| `/home/xymo/projects/hytorch/runs/fused_sgd_cuda_01/figure-review-01/` | 六张原始图稿；最终调整图例后的图在本报告 figures 目录 |

六份 profiler 原文件的确切前缀（均位于 rawroot `runs/fused_sgd_cuda_01/`，每个前缀同时有 `.nsys-rep`、`.sqlite`、`.log`、`-export.log`）：

- `trace-update-baseline`、`trace-update-fused`、`trace-update-torch`。
- `trace-training-baseline`、`trace-training-fused`、`trace-training-torch`。

## GitHub 原始数据副本

[逐文件完整索引](results/raw/file-index.csv) 列出 533 个服务器文件及可上传副本；[补充图稿索引](results/figure-review-index.csv) 列出归档后保留的 6 张原图稿。`results/raw/` 原样复制 276 个文本记录，超过 1 MB 的文本以 `.gz` 无损压缩。GitHub 数据不是只有摘要；完整记录和原始事件 CSV 在以下目录。

- [GPU 正确性、逐次微基准、每步训练、所有配置/日志/命令](results/raw/fused_sgd_cuda_01/)。各训练目录保留 `config.json/summary.json/steps.jsonl/losses.jsonl`；每方法、模式和重复独立命名。
- [六份完整 Trace CSV 与逐条事件](results/raw/fused_sgd_cuda_analysis_01/)。压缩 CSV 可用标准 gzip 工具展开；CPU 分析脚本也可直接使用服务器未压缩原文件。
- [完整硬件监控与退出状态](results/raw/fused_sgd_cuda_monitor_01/)。
- [构建日志](results/raw/fused_sgd_cuda_build_01/)、[PTX 编译与资源日志](results/raw/fused_sgd_cuda_ptx_01/)。
- [主机测试/分析/归档日志及 XML](logs/)、[系统和软件环境](environment/)。

## 可直接复核的合并数据

| 文件 | 口径 |
|---|---|
| [correctness.csv](results/raw/fused_sgd_cuda_01/correctness/correctness.csv) | 303 条实际数组比较，绝对/相对误差、超差元素数、有限值、容差、原 NPZ 文件名 |
| [update-benchmark-raw.csv](results/update-benchmark-raw.csv) | 1800 次 Update，包含独立进程 repeat、模式、step、耗时 |
| [update-benchmark-per-repeat.csv](results/update-benchmark-per-repeat.csv) | 每个进程内 100 次调用的 mean/std/median/P95/min/max |
| [update-benchmark-summary.csv](results/update-benchmark-summary.csv) | 三个独立进程均值的分布及速度比 |
| [update-cross-method-correctness.csv](results/update-cross-method-correctness.csv) | 288 条同初值/梯度的跨方法最终参数比较 |
| [training-raw.csv](results/training-raw.csv) | 18 个独立进程训练结果，吞吐与同步阶段分别保留 |
| [training-summary.csv](results/training-summary.csv) | 三独立重复的每项统计 |
| [training-steps.csv](results/training-steps.csv)、[training-losses.csv](results/training-losses.csv) | 4608 正式 Step 的原阶段计时与 Loss |
| [training-numerical-comparison.csv](results/training-numerical-comparison.csv) | 所有最终权重、每步 Loss 的误差和是否通过 |
| [numerical-comparison.json](results/numerical-comparison.json) | 实际初始输入/参数逐元素一致性，数值通过与失败行 |
| [hardware-process-windows.csv](results/hardware-process-windows.csv) | 约秒级监控按整个子进程窗口统计，不是阶段精确硬件指标 |
| [launch-counts.csv](results/trace/launch-counts.csv) | 各 Trace/phase 的 Kernel、Launch API、Memcpy 数 |
| [trace-update-raw.csv](results/trace/trace-update-raw.csv) | 648 个 Update 的独立 host/GPU/API/同步/间隙指标 |
| [trace-update-summary.csv](results/trace/trace-update-summary.csv) | 每种 trace/mode 分别统计 mean/std/median/P95 |
| [metadata-host-raw.csv](results/trace/metadata-host-raw.csv)、[metadata-host-summary.csv](results/trace/metadata-host-summary.csv) | 元数据准备/native/ctypes NVTX 主机计时，嵌套范围不能相加 |
| [trace-phase-raw.csv](results/trace/trace-phase-raw.csv)、[trace-training-step-raw.csv](results/trace/trace-training-step-raw.csv) | correlation 归因后的 phase 与 Step 数据 |
| [independent-review.json](results/trace/independent-review.json) | CPU 独立核查所有 Launch 数、关联和梯度新地址 |

## 全部最终图

1. [独立 Update 同步墙钟与主机提交](figures/update-timing.png)
2. [实际 Kernel Launch 与纯 GPU 核时](figures/kernel-launches.png)
3. [DonkeyNet 未插桩吞吐](figures/training-throughput.png)
4. [另测同步 Forward/Backward/Update](figures/training-phase-timing.png)
5. [原 SGD、Fused、PyTorch Loss 与差值](figures/training-loss.png)
6. [correlation 归因的 Update 时间线](figures/update-trace-timeline.png)

全部由真实数据绘制，不是桌面截图。绘制代码和统计实现均在 [scripts](scripts/)；原始 CPU 分析命令和图稿修订记录见 [commands.md](commands.md) 与 [analysis-review.log](logs/analysis-review.log)。
