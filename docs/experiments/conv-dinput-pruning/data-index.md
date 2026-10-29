# 阶段 4C 原始数据索引

服务器仓库 `/home/xymo/projects/hytorch` 指向 `/srv/lab-disks/disk2/users/xymo/projects/hytorch`。[file-index.csv](results/raw/file-index.csv) 列出 **1363 个原始文件**的绝对路径、字节数和 Git 副本位置，没有内容哈希。原始数据约 1.57 GB；其中 **269 个文本记录**复制到 Git，超过 1 MB 的文本以 `.gz` 无损压缩。

## 服务器原始目录

| 绝对路径 | 内容 |
|---|---|
| /home/xymo/projects/hytorch/runs/conv_dinput_cpu_tests_01/ | CPU 439 条数值比较、所有实际数组 |
| /home/xymo/projects/hytorch/runs/conv_dinput_gpu_01/ | GPU 测试；30 次训练配置/初始输入/权重/逐步记录；5 份 Nsight 和 SQLite；41 条命令及输出 |
| /home/xymo/projects/hytorch/runs/conv_dinput_trace_analysis_01/ | 全部 Trace 事件 CSV、NVTX、CUDA API/Kernel/拷贝、算子汇总和解析日志 |
| /home/xymo/projects/hytorch/runs/conv_dinput_monitor_01/ | GPU 0 被占用、未启动实验的检查记录 |
| /home/xymo/projects/hytorch/runs/conv_dinput_monitor_02/ | GPU 2 的空闲检查、映射、140 条监控、任务终端输出、成功退出 |
| /home/xymo/projects/hytorch/runs/conv_dinput_gpu_01/figure-drafts-01/ | 保留的六张初稿；正式 Kernel 图已修正 warmup 统计范围 |

五份 Nsight 原文件为 rawroot 下 `trace-A.nsys-rep` 至 `trace-E.nsys-rep`，各有同名前缀 `.sqlite`、`.log`、`-export.log` 及训练输出目录。Native 动态库复用 `/home/xymo/projects/hytorch/runs/fused_sgd_cuda_build_01/libhytorch_cuda.so`，本轮未修改它或旧实验。

## GitHub 原始副本

- [CPU 数值 CSV](results/raw/conv_dinput_cpu_tests_01/comparisons.csv)。
- [GPU 测试与全部训练/Trace 命令、配置、日志](results/raw/conv_dinput_gpu_01/)。`correctness/comparisons.csv` 为新 14 项 GPU 测试的 439 条比较；`fused-regression/correctness.csv` 为既有 Fused 回归。
- [五份完整 Trace CSV](results/raw/conv_dinput_trace_analysis_01/)；大 CSV 可按标准 gzip 解压，服务器原始 CSV 无需解压即可由脚本读取。
- [GPU 0 未启动记录](results/raw/conv_dinput_monitor_01/)、[GPU 2 全部监控和终端输出](results/raw/conv_dinput_monitor_02/)。
- [完整环境](environment/)、[CPU/GPU 分析和测试日志、XML、修复前后记录](tests/)。

## 核心结果文件

| 文件 | 含义 |
|---|---|
| [training-raw.csv](results/training-raw.csv) | 30 次独立训练，完整墙钟、吞吐、同步阶段、Loss、dispatch 和源路径 |
| [training-summary.csv](results/training-summary.csv) | 每方法/模式各 3 次的 mean/std/median/P95/min/max |
| [training-steps.csv](results/training-steps.csv)、[training-losses.csv](results/training-losses.csv) | 7680 正式 Step 的阶段时间和 Loss |
| [progress-table.csv](results/progress-table.csv) | A–E 长期性能表 |
| [speedups.csv](results/speedups.csv) | B/A、C/B、D/C、D/A、D/E 的吞吐变化、耗时加速比和耗时下降 |
| [numerical-errors.csv](results/numerical-errors.csv)、[numerical-summary.json](results/numerical-summary.json) | 相对 A、及 D 相对 C 的每参数/每步 Loss 误差；初始权重/输入一致性 |
| [trace-phases.csv](results/trace-phases.csv) | 各组 phase 与 outside_phase 原始统计；正式每步必须排除 outside_phase |
| [gradient-components.csv](results/gradient-components.csv) | 五层 dInput/dWeight/dBias 的范围数、Kernel 数、GPU 核时、主机时长 |
| [layout-kernels.csv](results/layout-kernels.csv) | 按层/梯度组件分组的 Col2im、Im2col、Tensordot、Copy；分类有包含关系 |
| [first-dinput-events.csv](results/first-dinput-events.csv) | C 首层 dInput 的 48 个实际 GPU 事件与 correlation/API ID |
| [native-kernel-events.csv](results/native-kernel-events.csv) | Trace 中本项目 Native Kernel 明细，A 无记录 |
| [trace-summary.csv](results/trace-summary.csv)、[trace-checks.json](results/trace-checks.json) | 五 Trace 的全量/正式计数、包络与归因检查 |
| [event-scope-review.json](results/event-scope-review.json) | 原 SQLite 总数和每个正式 backward 范围的复核 |
| [hardware-process-windows.csv](results/hardware-process-windows.csv) | 约秒级监控按进程窗口统计，不是训练阶段精确利用率 |

## 六张最终图

1. [五版本训练吞吐](figures/five-version-throughput.png)
2. [相对 A 的累计收益](figures/cumulative-speedup.png)
3. [每阶段增量收益](figures/incremental-gains.png)
4. [剪枝前后 Backward 两种时间口径](figures/backward-time.png)
5. [实际 Kernel 数与逐层 Col2im](figures/kernel-and-col2im.png)
6. [Loss 与数值正确性](figures/loss-and-correctness.png)

这些都是实际数据可视化，不是终端截图。绘图和分析源代码在 [scripts](scripts/)，复现入口见 [commands.md](commands.md)。失败高阶复现、GPU 被占用检查、退化重复和原图稿全部保留；不上传原始训练张量、模型权重、大型 profiler 二进制或动态库。
