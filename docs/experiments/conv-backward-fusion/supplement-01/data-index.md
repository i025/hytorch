# 阶段 4D GPU 补测数据索引

服务器项目别名为 `/home/xymo/projects/hytorch`，真实根目录为 `/srv/lab-disks/disk2/users/xymo/projects/hytorch`。下列相对路径均相对此实验目录；完整服务器文件路径、大小及 Git 对应路径分为 [测量索引](results/raw-measurements-02/file-index.csv)、[补采索引](results/raw-traces-01/file-index.csv)、[最终解析索引](results/raw-analysis-01/file-index.csv)。没有计算任何文件哈希。最终状态见 [status.json](results/status.json)，监控过程见 [monitor-summary.json](results/monitor-summary.json)。

## 已完成的正式验收与测量

| 数据 | 可管理的原始记录 / 派生数据 |
|---|---|
| 完整环境 | [environment.txt](environment.txt) |
| GPU 6 的 92 项回归 stdout/stderr | [gpu-tests.log](results/raw-measurements-02/conv_backward_fusion_wait_02/gpu-attempt-01/gpu-tests.log) |
| pytest JUnit | [gpu-tests.xml](results/raw-measurements-02/conv_backward_fusion_wait_02/gpu-attempt-01/gpu-tests.xml) |
| 新 Kernel 三方逐数组误差 | [correctness/comparisons.csv](results/raw-measurements-02/conv_backward_fusion_wait_02/gpu-attempt-01/correctness/comparisons.csv)；[汇总](results/gpu-correctness-02.json) |
| 每个真实执行命令、起止时间及返回值 | [commands.json](results/raw-measurements-02/conv_backward_fusion_wait_02/gpu-attempt-01/commands.json) |
| 单算子 2160 次原始测量 | [micro-latencies.csv](results/acceptance-01/micro-latencies.csv)；[各重复统计](results/acceptance-01/micro-repeats.csv)；[三次均值统计](results/acceptance-01/micro-summary.csv) |
| 单算子临时显存及请求数 | [micro-memory.csv](results/acceptance-01/micro-memory.csv) |
| 单算子全部误差 | [micro-errors.csv](results/acceptance-01/micro-errors.csv) |
| 36 个独立训练进程 | [training-raw.csv](results/acceptance-01/training-raw.csv)；[training-summary.csv](results/acceptance-01/training-summary.csv) |
| 9216 个真实训练 Step | [training-steps.csv](results/acceptance-01/training-steps.csv)；[training-losses.csv](results/acceptance-01/training-losses.csv) |
| 714 条数值比较及初始化/输入核对 | [training-errors.csv](results/acceptance-01/training-errors.csv)；[training-checks.json](results/acceptance-01/training-checks.json)；[误差汇总](results/acceptance-01/training-error-summary.json) |
| 长期六版本表和全部收益 | [progress-table.csv](results/acceptance-01/progress-table.csv)；[speedups.csv](results/acceptance-01/speedups.csv) |
| 正式测量窗口的 GPU 监控 | [gpu-load.jsonl](results/raw-measurements-02/conv_backward_fusion_wait_02/guard-attempt-01/gpu-load.jsonl)；[进程窗口汇总](results/hardware-process-windows-02.csv) |
| A 路径无本项目 Native Kernel 的检查 | [native-baseline-check.json](results/native-baseline-check.json)；基于第二窗口已完成的 A 全 Trace |
| 第二窗口已完成 A/B/C/D 的部分 Trace 指标 | [partial-traces-02.csv](results/partial-traces-02.csv)，只有此窗口内的插桩数据，F/E 未完成 |
| 最终完整六版本 Nsight 指标（GPU0） | [trace-training-summary.csv](results/trace-summary-02/trace-training-summary.csv) |
| Conv2–5 实际 dInput 父范围及核时 | [逐范围原始值](results/acceptance-01/gradient-components.csv)；[逐层汇总](results/trace-summary-02/trace-dinput-summary.csv) |
| 三微基准的实际 Kernel 数量/核时 | [逐调用](results/acceptance-01/micro-kernel-launches.csv)；[汇总](results/trace-summary-02/trace-micro-summary.csv) |
| 插桩数值对照（119条） | [trace-numerical-errors.csv](results/trace-summary-02/trace-numerical-errors.csv)；[初始化/输入](results/trace-summary-02/trace-checks.json) |
| Backward Kernel 类别与提交间隙 | [kernel-categories.csv](results/acceptance-01/kernel-categories.csv)；[backward-windows.csv](results/acceptance-01/backward-windows.csv)，E 同线程窗口不能代表工作线程全部核时，见汇总的空值说明 |
| 完整调用的分配请求（插桩） | [allocation-requests.csv](results/acceptance-01/allocation-requests.csv)，非分配器峰值或物理 cudaMalloc 次数 |

各独立进程的 config/summary、每步 JSONL、完整终端日志也在 `results/raw-measurements-02/conv_backward_fusion_wait_02/gpu-attempt-01/`。重复/方法不合并，较大的文本仅压缩为 `.gz`，不截断原始值。config 中的旧指纹字段为 null / DISABLED，训练命令均使用 `--no-fingerprints`。

## 完整服务器原始文件

- 首次 GPU0 中断窗口：`/home/xymo/projects/hytorch/runs/conv_backward_fusion_wait_01/`。
- 正式 GPU6 窗口：`/home/xymo/projects/hytorch/runs/conv_backward_fusion_wait_02/`。
- 第二窗口已完成的 A/B/C/D Trace 离线解析：`/home/xymo/projects/hytorch/runs/conv_backward_fusion_interrupted_trace_analysis_01/`。
- GPU6 专用空等记录（随后按用户指示取消）：`/home/xymo/projects/hytorch/runs/conv_backward_fusion_wait_03/`。
- 任意空闲 A100 的后续监控：`/home/xymo/projects/hytorch/runs/conv_backward_fusion_wait_04/`。
- 最终完整 GPU0 原始 Trace：`/home/xymo/projects/hytorch/runs/conv_backward_fusion_wait_05/gpu-attempt-01/`，包含 `trace-training-{A,B,C,D,F,E}.nsys-rep/.sqlite` 与 `trace-micro-{original,fused,torch}.nsys-rep/.sqlite`。
- 最终完整 correlationId 解析：`/home/xymo/projects/hytorch/runs/conv_backward_fusion_trace_analysis_01/`，每份 Trace 有 kernel-events、cuda-api-events/summary、operator-hotspots、NVTX、memcpy、gap CSV 和归因说明。
- 第四窗口可恢复 F 的审计解析：`/home/xymo/projects/hytorch/runs/conv_backward_fusion_recovered_trace_analysis_04/`。
- 复用的 SM80 动态库与编译资源原文：`/home/xymo/projects/hytorch/runs/conv_backward_fusion_build_01/`。

NPZ 输入、初始/最终权重和全部中间误差数组保留各原始进程目录；`.nsys-rep`、SQLite、编译二进制均留服务器，Git 仅上传其索引、导出数据和日志。第二窗口中断 F 没有生成有效 `.nsys-rep`；第四窗口可恢复 F 仅作审计，最终第五窗口9份均完整。

## 已生成的真实数据图

最终图共11张，全部标注为真实数据可视化；GPU6 普通测量与 GPU0 Nsight 属于不同窗口，各自匹配比较：

- [原/新/PyTorch Event 延迟](figures-final-02/backward-latency.png)
- [实际 Copy/GEMM/Col2im/Gather 数量](figures-final-02/kernel-counts.png)
- [各层纯设备 dInput 核时，均退化](figures-final-02/dinput-kernel-time.png)
- [Conv2–5 完整操作 Event/墙钟比](figures-final-02/per-layer-speedup.png)
- [六版本训练吞吐](figures-final-02/training-throughput.png)
- [相对 A 累计吞吐比](figures-final-02/cumulative-speedup.png)
- [逐阶段增量](figures-final-02/incremental-gains.png)
- [D/F 共享横轴时间线](figures-final-02/gpu-timeline.png)
- [Loss 与数值误差](figures-validation-01/loss-and-correctness.png)
- [原始测量波动](figures-validation-01/latency-variation.png)
- [独立调用临时分配](figures-validation-01/dinput-temporary-memory.png)

以下是等待期先行生成的图，保留原始版本；最终时间线改用共享横轴，另补纯设备核时图，初版完整图保留在 [figures-final-01](figures-final-01/)：

- [原/新/PyTorch dInput 与完整 backward](figures-measurements-01/backward-latency.png)
- [Conv2–Conv5 完整 backward 比值](figures-measurements-01/per-layer-speedup.png)
- [六版本训练吞吐，含每次重复](figures-measurements-01/training-throughput.png)
- [相对 A 的累计吞吐比](figures-measurements-01/cumulative-speedup.png)
- [各阶段增量](figures-measurements-01/incremental-gains.png)
- [逐步 Loss 与最终数值差异](figures-validation-01/loss-and-correctness.png)
- [所有 dInput Event 测量的波动](figures-validation-01/latency-variation.png)
- [独立 dInput 调用临时分配峰值](figures-validation-01/dinput-temporary-memory.png)

## 中断与后处理失败

GPU0 与 GPU6 的外部客户端中断、原始 guard 监控/终端输出均已归档。GPU0 的不完整训练不用于正式收益。此前 CPU 记录不变。

- `hardware-summary-02.log`：对中断 manifest 的 unfinished 命令读取 finished_at 失败；另存已完成的 45 条测量命令后，`hardware-summary-03.log` 成功。
- `archive-measurements-01.log`：绝对别名 `/home/...` 与解析后根目录 `/srv/...` 的路径前缀不同，旧归档工具报 ValueError。保留该日志和首个已复制文件；使用仓库相对路径、新目录重跑，`archive-measurements-02.log` 成功，索引 3048 个原始文件、复制 502 个文本记录。
- `native-baseline-check-01-invalid-pattern.json`：首次临时检查的正则边界误转义；仅作失败检查记录。有效检查经 Native 名称正控后保存为 `native-baseline-check.json`。正式 Trace 分析脚本的正则无此问题。

这些后处理问题没有改动生产计算路径或数值容差，也不是 GPU 正确性失败。

- `results/trace-summary-review-01.json`：首版 E 同线程 NVTX 包络未包含工作线程 Kernel，不可当作完整 PyTorch Backward。旧表保留于 `acceptance-01/trace-training-summary.csv`；有效替代为 `trace-summary-02/trace-training-summary.csv`，相关 E 包络字段为空且注明不适用。phase 核时/数量和所有 D/F 数据不变。
- `results/guard-review-04.json`：第四窗口 guard 误认自身 Nsight 目标 PID。完整 F 可恢复且16步齐全，但最终全部 profiling 比较采用第五窗口，未关闭 guard 或修改其他优化模块。
