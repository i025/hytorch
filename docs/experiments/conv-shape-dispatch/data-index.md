# 原始数据、导出和图像索引

服务器项目 `/home/xymo/projects/hytorch` 实际指向 `/srv/lab-disks/disk2/users/xymo/projects/hytorch`。下列服务器相对路径均以该根目录为准，不删除或覆盖原始文件。

## 可在 GitHub 审查的数据

| 路径 | 内容 |
| --- | --- |
| [environment/system.log](environment/system.log) | 操作系统、CPU、8卡初始状态、完整pip列表、GCC/nvcc/Nsight版本 |
| [environment/gpu-windows.json](environment/gpu-windows.json) | 各窗口监测范围、外部进程、显存采样峰值 |
| [results/predeclared-shapes-04.json](results/predeclared-shapes-04.json) | 全部100形状，32/20/24/24集合；旧3份manifest也保留 |
| [results/calibration-01/](results/calibration-01/) | 原始30次延迟、进程波动、误差、内存、首轮冻结规则 |
| [results/held-out-01/](results/held-out-01/) | 第一独立集，h10未过5%门槛 |
| [results/refined-policy-01/](results/refined-policy-01/) | 第二规则完整搜索、冻结参数 |
| [results/validation-2-01/](results/validation-2-01/) | 全新24形状，v10/v13拒绝记录 |
| [results/pointwise-frozen-policy.json](results/pointwise-frozen-policy.json) | 最终策略先冻结、未读取最后验证结果的记录 |
| [results/pointwise-validation-01/](results/pointwise-validation-01/) | 最后全新24形状、14Gather通过 |
| [results/actual-auto-01/](results/actual-auto-01/) | 每次实际选择器开销在内的四方法复测，含未选形状的退化 |
| [results/training-01/](results/training-01/) | 60个训练进程、15,360步、Loss/权重误差、三重复波动、同步phase、长期进展表 |
| [results/profiles-01/](results/profiles-01/) | 23Trace关联检查、逐形状Kernel/Launch/核时、主机NVTX、真实调度、分配请求 |
| [results/failure-inputs-01/](results/failure-inputs-01/) | 首次类型错误首批输入/初值的离线重建索引及provenance；不是现场捕获 |
| [results/report-numbers.json](results/report-numbers.json) | 从CSV派生的报告关键数字和对应范围 |
| [results/raw-01/file-index.csv](results/raw-01/file-index.csv) | **9,298份服务器文件**：绝对路径、大小、Git便携副本位置；无哈希 |
| [results/raw-profiles-01/file-index.csv](results/raw-profiles-01/file-index.csv) | **338份完整解析文件**的服务器路径及Git副本 |
| [logs/](logs/) | 外层完整stdout/stderr、命令/环境/退出码/开始结束时间 |
| [scripts/](scripts/) | 调度拟合、测量、guard、分析、绘图和报告数字派生源码 |

`raw-01`复制882份文本原始记录，`raw-profiles-01`复制338份完整导出/解析记录；超过1 MB的文本gzip，未改原始数值或丢弃行。NPZ、Nsight、SQLite、动态库只在服务器保留，索引里标明 server original only。大CSV压缩后仍可完整解压审阅，不只上传摘要。

## 服务器原始实验目录

| 服务器路径 | 状态及用途 | Git便携原始记录 |
| --- | --- | --- |
| `runs/conv_shape_calibration_guard_01/` | 起跑前GPU被占用，未执行 | [guard](results/raw-01/conv_shape_calibration_guard_01/) |
| `runs/conv_shape_calibration_wait_01/` | 外部任务进入导致中断；部分校准不进入正式拟合 | [部分记录](results/raw-01/conv_shape_calibration_wait_01/) |
| `runs/conv_shape_calibration_wait_02/attempt-01/results/` | 完整32形状×3方法×3重复 | [校准原始记录](results/raw-01/conv_shape_calibration_wait_02/) |
| `runs/conv_shape_heldout_wait_01/attempt-01/results/` | 20形状独立验证 | [原始记录](results/raw-01/conv_shape_heldout_wait_01/) |
| `runs/conv_shape_measurements_wait_01/attempt-01/results/` | 初次GPU132通过/4失败，未开始正式训练 | [失败原始记录](results/raw-01/conv_shape_measurements_wait_01/) |
| `runs/conv_shape_validation2_wait_01/attempt-01/results/` | 第二规则全新24验证 | [原始记录](results/raw-01/conv_shape_validation2_wait_01/) |
| `runs/conv_shape_pointwise_wait_01/attempt-01/results/` | 最终pointwise规则全新24验证 | [原始记录](results/raw-01/conv_shape_pointwise_wait_01/) |
| `runs/conv_shape_measurements_wait_02/attempt-01/results/` | 完整184项中间回归、实际auto复测、两个模型60个正式训练进程 | [全部训练/算子原始文本](results/raw-01/conv_shape_measurements_wait_02/) |
| `runs/conv_shape_traces_wait_01/` | 严格零显存等待器取消，未执行GPU | [监测/取消记录](results/raw-01/conv_shape_traces_wait_01/) |
| `runs/conv_shape_traces_wait_02/attempt-01/results/` | 完整23份Nsight及2项路由测试 | [命令/日志/配置/数据](results/raw-01/conv_shape_traces_wait_02/) |
| `runs/conv_shape_final_tests_wait_01/attempt-01/results/` | 最终188项完整GPU回归，含高阶路径 | [完整比较与XML](results/raw-01/conv_shape_final_tests_wait_01/) |
| `runs/conv_shape_trace_analysis_01/` | 23份SQLite的完整API/Kernel/NVTX导出，correlation归因 | [完整导出](results/raw-profiles-01/conv_shape_trace_analysis_01/) |

每个完整窗口的 `guard/` 保留起跑快照、`launch.json`、逐秒`gpu-load.jsonl`、完整`terminal.log`及退出状态。训练根 `commands.json` 保存每个子进程argv和开始/结束时间。正式训练目录为 `training-{donkey|spatial}-{throughput|phases}-{A|D|F|G|E}-r{1|2|3}`，包含config/summary/steps/losses/hardware及初始权重、输入、最终权重NPZ。正式训练关闭轮询监控以保持计时一致；hardware.jsonl为空/summary对应字段为null，并非采集失败或伪造GPU数据。独立guard硬件采样另留。

`auto-shapes/{original|fused|torch|auto}-r{1|2|3}`下保留24形状的每次延迟、内存请求、误差和 `pXX-arrays.npz`。校准/验证目录有对应c/h/v/p NPZ。最终回归的 `shape-correctness/`、`legacy-correctness/`、`pruning-correctness/`、`sgd-correctness/`保存逐值比较与NPZ（以实际索引为准），首次失败的比较原件也不删除。

## 原始 Nsight 文件

绝对根目录：

`/home/xymo/projects/hytorch/runs/conv_shape_traces_wait_02/attempt-01/results/`

- `trace-training-{donkey|spatial}-{A|D|F|G|E}.nsys-rep`，各自同名`.sqlite`。
- `trace-shapes-{calibration|held-out|validation-2|validation-final}-{original|fused|torch}.nsys-rep`，各自同名`.sqlite`。
- `trace-shapes-validation-final-auto.nsys-rep`及同名`.sqlite`。

23份均有同名`.log`和`-export.log`、完整commands.json。完整解析为 `/home/xymo/projects/hytorch/runs/conv_shape_trace_analysis_01/trace-.../`：`kernel-events.csv`、`cuda-api-events.csv`、`nvtx-events.csv`、Memcpy/Memset、phase/operator/API/kernel汇总、活动间隙和schema。公开导出按原有解析器将进程/线程编号替换为稳定标签，原始SQLite保持不变。

首次类型错误的离线重建文件为 `/home/xymo/projects/hytorch/runs/conv_shape_spatial_failure_inputs_01/reconstructed-first-batch.npz`；provenance及脚本记录未做GPU重放，不冒充原始抓取。

现有4D库 `/home/xymo/projects/hytorch/runs/conv_backward_fusion_build_01/libhytorch_cuda.so` 不上传、不改动；已有编译记录仍在原目录。本轮无新构建或模型权重发布。

## 图像与来源

全部10张PNG在 [figures/](figures/)，逐张链接见[报告](README.md#实测图像)。`scripts/plot_results.py`只读本轮CSV、Nsight导出和raw-location.json。图01来自四套comparisons；图02来自actual-auto；图03–07来自training-01；图08–10来自profiles-01及完整Nsight导出。图09特意并列不同计时口径，不将独立运行的数值相减解释为精确分解。没有真实桌面截图；全部明确为数据可视化。
