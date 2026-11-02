# 阶段4G完整数据索引

最终报告：[README.md](README.md)。最终分析：[results/acceptance-03](results/acceptance-03/summary.json)。所有GPU正式数据来自同一GPU0补测窗口；第一次失败独立保存。

## 服务器原始数据

项目入口 `/home/xymo/projects/hytorch` 指向 `/srv/lab-disks/disk2/users/xymo/projects/hytorch`；下表路径均相对于项目。完整逐文件绝对路径和字节数见[file-index.csv](raw-records/file-index.csv)，共21,430个文件、632份可移植原始记录。索引不包含哈希。

| 路径 | 内容 | Git可移植记录 |
|---|---|---|
| `runs/conv_dweight_build_01/` | 动态库、build.json、完整编译/资源信息 | [build](raw-records/conv_dweight_build_01) |
| `runs/conv_dweight_cpu_01/` | CPU C++算术模型、输入/样本NPZ、33,066比较 | [CPU](raw-records/conv_dweight_cpu_01) |
| `runs/conv_dweight_parser_cpu_01/` | 对4F历史Trace的CPU解析验证，非4G GPU成绩 | [parser](raw-records/conv_dweight_parser_cpu_01) |
| `runs/conv_dweight_window_01/` | 首次失败窗口，276通过/1失败、清理接口报错、输入/比较/监控 | [首次窗口](raw-records/conv_dweight_window_01) |
| `runs/conv_dweight_window_02/guard/` | 最终窗口空闲检查、329次GPU状态/进程监控、启动/结束记录 | [guard](raw-records/conv_dweight_window_02/guard) |
| `runs/conv_dweight_window_02/results/` | 76项完整命令、测试、算子、训练、17份Nsight原始文件 | [最终窗口](raw-records/conv_dweight_window_02/results) |
| `runs/conv_dweight_analysis_01/` | GPU释放后导出的17个SQLite及correlation归因CSV/命令 | [导出/归因](raw-records/conv_dweight_analysis_01) |

`.so/.nsys-rep/.sqlite/.npz`及CPU算术模型二进制/源码等不复制到Git，以索引中的服务器原始文件为准。文本`.log/.json/.jsonl/.csv/.xml/.txt`完整复制；单文件超过1MB时无损gzip，Git文件名在file-index中逐项标注。未删除或截断原始数值及失败日志。

## GPU任务、正确性与算子

- [完整76项执行命令及起止/返回码](raw-records/conv_dweight_window_02/results/commands.json)，[补测计划](results/gpu-plan-02.json)。
- [窗口开始](raw-records/conv_dweight_window_02/results/window-start.json)、[窗口结束与reset=0](raw-records/conv_dweight_window_02/results/window.json)、[guard完成](raw-records/conv_dweight_window_02/guard/completion.json)、[释放后只读检查](environment/gpu-release-02.log)。
- CPU回归：[host-tests-03.xml](results/host-tests-03.xml)、[完整stdout/stderr](logs/host-tests-03.log)；较早01/02亦保留。
- GPU新增：[277项XML](raw-records/conv_dweight_window_02/results/correctness-new.xml)、[完整日志](raw-records/conv_dweight_window_02/results/correctness-new.log)、[3831组误差](raw-records/conv_dweight_window_02/results/correctness-new-arrays/comparisons.csv)。
- GPU回归：[446项XML](raw-records/conv_dweight_window_02/results/correctness-regression.xml)、[完整日志](raw-records/conv_dweight_window_02/results/correctness-regression.log)。各测试生成的NPZ留服务器，包含exact producer inputs及actual/reference。
- [132个预先固定的形状](results/predeclared-shapes.json)、[所有形状速度比](results/acceptance-03/micro/comparisons.csv)、[全部普通计时](results/acceptance-03/micro/latencies.csv)、[全部独立CUPTI计时](results/acceptance-03/micro/trace-calls.csv)、[三进程统计](results/acceptance-03/micro/summary.csv)、[2376组误差](results/acceptance-03/micro/errors.csv)、[内存记录](results/acceptance-03/micro/memory.csv)。
- 原始九个微基准进程各自保留在`runs/conv_dweight_window_02/results/micro-{original,fused,torch}-r{1,2,3}/`；九个Trace为同根`trace-micro-{original,fused,torch}-r{1,2,3}.nsys-rep`，各包含132×30调用。

## 训练及Nsight

- [48组训练的原始墙钟/吞吐/阶段值](results/acceptance-03/training/training-raw.csv)。
- [12,288步内部计时](results/acceptance-03/training/training-steps.csv)、[全部Loss](results/acceptance-03/training/training-losses.csv)、[780组Loss/最终权重误差](results/acceptance-03/training/training-errors.csv)、[初始化/输入逐数组一致性](results/acceptance-03/training/checks.json)。
- [均值/SD/范围](results/acceptance-03/training/training-summary.csv)、[本窗口全部收益](results/acceptance-03/training/speedups.csv)、[长期进展表CSV](results/acceptance-03/training/progress-table.csv)。
- 每个训练目录：`runs/conv_dweight_window_02/results/training-{donkey,spatial}-{throughput,phases}-{A,F,I,E}-r{1,2,3}/`，含config/summary、steps/losses、初始/最终权重及实际输入NPZ；同级`.log`保存终端输出。NPZ只留服务器，其他原始记录同步Git。
- 八份训练Trace：`runs/conv_dweight_window_02/results/trace-training-{donkey,spatial}-{A,F,I,E}.nsys-rep`；每份10步预热、16步正式插桩，不能代替256步吞吐成绩。
- 导出SQLite位于`runs/conv_dweight_analysis_01/trace-*.sqlite`，相应无扩展名子目录中有原始Kernel/API/NVTX事件、operator/phase/launch统计、Memcpy/Memset记录。Git对应[归因目录](raw-records/conv_dweight_analysis_01)，大CSV使用`.gz`。
- [完整phase范围统计](results/acceptance-03/trace-phases.csv)、[每层每次dWeight Kernel/Copy/Im2col](results/acceptance-03/dweight-components.csv)、[每层均值](results/acceptance-03/dweight-layers.csv)、[Backward每次时间跨度/并集/间隙](results/acceptance-03/backward-windows.csv)、[每step总计](results/acceptance-03/trace-totals.csv)、[17Trace关联检查](results/acceptance-03/trace-checks.json)。

## 修正与版本

- [首次GPU错误和用户授权补测](results/window-01-failure.json)；[首次完整终端输出](logs/gpu-window-01.log)与[最终补测](logs/gpu-window-02.log)分开。
- `results/acceptance-01`为首版离线统计；PyTorch E的`backward-windows.csv`和派生`trace-totals.csv`的window核时/间隙漏算工作线程，**这些字段作废**，使用acceptance-03。原始phase-summary、数值、微基准和训练吞吐未受影响。保留01数据和`figures/final-01`用于追溯。
- [postprocess-02.log](logs/postprocess-02.log)保存首次重汇总拒绝覆盖旧日志的退出；无GPU工作，无数值变化。`acceptance-02`空输出目录不提交。
- [最终离线重汇总](logs/postprocess-03.log)、[汇总检查](logs/acceptance-checks-03.log)、[最终绘图](logs/figures-03.log)。
- 每次CPU/外层GPU命令的`logs/*.command.json`保存准确argv、环境、起止时间、返回码；统一入口的每项GPU命令见上文commands.json。

## 全部正式图像

| 图 | 链接 |
|---|---|
| 原/新/PyTorch dWeight延迟 | [01](figures/final-03/01-dweight-latencies.png) |
| 跨形状加速与退化 | [02](figures/final-03/02-cross-shape-speedups.png) |
| Im2col/Copy/Launch变化 | [03](figures/final-03/03-kernel-copy-launch-counts.png) |
| 两模型吞吐及三次独立测量 | [04](figures/final-03/04-matched-throughput.png) |
| 增量与累计收益 | [05](figures/final-03/05-incremental-and-cumulative.png) |
| 与PyTorch差距 | [06](figures/final-03/06-pytorch-gap.png) |
| CPU/GPU时间线 | [07](figures/final-03/07-backward-timeline.png) |
| 独立同步阶段耗时 | [08](figures/final-03/08-synchronized-phases.png) |
| 真实逐步Loss | [09](figures/final-03/09-losses.png) |
| 工作区大小 | [10](figures/final-03/10-workspaces.png) |

图均为实测日志/CSV/CUPTI可视化，不冒充终端截图。绘图源码为[scripts/plot_results.py](scripts/plot_results.py)。
