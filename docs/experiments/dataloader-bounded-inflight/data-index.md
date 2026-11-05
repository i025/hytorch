# 原始数据索引

报告根：`/home/xymo/projects/hytorch/docs/experiments/dataloader-bounded-inflight/`。
服务器原始根：`/home/xymo/projects/hytorch/runs/dataloader_bounded_01/`。
[完整报告文件名/字节清单](results/artifact-index.csv)不使用文件哈希。

## 输入与源码版本

- [输入协议](results/input-protocol.json)：全部1024个真实样本索引、Batch及预处理、合成数据公式、延迟和规模。
- `/home/xymo/projects/hytorch/runs/udacity_real_prepared_01/protocol.json`与`throughput-inputs.npz`：历史划分及真实参考数组，原文件不改动。
- `/home/xymo/datasets/udacity/data/driving_log.csv`及`IMG/`：已有真实数据，只读不上传。
- [已提交的完整历史protocol](../udacity-real-training/raw-preparation-01/udacity_real_prepared_01/protocol.json)。
- 原DataLoader快照`runs/dataloader_bounded_01/baseline_dataloaders.py`由`git show dc77d06:hytorch/dataloaders.py`生成。
- 测试及性能所用新DataLoader对应提交`29fa065a5fd136221f760e764b05f8a22171696a`；最终结果提交不再修改生产代码。
- [合成输入实现](scripts/common.py)、[确定性回归样本](../../../tests/test_dataloader_bounded.py)。无需创建大体积合成数组文件。

## 未插桩正式性能

- [132次匹配原值](results/raw-01/benchmark-01/runs.csv)，每配置3次独立进程，不删慢轮次。
- [4个Epoch长度观察](results/raw-01/benchmark-01/scaling.csv)，每个端点单次，与三重复主实验分开。
- [全部实际子命令](results/raw-01/benchmark-01/commands.json)。
- 每个run子目录有`summary.json`（时间、Git版本、样本、配置、CPU、RSS、吞吐等）和`batches.csv`（每Batch next墙钟）。
- [按配置均值/SD](results/timing-summary.csv)、[吞吐变化/墙钟加速/内存差值](results/comparisons.csv)。
- [完整stdout/stderr](logs/benchmark-01.log)、[开始/结束/环境/返回值](logs/benchmark-01.log.command.json)。

## 独立插桩诊断

- [36种Queue/pending汇总](results/raw-01/diagnostics-01/summary.csv)。18种Bounded配置均通过窗口断言，18种基线对照保留。
- [全部诊断子命令](results/raw-01/diagnostics-01/commands.json)。
- 每配置目录保存`queue-events.csv`（操作、线程、耗时、深度、block及成功状态，包含失败尝试）、`pending-events.csv`（插入时间、seq、pending数量/字节、提交/交付数）、`batches.csv`和`summary.json`。
- [诊断完整日志](logs/diagnostics-01.log)、[命令及环境元数据](logs/diagnostics-01.log.command.json)。
- 该组计时为独立插桩口径，不能替代未插桩吞吐，也不能相加当作不重叠CPU计算时间。

## 正确性、失败与环境

- [原实现4项失败](logs/bound-before-01.log)、[XML](results/bound-before-01.xml)、[实际命令](logs/bound-before-01.log.command.json)。9项未选择，不计为通过。
- [修改后首轮36专项](logs/loader-after-01.log)、[XML](results/loader-after-01.xml)。
- [最终139项CPU回归](logs/cpu-regression-01.log)、[XML](results/cpu-regression-01.xml)、[命令及环境](logs/cpu-regression-01.log.command.json)。
- [20种真实数据逐元素结果](results/real-correctness-01/batches.csv)、[状态](results/real-correctness-01/summary.json)、[完整日志](logs/real-correctness-01.log)。
- [机器最终汇总](results/summary.json)、[环境](environment/system.txt)、[Python包](environment/python-packages.txt)。
- 元数据采集首次系统Python失败的说明在[日志](logs/environment-attempt-01.log)，部分输出保留在`environment/system-attempt-01.txt`；正式环境改用既有venv，未安装依赖。
- [命令](commands.md)、[时间线](timeline.md)、[全部PNG](figures/)、[绘图脚本](scripts/analyze.py)。

[逐文件服务器原始路径与Git副本索引](results/raw-01/file-index.csv)：423个服务器文件，423份可移植记录，全部保留。
本轮无新GPU数据、模型权重、二进制Trace或其他功能实验。发布验证日志仅保留服务器
`/home/xymo/projects/hytorch/runs/dataloader_bounded_publication_01/`。
