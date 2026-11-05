# 原始数据索引

报告目录：`/home/xymo/projects/hytorch/docs/experiments/cpu-training-pipeline/`。
服务器原始根：`/home/xymo/projects/hytorch/runs/cpu_pipeline_01/`。
[报告文件清单](results/artifact-index.csv)记录文件名与字节数，不使用哈希。

## 正式CPU测量及分项

- [初始grid逐次结果](results/raw-01/loader-grid-01/runs.csv)：90独立进程，各1024张；existing/repaired/inplace、workers0/1/2/4、异步prefetch1/2/4、三重复。
- [初始grid全部子命令](results/raw-01/loader-grid-01/commands.json)；各子目录含`summary.json`和256行`batches.csv`。
- [原型句柄对照](results/raw-02/stream-grid-01/runs.csv)：30独立进程，两版本×workers0/2/4×五重复。
- [集成句柄对照](results/raw-02/stream-integrated-01/runs.csv)：18独立进程，两版本×workers0/2/4×三重复。
- [原始归档1索引](results/raw-01/file-index.csv)：218服务器文件、217份可移植记录。
- [原始归档2索引](results/raw-02/file-index.csv)：137服务器文件、136份可移植记录。
- [按配置均值/SD](results/loader-summary.csv)、[初始增量及负值](results/incremental-comparisons.csv)、[句柄均值/SD](results/stream-summary.csv)、[句柄逐重复比值与墙钟加速](results/stream-comparisons.json)。
- [完整机器汇总](results/summary.json)：CPU124项、40配置真实数据、138次CPU性能；GPU指标空值/NOT_RUN。

## CPU Profiling、线程与队列

- [逐图分阶段](results/raw-01/profile-01/preprocess-stages.csv)：1024张，读取/解码/RGB/Crop/Resize/归一化CHW；与原处理逐元素相等。
- [原路径完整cProfile文本](results/raw-01/profile-01/serial-profile.txt)、[函数CSV](results/raw-01/profile-01/serial-profile.csv)。
- [文件句柄完整cProfile文本](results/raw-02/profile-stream-01/serial-profile.txt)、[函数CSV](results/raw-02/profile-stream-01/serial-profile.csv)。
- [原路径Queue/组装统计](results/raw-01/profile-01/queue-summary.csv)、[句柄路径Queue统计](results/raw-02/profile-stream-01/queue-summary.csv)。
- 两个profile目录内每个`w*-p*`子记录有`queues.csv`（每次put/get的线程、时长、队列深度）、`arrays.csv`（组装耗时及字节）、`samples.csv`（样本及Worker时长）。这些是独立插桩诊断，不参与正式吞吐。
- `serial.pstats`二进制分别保存在服务器`runs/cpu_pipeline_01/profile-01/`及`profile-stream-01/`，见上述文件索引。
- [满队列修复前](results/lifecycle-before.json)、[修复后](results/lifecycle-after.json)：同一合成用例单次观察。
- [pending无严格全局上界的实测](results/pending-observation.json)：结果容量2、pending99个小Batch，不冒充真实图像峰值。

## 正确性、异常与后续入口

- [最终CPU124项stdout/stderr](logs/host-regression-03.log)、[XML](results/host-regression-03.xml)、[实际命令/环境/时间](logs/host-regression-03.log.command.json)。
- [首次122通过/1失败](logs/host-regression-01.log)、[首次XML](results/host-regression-01.xml)，CUDA_PATH缺失；[补齐变量后123通过](logs/host-regression-02.log)。最初27项专项输出另见`logs/loader-tests-01.log`。
- [40配置真实图像/标签/划分逐元素验证](results/exact-data-03/exact-batches.csv)、[汇总](results/exact-data-03/summary.json)、[完整日志](logs/real-data-equality-03.log)。初始30配置通过保留在`exact-data-02/`。
- [首次校验中断说明](results/equality-attempt-01.json)及`logs/real-data-equality-01.log`；首次wrapper被中断，未生成command元数据。实际命令仅输出目录为`exact-data-01`，其余参数同后续重跑；不记为通过。
- [最终统一策略CPU验证](results/future-gpu-02/cpu-policy-validation.csv)、[126项未来计划](results/future-gpu-02/plan.json)、[CPU dry-run配置](results/future-dry-run-02/config.json)。早期未增加句柄选项的`future-gpu-01`与`future-dry-run-01`也保留。
- [环境](environment/system.txt)、[全部Python包](environment/python-packages.txt)、[时间线](timeline.md)、[命令](commands.md)。全部图见[figures](figures/)。

## 原始源码与历史数据

服务器`runs/cpu_pipeline_01/baseline_dataloaders.py`与`baseline_udacity_dataset.py`来自Git基线
`9909e546d9314e342663d75c4a10d1f46b4b7351`，未做哈希。原型/修复源码及重放脚本在本分支。
历史protocol/参考数组不覆盖：

- `/home/xymo/projects/hytorch/runs/udacity_real_prepared_01/protocol.json`
- `/home/xymo/projects/hytorch/runs/udacity_real_prepared_01/throughput-inputs.npz`
- `/home/xymo/datasets/udacity/data/driving_log.csv`和原始IMG。
- [4H原报告](../udacity-real-training/README.md)、[前一轮自主优化](../autonomous-optimization-6h/README.md)。

没有新GPU数据、模型权重、Nsight Trace或端到端训练成绩。所有未来GPU计划状态均为NOT_RUN。
