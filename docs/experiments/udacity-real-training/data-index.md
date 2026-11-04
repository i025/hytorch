# 原始数据索引

最终状态见[README](README.md)：47项正式任务完成，长期严格数值验收FAIL，未证明泛化改善。服务器项目入口`/home/xymo/projects/hytorch`，规范路径`/srv/lab-disks/disk2/users/xymo/projects/hytorch`。

## 原始目录与可移植归档

| 服务器路径（相对项目） | 内容 | 可移植记录 |
|---|---|---|
| `runs/udacity_real_prepared_01/` | 数据审计、8036行manifest、固定划分/顺序/预处理协议、1024张真实输入NPZ、逐图预处理时间 | [准备数据](raw-preparation-01/udacity_real_prepared_01) |
| `runs/udacity_real_cpu_correctness_01/` | 真实图像CPU三步比较、生产输入和参数/梯度NPZ | [CPU数值](raw-preparation-01/udacity_real_cpu_correctness_01) |
| `runs/udacity_real_window_01/` | GPU1首次窗口、816组GPU数值比较、被监控终止的烟测、完整资源/进程状态 | [GPU首次窗口](raw-preparation-01/udacity_real_window_01) |
| `runs/udacity_real_window_02/` | 用户批准同GPU1完整补测：47任务、全部真实训练/验证/性能、原始监控和2份Nsight | [完整补测](raw-final-01/udacity_real_window_02) |
| `runs/udacity_real_analysis_01/` | GPU释放后的SQLite、关联Kernel/API/NVTX原始CSV和命令/日志 | [Nsight离线导出](raw-final-01/udacity_real_analysis_01) |

[file-index.csv](raw-preparation-01/file-index.csv)列出37份原始文件的绝对路径、大小和可移植记录。其中16份文本/CSV/JSON/日志复制到Git；NPZ仅留服务器，单文件大于1MB的文本无损gzip，索引给出实际Git文件名。不进行哈希验证，不上传原始图像、模型权重或下载zip。

补测[final file-index.csv](raw-final-01/file-index.csv)另列477份服务器原始文件、368份可移植记录。两份索引合计514份原始文件、384份完整可移植记录；不是仅摘要。构建动态库复用4G原始路径，不复制二进制。所有初始/最终权重和逐值对照NPZ留服务器，不进入Git。

## 关键结果

- 数据审计：[dataset.json](raw-preparation-01/udacity_real_prepared_01/dataset.json)。
- 固定划分/训练顺序/预处理：[protocol.json](raw-preparation-01/udacity_real_prepared_01/protocol.json)。
- 标签/图像行号与划分：[manifest.csv](raw-preparation-01/udacity_real_prepared_01/manifest.csv)。
- 逐图预处理：[preprocess-times.csv](raw-preparation-01/udacity_real_prepared_01/preprocess-times.csv)。
- CPU回归：[82项XML](results/host-tests-02.xml)、[完整日志](logs/host-tests-02.log)；初版80项亦保留。
- CPU真实数值：[408组误差](raw-preparation-01/udacity_real_cpu_correctness_01/errors.csv)。
- GPU真实数值：[816组误差](raw-preparation-01/udacity_real_window_01/correctness-real/errors.csv)、[通过摘要](raw-preparation-01/udacity_real_window_01/correctness-real/summary.json)。
- GPU启动状态：[before.json](raw-preparation-01/udacity_real_window_01/before.json)。
- 逐次GPU/进程/pmon记录：[resources.jsonl](raw-preparation-01/udacity_real_window_01/resources.jsonl)。
- 实际命令和返回码：[commands.json](raw-preparation-01/udacity_real_window_01/commands.json)。
- 退出后状态：[after.json](raw-preparation-01/udacity_real_window_01/after.json)、[窗口结束](raw-preparation-01/udacity_real_window_01/window.json)。
- 中断解释：[window-01-interruption.json](results/window-01-interruption.json)。
- 47任务完整计划：[gpu-plan-01.json](results/gpu-plan-01.json)。
- 准备阶段记录：[preparation.json](results/preparation.json)。

完整stdout/stderr与命令元数据在[logs](logs)；`*.command.json`保存argv、环境、起止时间、返回码。CPU环境见[environment](environment)。执行命令见[commands.md](commands.md)。

## 准备图与正式数据

[标签分布](figures/preparation-01/dataset-steering.png)、[CPU/GPU数值误差分布](figures/preparation-01/real-image-correctness.png)，绘图源码[scripts/plot_preparation.py](scripts/plot_preparation.py)。这些图均来自实测CSV，不是截图。

正式第二窗口已完成上述任务；完整原始数据与负结果如下。

- [47条实际命令/返回码/PID](raw-final-01/udacity_real_window_02/commands.json)、[完整外层终端输出](logs/gpu-window-02.log)。
- [第二窗口启动](raw-final-01/udacity_real_window_02/before.json)、[结束/释放](raw-final-01/udacity_real_window_02/window.json)、[释放后仅原任务](raw-final-01/udacity_real_window_02/after.json)、[原始逐次监控/pmon](raw-final-01/udacity_real_window_02/resources.jsonl)。
- [第二窗口GPU816组比较](raw-final-01/udacity_real_window_02/correctness-real/errors.csv)、[通过摘要](raw-final-01/udacity_real_window_02/correctness-real/summary.json)。
- 每个`compute/io/phases-{A,F,I,E}-r{1,2,3}/`、`train-{A,F,I,E}/`、`smoke-{A,F,I,E}/`都有config、device、summary、steps、losses、epochs和初始/最终权重。原始根为`runs/udacity_real_window_02/`；可移植日志全部在[同名归档](raw-final-01/udacity_real_window_02)，较大jsonl压缩文件名见索引。
- [逐进程原始吞吐/墙钟](results/acceptance-01/training-raw.csv)、[全部逐步计时](results/acceptance-01/steps.csv)、[全部Loss](results/acceptance-01/losses.csv)、[三重复统计](results/acceptance-01/timing-summary.csv)、[全部本窗口收益](results/acceptance-01/speedups.csv)。
- [逐epoch训练/验证/阶段/内存记录](results/acceptance-01/epochs.csv)、[全部验证逐图预测](results/acceptance-01/validation-predictions.csv)、[零预测/训练均值常数基线](results/acceptance-01/constant-baselines.csv)。
- [1530组数值结果（含119失败）](results/acceptance-01/numerical-errors.csv)、[首个Loss超差及图像行号](results/drift-01/loss-drift.csv)、[全部末态参数误差和最坏坐标](results/drift-01/weight-drift.csv)、[未确认根因的诊断摘要](results/drift-01/summary.json)。
- [共享资源CSV](results/acceptance-01/resources.csv)、[进程pmon字段](results/acceptance-01/process-activity.csv)、[逐任务/epoch采样覆盖](results/extra-01/per-run-resource-samples.csv)、[采样限制摘要](results/extra-01/summary.json)。原任务SM字段无可用值，不能用空值当0。
- Nsight原件：`/home/xymo/projects/hytorch/runs/udacity_real_window_02/trace-F.nsys-rep`和`trace-I.nsys-rep`；SQLite：`runs/udacity_real_analysis_01/trace-F.sqlite`和`trace-I.sqlite`。均留服务器。Git上传其[完整导出CSV/日志](raw-final-01/udacity_real_analysis_01)、[phase统计](results/acceptance-01/trace-phases.csv)、[每层每次dWeight组成](results/acceptance-01/dweight-components.csv)、[每step归并](results/extra-01/trace-totals.csv)。
- [最终机器汇总](results/acceptance-01/summary.json)的COMPLETED表示执行完成，`numerical_passed=false`明确表示长期验收未通过。

## 全部正式图像

| 图 | 链接 |
|---|---|
| 四版本真实吞吐 | [01](figures/final-01/01-throughput.png) |
| Train/Validation Loss | [02](figures/final-01/02-train-validation-loss.png) |
| Steering MAE/RMSE | [03](figures/final-01/03-steering-errors.png) |
| Forward/Backward/Update与数据处理 | [04](figures/final-01/04-phases.png) |
| 相对无Native累计收益 | [05](figures/final-01/05-cumulative-gains.png) |
| 与PyTorch差距 | [06](figures/final-01/06-pytorch-gap.png) |
| GPU1共享资源 | [07](figures/final-01/07-shared-resources.png) |
| Nsight本轮进程CPU/GPU时间线 | [08](figures/final-01/08-nsight-timeline.png) |
| 全部标签分布 | [09](figures/final-01/09-steering-distribution.png) |
| 实际各epoch耗时 | [10](figures/final-01/10-epoch-times.png) |
| 长轨迹严格数值失败 | [11](results/extra-01/11-long-run-drift.png) |
| 第三轮验证与初始/零预测对照 | [12](results/extra-01/12-validation-baselines.png) |

所有图来自真实CSV/CUPTI，不是截图。脚本[plot.py](scripts/plot.py)、[extra_analysis.py](scripts/extra_analysis.py)；没有追加GPU实验、优化Kernel或掩盖负结果。
