# 复现实验

固定环境记录在ENVIRONMENT.json、baseline/ENVIRONMENT.json与raw/resource_*.json。使用现有Python3.10环境：`/home/xieminhui_pubuse/hytorch-env/bin/python`。不要在原交付目录覆写数据；run_tasks遇到已有日志会拒绝运行。

## 创建隔离副本（无Git操作）

```bash
# 在原仓库根执行；TASK_REPRO必须是尚不存在的新目录。
TASK_SRC=/home/xieminhui_pubuse/hytorch/runs/a100_20261107/conv_pool_framework_optimization
TASK_REPRO=/home/xieminhui_pubuse/hytorch_conv_pool_reproduction
mkdir "$TASK_REPRO"
cp -a "$TASK_SRC/source/." "$TASK_REPRO/"
mkdir -p "$TASK_REPRO/runs/a100_20261107/conv_pool_framework_optimization"
cp -a "$TASK_SRC/diagnostics" "$TASK_REPRO/runs/a100_20261107/conv_pool_framework_optimization/"
mkdir -p "$TASK_REPRO/runs/a100_20261107/conv_pool_framework_optimization/"{raw,stage_a,traces,figures}
cd "$TASK_REPRO"
```

source包含本轮所需hytorch、tests、benchmarks、donkeycar_hy及原CUDA动态库。再次构建不是本次Python-only变更所需。原始baseline冻结命令为diagnostics/save_baseline.py；它要求真实Git工作区且目录不存在，隔离副本直接使用交付的baseline快照，不在无Git副本重跑。

## GPU任务

runner固定物理GPU1并在每个任务前检查其是否已有计算进程；占用时退出，不终止别人任务。CUDA_PATH=/usr/local/cuda、CUDA_VISIBLE_DEVICES=1、NVIDIA_TF32_OVERRIDE=0、PYTHONHASHSEED=0、CPU线程4由runner设置；每项运行命令、UTC与exitcode在raw/commands.jsonl。先查询nvidia-smi并遵守服务器授权。

```bash
PY=/home/xieminhui_pubuse/hytorch-env/bin/python
D=runs/a100_20261107/conv_pool_framework_optimization/diagnostics
$PY $D/run_tasks.py baseline cpu special tests tests_on memcheck racecheck
$PY $D/run_tasks.py layers:--cpu layers:--repeats:18:--tag:balanced18
$PY $D/run_tasks.py layertrace:B0 layertrace:O1 layertrace:torch
$PY $D/run_tasks.py balanced cpu:O1
$PY $D/run_tasks.py trace:B0:b trace:O1:b trace:torch:b trace:B0:c trace:O1:c trace:torch:c
$PY $D/run_tasks.py supplement memory memory_torch workload tiny_workload
$PY $D/run_tasks.py layers:--repeats:18:--tag:actual18:--case-prefix:tiny_actual_pool
$PY $D/run_tasks.py layertrace:B0:actual layertrace:O1:actual layertrace:torch:actual
$PY $D/analyze_profiles.py
$PY $D/summarize.py
$PY $D/aggregate.py
```

主局部计时balanced18使用layers_main18.py，其输入生成顺序与原始主窗口一致。layers.py额外构建4通道真实pool；`layertrace`主shape应使用与主窗口一致的layers_main18.py，runner已为非actual追踪选择它。原15次探索数据保存在raw/LAYER_TIMINGS.csv；正式图表只用18次平衡样本。trace每shape15次，三种框架单独capture；所有trace仅作解释，不参与wall验收。

原生产路径在baseline/source/core.py；正式B0是同一最终源码关闭新开关，O1开启，二者其他配置相同。隔离副本测试通过后如需生成报告，把交付baseline/复制到对应输出目录；report.py中的断言要求仅core.py区别于基线。报告整理脚本不需要GPU：

```bash
$PY $D/report.py
$PY $D/figures.py
$PY $D/package.py
```

报告时间窗口和历史事实针对原始运行；重新运行后请更新人工说明，不能把它们当新实验元数据。代码/测试都不要求提交Git。
