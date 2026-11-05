# 命令与复现

以下在`/home/xymo/projects/hytorch`执行；没有哈希/Checksum/Fingerprint步骤。
实验Python命令前加：

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
 taskset -c 0-3 nice -n 10 .venv/bin/python ...
```

分支基线c00a8d3，创建`experiment/gpu0-autonomous-7h`。
完整实际CPU命令/环境/开始结束/返回码保存在`logs/*.command.json`，stdout/stderr在相应log。

```bash
.venv/bin/python scripts/build_cuda.py --arch 80 --resource-usage --output runs/gpu0_autonomous_7h_01/build-01
.venv/bin/python docs/experiments/gpu0-autonomous-7h/scripts/profile_cpu.py --output runs/gpu0_autonomous_7h_01/cpu-profile-01
.venv/bin/python docs/experiments/gpu0-autonomous-7h/scripts/bench_schedule.py --prepared runs/udacity_real_prepared_01 --output runs/gpu0_autonomous_7h_01/schedule-bench-02
.venv/bin/python -m pytest tests/test_backward_schedule.py -q -ra --junitxml=docs/experiments/gpu0-autonomous-7h/correctness/schedule-host-02.xml
.venv/bin/python -m pytest tests/test_reduction_gpu.py tests/test_relu_backward_gpu.py tests/test_backward_schedule_gpu.py --collect-only -q
.venv/bin/python docs/experiments/gpu0-autonomous-7h/scripts/check_sum_cpu_training.py --prepared runs/udacity_real_prepared_01 --output runs/gpu0_autonomous_7h_01/cpu-sum-training-01
.venv/bin/python docs/experiments/gpu0-autonomous-7h/scripts/cpu_sum_trajectory.py --prepared runs/udacity_real_prepared_01 --schedule runs/gpu0_autonomous_7h_01/schedule.json --output runs/gpu0_autonomous_7h_01/cpu-sum-trajectory-01
.venv/bin/python docs/experiments/gpu0-autonomous-7h/scripts/analyze_phases.py --input runs/udacity_real_analysis_01/trace-I.sqlite --output runs/gpu0_autonomous_7h_01/historical-I-phase-02
```

完整最终213项主机集合的实际argv见`logs/regression-final-01.log.command.json`，不是GPU结果；此前210项及定向测试日志保留。
初次schedule-host-01和schedule-bench-01失败记录保留。

## 持续监控与自动触发

tmux会话`hytorch-gpu0-7h`执行以下命令（需在允许持续进程的服务器终端中运行）：

```bash
.venv/bin/python docs/experiments/gpu0-autonomous-7h/scripts/watch_gpu0.py \
 --schedule runs/gpu0_autonomous_7h_01/schedule.json
```

监控将CSV逐行flush；GPU空闲且内部准备好的`gpu-plan.json`已存在时自动启动`run_window.py`。
不需要用户创建任何review-allow文件。`gpu-plan-staged.json`是未激活的内部准备记录，
CPU检查、源码提交就绪后由代理原子改名激活；不是用户审批流程。
独立控制器1秒检查截止、每5秒只读检查冲突；不持有GPU Context，不创建并行重型任务。
所有GPU子命令/开始结束/返回码/UUID在窗口`window.json`中，完整输出各自单独保存。

Sum测试原始失败不修改：普通输入/实现资格、强消去误差分别记录。
只有符合本轮数值资格的shape进入Sum微基准；ReLU独立验收；基线数值失败停止训练性能实验。
正常吞吐与Nsight命令不同，插桩数据不混算。GPU控制器固定NVIDIA_TF32_OVERRIDE=0。

## 公平协议

复用4H `runs/udacity_real_prepared_01/protocol.json`、`throughput-inputs.npz`、真实CSV和固定indices。
同步与2worker预取都使用同一个UdacityDataset/open_stream=True；E-prefetch获得同样的CPU预取。
10预热重置，256step×3独立重复，轮换方法；compute/io/phases分别运行。
每轮CPU/GPU原型都有独立目录，不覆盖历史。

## 离线解析

窗口结束后再使用`nsys export --type sqlite`及既有
`docs/experiments/conv-dweight-fusion/scripts/analyze_micro_trace.py`关联API/correlationId与同线程NVTX。
Sum标签每(case,repeat,implementation)30次，expected-cases=实际通过shape数×3×4；
ReLU trace每shape共30范围，expected-cases=12。
解析器旧dWeight分类列不用于Sum/ReLU，按实际Kernel名称重新统计。
事件span包含可能的提交空隙，不是纯Kernel时间；纯Kernel时间来自相关联CUPTI duration。

本轮 CPU 汇总复核命令（同样使用 CUDA_VISIBLE_DEVICES=''、BLAS 单线程及 CPU 0–3）：

```bash
.venv/bin/python docs/experiments/gpu0-autonomous-7h/scripts/summarize.py --schedule runs/gpu0_autonomous_7h_01/schedule.json --label cpu-review-01
.venv/bin/python docs/experiments/gpu0-autonomous-7h/scripts/summarize.py --schedule runs/gpu0_autonomous_7h_01/schedule.json --label monitor-review-01
.venv/bin/python docs/experiments/gpu0-autonomous-7h/scripts/analyze_cpu_calls.py --input runs/gpu0_autonomous_7h_01/cpu-profile-01/profile-functions.csv --output runs/gpu0_autonomous_7h_01/cpu-calls-01
.venv/bin/python docs/experiments/a100-stage1/run_logged.py --log docs/experiments/gpu0-autonomous-7h/logs/report-status-01.log -- .venv/bin/python -m pytest tests/test_gpu0_report_status.py -q -ra --junitxml=docs/experiments/gpu0-autonomous-7h/correctness/report-status-01.xml
```

旧阶段性文件是当时采样的快照；最终汇总使用新的独立 label，不覆盖上述数据。

收尾执行正常Git commit/push到本分支，不force、不合并、不启动下一轮。

## CPU 拓扑补测

继续使用 CUDA_VISIBLE_DEVICES=''、BLAS 单线程和 CPU 0–3 affinity；遇本轮 GPU 窗口自动暂停新的 CPU 计时组。

```bash
.venv/bin/python docs/experiments/a100-stage1/run_logged.py --log docs/experiments/gpu0-autonomous-7h/logs/graph-shapes-01.log -- .venv/bin/python docs/experiments/gpu0-autonomous-7h/scripts/bench_graph_shapes.py --output runs/gpu0_autonomous_7h_01/graph-shapes-01 --schedule runs/gpu0_autonomous_7h_01/schedule.json
.venv/bin/python docs/experiments/gpu0-autonomous-7h/scripts/plot_graph_shapes.py --input runs/gpu0_autonomous_7h_01/graph-shapes-01/summary.csv --output docs/experiments/gpu0-autonomous-7h/figures/cpu-graph-topologies-01
.venv/bin/python docs/experiments/a100-stage1/run_logged.py --log docs/experiments/gpu0-autonomous-7h/logs/sum-conditioning-01.log -- .venv/bin/python docs/experiments/gpu0-autonomous-7h/scripts/cpu_sum_conditioning.py --output runs/gpu0_autonomous_7h_01/sum-conditioning-01 --schedule runs/gpu0_autonomous_7h_01/schedule.json
.venv/bin/python docs/experiments/gpu0-autonomous-7h/scripts/plot_sum_conditioning.py --input runs/gpu0_autonomous_7h_01/sum-conditioning-01/samples.csv --output docs/experiments/gpu0-autonomous-7h/figures/cpu-conditioning-01
.venv/bin/python docs/experiments/a100-stage1/run_logged.py --log docs/experiments/gpu0-autonomous-7h/logs/schedule-memory-02.log -- .venv/bin/python docs/experiments/gpu0-autonomous-7h/scripts/measure_schedule_memory.py --output runs/gpu0_autonomous_7h_01/schedule-memory-02 --schedule runs/gpu0_autonomous_7h_01/schedule.json --warmup 2
```

无预热首轮 `schedule-memory-01` 的原版脚本和完整argv保留在服务器及日志，完整36次原始值未修改；
现版脚本也可用 `--warmup 0` 重现该测量口径。
