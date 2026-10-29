# 补测执行命令

工作目录 `/home/xymo/projects/hytorch`（真实路径 `/srv/lab-disks/disk2/users/xymo/projects/hytorch`），基线 `3e58e7093e152c7aae1c8a2334bb1f9f84f100ea`。所有目录均新建，原 CPU 实验和此前结果未覆盖。

## 监控和 GPU 执行

```bash
source docs/experiments/a100-stage2a-gemm/environment/activate.sh
python docs/experiments/a100-stage1/run_logged.py \
  --log docs/experiments/conv-backward-fusion/supplement-01/monitor-launch.log -- \
  python docs/experiments/conv-backward-fusion/supplement-01/scripts/watch_gpu.py \
  --output runs/conv_backward_fusion_wait_01 --seconds 7200 --interval 20 \
  --library runs/conv_backward_fusion_build_01/libhytorch_cuda.so
```

初始时间为 UTC 2026-10-29 15:52:47.008723，两小时截止 UTC17:52:47.008723（北京时间跨日到 10月30日01:52:47）。第一窗口物理 GPU0，通过测试/微基准，但被外部客户端打断。

第二次等待只使用原期限剩余时间，不重新加两小时：

```bash
remaining_seconds=$(python -c 'import json,time,datetime; d=json.load(open("runs/conv_backward_fusion_wait_01/config.json")); print(max(0,datetime.datetime.fromisoformat(d["started_at"]).timestamp()+7200-time.time()))')
python docs/experiments/a100-stage1/run_logged.py \
  --log docs/experiments/conv-backward-fusion/supplement-01/monitor-resume-02.log -- \
  python docs/experiments/conv-backward-fusion/supplement-01/scripts/watch_gpu.py \
  --output runs/conv_backward_fusion_wait_02 --seconds "$remaining_seconds" --interval 20 \
  --library runs/conv_backward_fusion_build_01/libhytorch_cuda.so
```

第二窗口选择物理 GPU6 / UUID `GPU-d40fdd6f-33be-8ccc-fc58-f4daf39a4b82`，完成全部测试及非插桩测量，Nsight F 启动时被外部客户端打断。新批次只调整测量顺序为三次 throughput 全部完成后再测 phases；仍按重复轮换 A/B/C/D/F/E，配置、步数、算法及数值容差未改。

后续在同一 GPU6 上仅补采 profiling：

```bash
remaining_seconds=$(python -c 'import json,time,datetime; d=json.load(open("runs/conv_backward_fusion_wait_01/config.json")); print(max(0,datetime.datetime.fromisoformat(d["started_at"]).timestamp()+7200-time.time()))')
python docs/experiments/a100-stage1/run_logged.py \
  --log docs/experiments/conv-backward-fusion/supplement-01/monitor-resume-03.log -- \
  python docs/experiments/conv-backward-fusion/supplement-01/scripts/watch_gpu.py \
  --output runs/conv_backward_fusion_wait_03 --seconds "$remaining_seconds" --interval 20 \
  --library runs/conv_backward_fusion_build_01/libhytorch_cuda.so \
  --gpu-uuid GPU-d40fdd6f-33be-8ccc-fc58-f4daf39a4b82 --trace-only
```

每个监控根目录保存 config.json、snapshots.jsonl、attempt-NN.json/log、guard-NN/、gpu-attempt-NN/ 和 completion.json。监控只读取 NVIDIA 状态；实际执行由既有 `run_gpu.py` 再检查两次空闲、用 UUID 设置 CUDA_VISIBLE_DEVICES，并持续检查外部客户端。逻辑设备始终为0，不改驱动、权限、调度器或其他用户进程。

用户随后明确允许其他空闲 A100。00:44 检查 watcher-03 没有子进程、attempt 或 guard launch，暂停确认后仅终止本次空等进程（PID1955690），原因保存为 `wait_03/selection-change.json`。随后取消 UUID 筛选，原截止时间不变：

```bash
remaining_seconds=$(python -c 'import json,time,datetime; d=json.load(open("runs/conv_backward_fusion_wait_01/config.json")); print(max(0,datetime.datetime.fromisoformat(d["started_at"]).timestamp()+7200-time.time()))')
python docs/experiments/a100-stage1/run_logged.py \
  --log docs/experiments/conv-backward-fusion/supplement-01/monitor-resume-04.log -- \
  python docs/experiments/conv-backward-fusion/supplement-01/scripts/watch_gpu.py \
  --output runs/conv_backward_fusion_wait_04 --seconds "$remaining_seconds" --interval 20 \
  --library runs/conv_backward_fusion_build_01/libhytorch_cuda.so --trace-only
```

`wait_04/config.json` 记录 `gpu_uuid=null`，表示任意空闲 A100；实际开始后仍由 guard 固定所选单卡。不同窗口、不同设备的正式性能数据和插桩数据分别保存。

为在较短空闲窗口优先取得核心证据，后续 Trace 顺序改为 D、F、A、B、C、E，再采三种微基准；只改采集顺序，不改训练配置或 Kernel。18 条命令的更新计划保存于 `runs/conv_backward_fusion_trace_plan_03/`，见 `trace-priority-plan.log`。计划仍标记 NOT_RUN，不代替实际执行状态。

各窗口 **真正执行的完整 argv 和起止时间** 在 `gpu-attempt-01/commands.json`，未完成命令不会伪填 finished_at。对应日志与每次独立训练输出均在同一目录。

GPU 测试由原脚本展开为：

```bash
python -m pytest tests/test_conv_backward_fusion_gpu.py tests/test_conv_dinput_pruning.py \
  tests/test_fused_sgd_gpu.py tests/test_cuda_device.py -q -ra \
  --junitxml=NEW_GPU_DIRECTORY/gpu-tests.xml
```

环境由 runner 设置 HYTORCH_REQUIRE_CUDA=1、HYTORCH_PRUNING_DEVICE=cuda、三个独立数值数组输出路径和 HYTORCH_CUDA_LIBRARY；明确请求的 GPU 运行失败时不能 skip。全部训练有 `--no-fingerprints`。没有执行任何内容哈希验证。

正式非插桩测量保留原配置：独立微基准 original/fused/torch 各3次，Conv2–5、dinput/full、5 warmup +30测量；训练六版本 × throughput/phases ×3次，每次10 warmup、256Step、Batch4、FP32、SGD.001、seed42、Dropout0、TF32关闭。每个进程独立初始化，预热后恢复权重，相同数据顺序。

profile-only 执行计划已实际生成到 `runs/conv_backward_fusion_trace_plan_02`，记录在 trace-plan.log。计划不是成功执行记录；真实执行以各窗口 commands.json 为准。

## 完成的数据分析

```bash
python docs/experiments/a100-stage1/run_logged.py \
  --log docs/experiments/conv-backward-fusion/supplement-01/measurements-analysis-01.log -- \
  python docs/experiments/conv-backward-fusion/supplement-01/scripts/analyze_completed.py \
  --stage measurements --raw runs/conv_backward_fusion_wait_02/gpu-attempt-01 \
  --output docs/experiments/conv-backward-fusion/supplement-01/results/acceptance-01
```

复用原 training()/micro() 实现，逐数组核对初始化/输入、714 条训练权重/Loss 对照，以及全部微基准输出。新增完整 epoch 墙钟/Step、主机提交/等待、长期表和误差汇总字段；没有改容差或计算公式。

第一窗口完整的9次微基准另做离线分析：用 importlib 加载同一 analyze_results.py，调用 `micro(Path('runs/conv_backward_fusion_wait_01/gpu-attempt-01'), Path('docs/experiments/conv-backward-fusion/supplement-01/results/partial-01'))`。记录在 partial-micro-analysis.log。该目录标明整个训练批次不完整，不参与正式训练收益。

## Hardware 后处理与保留的失败

直接将被 guard 中断的原 commands.json 传给旧硬件汇总工具时，因最后一条无 finished_at 出现 KeyError，见 hardware-summary-02.log。原始日志保留。

从原 manifest 中另存45条 `name.startswith(('training-','micro-')) and returncode==0` 的记录到 `results/measurement-commands-02.json`，不修改原文件，然后执行：

```bash
python docs/experiments/a100-stage1/run_logged.py \
  --log docs/experiments/conv-backward-fusion/supplement-01/hardware-summary-03.log -- \
  python docs/experiments/fused-sgd-cuda/scripts/summarize_hardware.py \
  --commands docs/experiments/conv-backward-fusion/supplement-01/results/measurement-commands-02.json \
  --monitor runs/conv_backward_fusion_wait_02/guard-attempt-01/gpu-load.jsonl \
  --output docs/experiments/conv-backward-fusion/supplement-01/results/hardware-process-windows-02.csv
```

90条整进程窗口统计来自175个约秒级样本，45个已完成测量的 external_client_samples 全为0。不是逐阶段利用率，也不是瞬时分配器峰值。

## 已完成的部分 Trace 解析

对第二窗口中已完整导出的 A/B/C/D，逐个执行同样模板（真实 argv 在 interrupted-trace-X-analysis.log.command.json）：

```bash
python docs/experiments/donkeynet-backward-profiling/scripts/analyze_trace.py \
  --input runs/conv_backward_fusion_wait_02/gpu-attempt-01/trace-training-X.sqlite \
  --output runs/conv_backward_fusion_interrupted_trace_analysis_01/trace-training-X \
  --label stage4d-window02-X-completed-before-interruption
```

此处 X=A/B/C/D。A 全 Trace 3952个Kernel中，本项目 Native symbol 匹配为0；正式phase内2432个。A 的独立检查结果在 native-baseline-check.json。初次临时检查误转义单词边界，原检查保留为 invalid-pattern；更正后用 gemm_shared/conv_dinput_gather 正控确认模式有效，再重新计算，仍为0。主分析脚本的模式没有此问题。部分 Trace 不与后续 Trace 拼接成 D/F 性能比。

已完成的部分 Trace 另存逐组指标，不补填 F/E：

```bash
python docs/experiments/a100-stage1/run_logged.py \
  --log docs/experiments/conv-backward-fusion/supplement-01/partial-trace-summary.log -- \
  python docs/experiments/conv-backward-fusion/supplement-01/scripts/collect_partial_traces.py \
  --analysis runs/conv_backward_fusion_interrupted_trace_analysis_01 \
  --output docs/experiments/conv-backward-fusion/supplement-01/results/partial-traces-02.csv
```

## 图表

```bash
python docs/experiments/a100-stage1/run_logged.py \
  --log docs/experiments/conv-backward-fusion/supplement-01/measurement-figures-01.log -- \
  python docs/experiments/conv-backward-fusion/scripts/plot_results.py \
  --results docs/experiments/conv-backward-fusion/supplement-01/results/acceptance-01 \
  --analysis runs/conv_backward_fusion_trace_analysis_01 \
  --figures docs/experiments/conv-backward-fusion/supplement-01/figures-measurements-01 \
  --measurements-only
python docs/experiments/a100-stage1/run_logged.py \
  --log docs/experiments/conv-backward-fusion/supplement-01/validation-figures.log -- \
  python docs/experiments/conv-backward-fusion/supplement-01/scripts/plot_validation.py \
  --results docs/experiments/conv-backward-fusion/supplement-01/results/acceptance-01 \
  --output docs/experiments/conv-backward-fusion/supplement-01/figures-validation-01
```

这一等待期版本仅绘制真实已完成的普通测量。最终 Kernel 计数图/时间线使用下文完整匹配 Trace，没有用预期计数代替。

## 最终完成的补采、分析与归档

第四窗口 guard 误判自身 Nsight F 目标 PID1995660 后，第五窗口继续在原截止时间内监控任意空闲卡：

```bash
remaining_seconds=$(python -c 'import json,time,datetime; d=json.load(open("runs/conv_backward_fusion_wait_01/config.json")); print(max(0,datetime.datetime.fromisoformat(d["started_at"]).timestamp()+7200-time.time()))')
python docs/experiments/a100-stage1/run_logged.py \
  --log docs/experiments/conv-backward-fusion/supplement-01/monitor-resume-05.log -- \
  python docs/experiments/conv-backward-fusion/supplement-01/scripts/watch_gpu.py \
  --output runs/conv_backward_fusion_wait_05 --seconds "$remaining_seconds" --interval 20 \
  --library runs/conv_backward_fusion_build_01/libhytorch_cuda.so --trace-only
```

实际选中 GPU0 / UUID `GPU-f099fdbd-adf2-6157-3462-4145bc2a8359`，18 条命令全部0返回，9份 Trace/9次导出完成，guard84个样本无外部客户端。01:40:19 完成后不再运行 GPU 任务。

第四窗口的 F 已输出16步 summary 和完整 nsys-rep，另行做 CPU 恢复，仅作审计，不混入最终组：

```bash
nsys export --type sqlite \
  --output runs/conv_backward_fusion_wait_04/gpu-attempt-01/trace-training-F.sqlite \
  runs/conv_backward_fusion_wait_04/gpu-attempt-01/trace-training-F.nsys-rep
python docs/experiments/donkeynet-backward-profiling/scripts/analyze_trace.py \
  --input runs/conv_backward_fusion_wait_04/gpu-attempt-01/trace-training-F.sqlite \
  --output runs/conv_backward_fusion_recovered_trace_analysis_04/trace-training-F \
  --label stage4d-window04-F-recovered-after-wrapper-interruption
```

对应完整终端与 argv 在 recover-F-trace-04.log、recovered-F-analysis-04.log 及 command.json。PROCESSES 表与 guard PID 的比较见 guard-review-04.json；原始 guard 未改动，具体误判机制未证实。

```bash
python docs/experiments/conv-backward-fusion/supplement-01/scripts/analyze_completed.py \
  --stage profiling --raw runs/conv_backward_fusion_wait_02/gpu-attempt-01 \
  --trace-raw runs/conv_backward_fusion_wait_05/gpu-attempt-01 \
  --analysis runs/conv_backward_fusion_trace_analysis_01 \
  --output docs/experiments/conv-backward-fusion/supplement-01/results/acceptance-01
python docs/experiments/conv-backward-fusion/supplement-01/scripts/summarize_trace.py \
  --raw runs/conv_backward_fusion_wait_05/gpu-attempt-01 \
  --analysis runs/conv_backward_fusion_trace_analysis_01 \
  --results docs/experiments/conv-backward-fusion/supplement-01/results/acceptance-01 \
  --output docs/experiments/conv-backward-fusion/supplement-01/results/trace-summary-02
python docs/experiments/conv-backward-fusion/scripts/plot_results.py \
  --results docs/experiments/conv-backward-fusion/supplement-01/results/acceptance-01 \
  --analysis runs/conv_backward_fusion_trace_analysis_01 \
  --figures docs/experiments/conv-backward-fusion/supplement-01/figures-final-02
```

以上实际执行均经 run_logged，完整命令/输出在 profiling-analysis-01.log、trace-summary-02.log、final-figures-02.log 及各自 command.json。首版汇总误将 E 的同线程窗口当作完整工作线程包络，原表保留并标注 superseded；新表相关字段明确为不适用，phase核时和全部D/F值不变。首版图也保留，新图共享时间线横轴并补充纯设备核时。

归档实际执行（外层 run_logged 的原文另见三个 archive 日志/command.json）：

```bash
python docs/experiments/fused-sgd-cuda/scripts/archive_results.py \
  --destination docs/experiments/conv-backward-fusion/supplement-01/results/raw-measurements-02 \
  runs/conv_backward_fusion_wait_01 runs/conv_backward_fusion_wait_02 \
  runs/conv_backward_fusion_wait_03 runs/conv_backward_fusion_interrupted_trace_analysis_01 \
  runs/conv_backward_fusion_trace_plan_02
python docs/experiments/fused-sgd-cuda/scripts/archive_results.py \
  --destination docs/experiments/conv-backward-fusion/supplement-01/results/raw-traces-01 \
  runs/conv_backward_fusion_wait_04 runs/conv_backward_fusion_wait_05 \
  runs/conv_backward_fusion_recovered_trace_analysis_04 runs/conv_backward_fusion_trace_plan_03
python docs/experiments/fused-sgd-cuda/scripts/archive_results.py \
  --destination docs/experiments/conv-backward-fusion/supplement-01/results/raw-analysis-01 \
  runs/conv_backward_fusion_trace_analysis_01
```

共索引3371个服务器原始文件、复制743个可管理文本记录；大文件保留服务器，较大文本只 gzip，不删减数值。首次绝对别名路径归档失败保留在 archive-measurements-01.log 和未完成的新目录，成功版本使用相对路径。没有覆盖旧实验或进行内容哈希。

最终 status.json 从已有数值 CSV/JSON 和第五窗口 completion/guard 生成；monitor-summary.json 汇集五个新监控目录的 config、快照数量与完成状态。首版监控 config 没有 gpu_uuid 字段，汇总按该版未启用 UUID 过滤的实际行为记录 null。原 config 未改动。
