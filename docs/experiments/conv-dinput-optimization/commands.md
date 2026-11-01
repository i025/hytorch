# 可复现命令与执行记录

工作目录 `/home/xymo/projects/hytorch`，真实路径 `/srv/lab-disks/disk2/users/xymo/projects/hytorch`。所有新输出使用唯一目录，失败、外部任务中断、成功三类记录都保留。

```bash
source docs/experiments/a100-stage2a-gemm/environment/activate.sh
git switch -c experiment/conv-dinput-kernel-optimization e2d60c26cdb28facd4a41dd3b443c3148e397b66
python scripts/build_cuda.py --arch 80 --resource-usage --output runs/conv_dinput_opt_build_01
cuobjdump --dump-resource-usage runs/conv_dinput_opt_build_01/libhytorch_cuda.so
python docs/experiments/conv-dinput-optimization/scripts/cases.py docs/experiments/conv-dinput-optimization/results/predeclared-shapes.json
python docs/experiments/conv-dinput-optimization/scripts/check_host_equations.py --output runs/conv_dinput_opt_host_equations_01
python -m pytest tests/test_conv_dinput_opt_host.py tests/test_conv_shape_dispatch.py \
 tests/test_conv_backward_fusion_host.py tests/test_conv_dinput_pruning.py tests/test_cuda_host.py \
 tests/test_conv2d_gemm_cpu.py tests/test_gemm_blocked_packed.py -q -ra \
 --junitxml=docs/experiments/conv-dinput-optimization/results/host-tests-01.xml
```

CPU任务保持激活脚本的 `CUDA_VISIBLE_DEVICES=''`；显式SM80编译/符号加载不初始化GPU。完整环境/包列表在 environment/system-info.log，原始argv、环境和时间在同名 `.command.json`。

## GPU选择与回归

开始时八卡均有其他用户进程，先完成CPU工作。用户通知空闲后首先使用物理GPU2，UUID `GPU-158426a4-6684-7863-8cbe-ed6595f97683`。第二次自动等待器短暂选到GPU0，外部任务重新进入后guard停止本次进程，保留中断。随后固定GPU2，逻辑cuda:0，其他卡不可见。

复用未修改的4E `scripts/run_gpu.py`：开始连续两次无计算客户端、0利用率、显存≤4 MiB才启动；持续监测外部客户端，出现即停止本次进程。无sudo、reset、GPU权限或系统配置修改。

完整最终回归命令保存在 `logs/gpu-correctness-03.log.command.json` 和 `runs/conv_dinput_opt_correctness_03/guard/launch.json`。设置 `HYTORCH_REQUIRE_CUDA=1 HYTORCH_PRUNING_DEVICE=cuda`、新动态库及每类原始数据输出目录，执行：

```bash
python -m pytest tests/test_conv_dinput_opt_gpu.py tests/test_conv_shape_dispatch_gpu.py \
 tests/test_conv_backward_fusion_gpu.py tests/test_conv_dinput_pruning.py \
 tests/test_fused_sgd_gpu.py tests/test_cuda_device.py -q -ra \
 --junitxml=runs/conv_dinput_opt_correctness_03/results/gpu-tests.xml
```

首次失败诊断：

```bash
python docs/experiments/conv-dinput-optimization/scripts/diagnose_forward_reference.py \
 --inputs runs/conv_dinput_opt_correctness_wait_01/attempt-01/results/shape-correctness \
 --output runs/conv_dinput_opt_forward_diagnosis_01/attempt-01/results
```

这些为实际目录，重跑必须换名；程序故意不覆盖。

## 匹配性能与Nsight

所有下述GPU命令通过单卡guard执行，例：

```bash
python docs/experiments/conv-shape-dispatch/scripts/run_gpu.py \
 --gpu GPU-158426a4-6684-7863-8cbe-ed6595f97683 --idle-memory-mib 4 \
 --record runs/NEW_UNIQUE_WINDOW/guard -- env \
 HYTORCH_CUDA_LIBRARY=/home/xymo/projects/hytorch/runs/conv_dinput_opt_build_01/libhytorch_cuda.so \
 python docs/experiments/conv-dinput-optimization/scripts/run_micro.py \
 --correctness docs/experiments/conv-dinput-optimization/results/gpu-correctness-03.json \
 --output runs/NEW_UNIQUE_WINDOW/results
```

实际微基准输出 `runs/conv_dinput_opt_micro_01/results`：history100+fresh24，original/fused/fused_opt/torch四方法完整操作；旧/新Gather另测预分配Kernel范围。5预热、30测量、3独立进程、轮换方法顺序，共36进程。

训练启动脚本（同样包在guard中）：

```bash
python docs/experiments/conv-dinput-optimization/scripts/run_training.py \
 --output runs/conv_dinput_opt_training_01/results \
 --gpu-uuid GPU-158426a4-6684-7863-8cbe-ed6595f97683 \
 --library /home/xymo/projects/hytorch/runs/conv_dinput_opt_build_01/libhytorch_cuda.so \
 --correctness docs/experiments/conv-dinput-optimization/results/gpu-correctness-03.json
```

A/D/F/H/E、DonkeyNet/SpatialNet、throughput/phases各三重复，共60进程。固定batch4、FP32、seed42、SGD0.001、10step预热后恢复初始权重、256step、相同输入/顺序、Dropout0、TF32关闭，全部 `--no-fingerprints --save-comparison-inputs --no-monitor`。

A关闭全部Native并指定不存在的动态库；D为 `--matmul shared --conv im2col --pool cupy --tile 16 --fused-sgd --prune-input-grad`；F=D+`--conv-backward fused`；H=D+`--conv-backward fused_opt`；E为匹配PyTorch CUDA。完整子进程argv与时间保存在各根目录commands.json。

另开唯一目录执行上述 `run_training.py --traces` 和 `run_micro.py --traces`，均在普通计时结束后。训练只采A/F/H两个模型各16step，共6Trace；微基准完整操作每形状3记录、预分配Kernel每形状30记录×3独立进程，共20Trace。使用 `nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none`，再 `nsys export --type sqlite`；完整argv也保留，不计算内容哈希。

## 数据分析

```bash
python docs/experiments/conv-dinput-optimization/scripts/analyze_micro.py \
 --raw runs/conv_dinput_opt_micro_01/results --output docs/experiments/conv-dinput-optimization/results/micro-01
python docs/experiments/conv-dinput-optimization/scripts/analyze_training.py \
 --raw runs/conv_dinput_opt_training_01/results --output docs/experiments/conv-dinput-optimization/results/training-01
```

Nsight关联解析、绘图及归档的最终实际命令见logs对应 `.command.json`，大型原始张量/.nsys-rep/SQLite留在服务器，文本CSV/JSON/XML/日志和PNG上传。归档复用4B archive_results.py，不改写历史原始记录。

## 第二完整匹配窗口与部分Nsight

首个训练窗口各组波动较大，先保存 `results/repeat-training-protocol.json`，再完全重复相同A/D/F/H/E、两个模型、throughput/phases的60个进程，不挑选单组重测：

```bash
python docs/experiments/conv-dinput-optimization/scripts/run_training.py \
 --output runs/conv_dinput_opt_training_02/results \
 --gpu-uuid GPU-158426a4-6684-7863-8cbe-ed6595f97683 \
 --library /home/xymo/projects/hytorch/runs/conv_dinput_opt_build_01/libhytorch_cuda.so \
 --correctness docs/experiments/conv-dinput-optimization/results/gpu-correctness-03.json
python docs/experiments/conv-dinput-optimization/scripts/analyze_training.py \
 --raw runs/conv_dinput_opt_training_02/results --output docs/experiments/conv-dinput-optimization/results/training-02
python docs/experiments/conv-dinput-optimization/scripts/analyze_profiles.py \
 --micro runs/conv_dinput_opt_traces_01/micro --training runs/conv_dinput_opt_traces_01/training \
 --partial --kernel-repeats 2 --analysis runs/conv_dinput_opt_trace_analysis_partial_01 \
 --output docs/experiments/conv-dinput-optimization/results/profiles-partial-01
python docs/experiments/conv-dinput-optimization/scripts/summarize_results.py \
 --profiles docs/experiments/conv-dinput-optimization/results/profiles-partial-01 \
 --output docs/experiments/conv-dinput-optimization/results/report-partial-01
python docs/experiments/conv-dinput-optimization/scripts/plot_results.py \
 --profiles docs/experiments/conv-dinput-optimization/results/profiles-partial-01 \
 --output docs/experiments/conv-dinput-optimization/figures/partial-01
```

GPU启动仍包在guard中，完整嵌套命令以 `logs/training-02.log.command.json` 和原始 `guard/launch.json` 为准。解析部分数据时明确 `--partial --kernel-repeats 2`，不让两重复冒充三重复。

Nsight第一次窗口在后期中断，保留22份完整配对导出和未纳入的第三重复。后续窗口02被真实外部进程进入中断；wait03在GPU6中断，wait04在GPU0发生Nsight客户端归属判断中断。4F独立guard在原guard基础上补充父进程链、消失/退出客户端判断，不改历史脚本；05随后确有外部PID3890942进入GPU0并持续占用。部分早期归属中断的具体生命周期原因仍属推断，日志不改写为确定外部负载。

最后等待最多1200秒、每20秒检查全部A100，连续两次完全空闲才进入一次新采集尝试，不要求固定GPU6：

```bash
python docs/experiments/conv-dinput-optimization/scripts/wait_gpu.py \
 --output runs/conv_dinput_opt_traces_wait_06 --seconds 1200 --idle-memory-mib 4 --idle-samples 2 -- env \
 HYTORCH_CUDA_LIBRARY=/home/xymo/projects/hytorch/runs/conv_dinput_opt_build_01/libhytorch_cuda.so bash -ec \
 'python docs/experiments/conv-dinput-optimization/scripts/run_training.py --traces --output {output}/training --gpu-uuid {gpu} --library /home/xymo/projects/hytorch/runs/conv_dinput_opt_build_01/libhytorch_cuda.so --correctness docs/experiments/conv-dinput-optimization/results/gpu-correctness-03.json
 python docs/experiments/conv-dinput-optimization/scripts/run_micro.py --traces --output {output}/micro --correctness docs/experiments/conv-dinput-optimization/results/gpu-correctness-03.json'
```

`wait_gpu.py`是4F的独立副本，使用本轮guard；首次CPU/GPU测试使用原4E guard的历史事实不变。每个新目录拒绝覆盖旧输出。最后等待/采集状态以独立completion.json为准。

首批便携归档的完整15个源根目录见 `logs/archive-01.log.command.json`；对二进制只记录服务器路径和字节数。Token交互脚本和凭据不参与归档，也不提交。

## 最终成功的完整Nsight窗口

wait06在物理GPU1 UUID `GPU-c6683db1-d2aa-97b9-c1b4-e497bd7ff0c5` 上完成，UTC17:46:44–17:51:48；全程无外部负载标记。正式训练仍采用GPU2的未插桩窗口，不把两个设备窗口混算比值。

```bash
python docs/experiments/conv-dinput-optimization/scripts/analyze_profiles.py \
 --micro runs/conv_dinput_opt_traces_wait_06/attempt-01/results/micro \
 --training runs/conv_dinput_opt_traces_wait_06/attempt-01/results/training \
 --analysis runs/conv_dinput_opt_trace_analysis_01 \
 --output docs/experiments/conv-dinput-optimization/results/profiles-01
python docs/experiments/conv-dinput-optimization/scripts/summarize_results.py \
 --profiles docs/experiments/conv-dinput-optimization/results/profiles-01 \
 --output docs/experiments/conv-dinput-optimization/results/report-01
python docs/experiments/conv-dinput-optimization/scripts/plot_results.py \
 --profiles docs/experiments/conv-dinput-optimization/results/profiles-01 \
 --output docs/experiments/conv-dinput-optimization/figures/final-01
python docs/experiments/conv-dinput-optimization/scripts/summarize_hardware.py \
 --runs runs --output docs/experiments/conv-dinput-optimization/results/hardware-02
```

上述解析不带partial开关，要求全部26导出和3×30纯Kernel记录满足断言。先前22份/two-repeat结果保持原目录，只作为GPU2的辅助重复，不替换成功窗口或与GPU1配对。

归档02：`archive_results.py --destination docs/experiments/conv-dinput-optimization/raw-records/successful-traces-01 runs/conv_dinput_opt_traces_wait_06`。归档03为解析完成后的 `conv_dinput_opt_trace_analysis_01`；实际完整命令见各同名log.command.json。完整硬件采样按各子进程起止窗口连接，空采样不插值，整卡峰值不冒充进程峰值。

wait06最外层执行会话返回143，未产生run_logged的command.json，原因未确定，见 `logs/wrapper-exit-06.json`。不补造成功的外层元数据。其内部waiter/guard均回写成功，training12条及micro40条子命令各自完整记录argv、时间和returncode，26对nsys-rep/SQLite均完成；完整子命令stdout/stderr和终端日志在成功采集归档中。采集成功判据是这些内部记录及完整导出的解析断言，不能把外层退出码报告成0。
