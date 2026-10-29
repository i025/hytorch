# 阶段 4C 执行命令与复现

工作目录 `/home/xymo/projects/hytorch`，真实路径 `/srv/lab-disks/disk2/users/xymo/projects/hytorch`。以下记录已执行的实验；复现必须选择当时空闲且获准的设备，并用新的输出目录。所有原始命令、返回码和输出均保留，不执行哈希/fingerprint 校验。

## 分支与 CPU

从干净的 `experiment/fused-sgd-cuda` / `5476cb65f3b0493e8edc7a495551c9cae25587fb` 创建：

```bash
git switch -c experiment/conv-backward-dinput-pruning
source docs/experiments/a100-stage2a-gemm/environment/activate.sh
```

激活脚本使用独立 `.venv`、Toolkit 12.2，初始 CUDA_VISIBLE_DEVICES 为空；OMP/OPENBLAS 线程数均为 4，CUPY_TF32=0，NVIDIA_TF32_OVERRIDE=0。没有安装包或修改系统。

在修改原高阶路径前执行并保留失败：

```bash
python docs/experiments/a100-stage1/run_logged.py \
  --log docs/experiments/conv-dinput-pruning/tests/higher-order-before.log -- \
  python docs/experiments/conv-dinput-pruning/scripts/reproduce_higher_order.py
```

局部修正后同命令输出到 `higher-order-after.log`。完整 argv、退出码见各 `.log.command.json`。然后运行 CPU 回归：

```bash
export HYTORCH_PRUNING_TEST_OUTPUT=/home/xymo/projects/hytorch/runs/conv_dinput_cpu_tests_01
python docs/experiments/a100-stage1/run_logged.py \
  --log docs/experiments/conv-dinput-pruning/tests/cpu-01.log -- \
  python -m pytest tests/test_conv_dinput_pruning.py tests/test_fused_sgd_host.py \
  tests/test_cuda_host.py tests/test_conv2d_gemm_cpu.py tests/test_gemm_blocked_packed.py \
  -q -ra --junitxml=docs/experiments/conv-dinput-pruning/tests/cpu-01.xml
```

完整环境采集的 argv、输出分别在 [system.txt.command.json](environment/system.txt.command.json)、[system.txt](environment/system.txt)，包含日期、hostname/id、OS、CPU、全部 GPU、pip list、GCC、nvcc、Nsight 版本。

## 单卡窗口

首次 guard 请求 UUID `GPU-f099fdbd-adf2-6157-3462-4145bc2a8359`（物理 0）、记录目录 `runs/conv_dinput_monitor_01`；因两次检查都有外部计算进程，返回 BLOCKED，没有启动 GPU 子命令，检查原文保留。

随后在物理 2 的空闲窗口运行：

```bash
source docs/experiments/a100-stage2a-gemm/environment/activate.sh
python docs/experiments/donkeynet-backward-profiling/scripts/run_gpu.py \
  --gpu GPU-158426a4-6684-7863-8cbe-ed6595f97683 \
  --record runs/conv_dinput_monitor_02 -- \
  python docs/experiments/conv-dinput-pruning/scripts/run_experiments.py \
  --output runs/conv_dinput_gpu_01 \
  --library runs/fused_sgd_cuda_build_01/libhytorch_cuda.so \
  --gpu-uuid GPU-158426a4-6684-7863-8cbe-ed6595f97683
```

guard 将此 UUID 映射为逻辑 GPU 0，持续检查外部占用。本轮全部 41 条 GPU 子命令成功，精确 argv、开始/结束时间、库路径和 returncode 在 [commands.json](results/raw/conv_dinput_gpu_01/commands.json)，guard 的完整父命令在 [launch.json](results/raw/conv_dinput_monitor_02/launch.json)。

执行顺序：真实 GPU tests → 30 次正式训练 → 5 次 Nsight + 5 次 SQLite export。只有测试通过才继续训练。GPU tests 环境为 HYTORCH_REQUIRE_CUDA=1、HYTORCH_PRUNING_DEVICE=cuda；新测试数组写入 rawroot/correctness，既有 Fused 测试写入 rawroot/fused-regression。二者分别保留比较 CSV 和实际 NPZ。

A 的子进程 HYTORCH_CUDA_LIBRARY 指向 `rawroot/intentionally_absent_native_library.so`，任何 Native 加载将报错；其余使用 4B 库。此文件并未创建，也不是 fallback。

## 统一测量模板

五组共用：

```bash
python benchmarks/bench_cuda_training.py \
  --device cuda --gpu 0 --monitor-id GPU-158426a4-6684-7863-8cbe-ed6595f97683 \
  --model donkey --samples 256 --batch 4 --epochs 4 --warmup 10 \
  --seed 42 --lr 0.001 --workers 0 --no-monitor --no-fingerprints \
  --save-comparison-inputs --mode throughput --output NEW_DIRECTORY
```

每组附加参数由 [run_experiments.py](scripts/run_experiments.py) 的 CONFIGS 明确列出：A=`--framework hytorch`（其余默认 CuPy）；B 加 shared/im2col/cupy/tile16；C 加 `--fused-sgd`；D 再加 `--prune-input-grad`；E=`--framework torch`。每组独立运行 throughput/phases，各 3 次，每次 256 Step。实际命令全部在 JSON 中展开，不依赖手工替换模板推测参数。

未插桩吞吐与阶段计时均无 NVTX。为避免每个短实验内监控扰动，使用 --no-monitor，硬件由外层单卡 guard 统一采样；summary 中内部 monitor 为零样本是预期，外层原始采样均保留。

详细 Trace 复用 4A profile_training.py，传递同样方法配置，替换为 `--samples 64 --epochs 1 --warmup 10 --mode throughput --nvtx --operator-nvtx`，每组 16 正式 Step：

```bash
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  -o NEW_PREFIX python docs/experiments/donkeynet-backward-profiling/scripts/profile_training.py \
  APPLICATION_ARGUMENTS
nsys export --type sqlite --output NEW_PREFIX.sqlite NEW_PREFIX.nsys-rep
```

实际完整 APPLICATION_ARGUMENTS 在 commands.json。没有新增每算子同步；没有使用 Nsight Compute 或改权限。历史插桩/解析脚本未修改。

## CPU 分析与图表

```bash
python docs/experiments/a100-stage1/run_logged.py \
  --log docs/experiments/conv-dinput-pruning/tests/analysis-01.log -- \
  python docs/experiments/conv-dinput-pruning/scripts/analyze.py \
  --raw runs/conv_dinput_gpu_01 --analysis runs/conv_dinput_trace_analysis_01 \
  --output docs/experiments/conv-dinput-pruning/results \
  --figures docs/experiments/conv-dinput-pruning/figures
```

该脚本调用旧 analyze_trace.py 逐个解析；其五条命令和日志在 [trace analysis](results/raw/conv_dinput_trace_analysis_01/) 中。数值检查直接比较实际数组，未使用哈希。初稿计数图混入 warmup，发现后修改绘图过滤条件，原六图移到 rawroot/figure-drafts-01，重新绘图：

```bash
python docs/experiments/a100-stage1/run_logged.py \
  --log docs/experiments/conv-dinput-pruning/tests/replot-02.log -- \
  python docs/experiments/conv-dinput-pruning/scripts/plot_results.py \
  --results docs/experiments/conv-dinput-pruning/results \
  --figures docs/experiments/conv-dinput-pruning/figures
python docs/experiments/a100-stage1/run_logged.py \
  --log docs/experiments/conv-dinput-pruning/tests/event-review.log -- \
  python docs/experiments/conv-dinput-pruning/scripts/verify_results.py \
  --raw runs/conv_dinput_gpu_01 --analysis runs/conv_dinput_trace_analysis_01 \
  --output docs/experiments/conv-dinput-pruning/results
```

更正只涉及图的统计范围，原始 CSV 不变；最终代码从头复现会直接画正确版本。原图、更正记录及直接 SQLite/逐步事件核查均保留。

## 监控统计与归档

```bash
python docs/experiments/a100-stage1/run_logged.py \
  --log docs/experiments/conv-dinput-pruning/tests/hardware-summary.log -- \
  python docs/experiments/fused-sgd-cuda/scripts/summarize_hardware.py \
  --commands runs/conv_dinput_gpu_01/commands.json \
  --monitor runs/conv_dinput_monitor_02/gpu-load.jsonl \
  --output docs/experiments/conv-dinput-pruning/results/hardware-process-windows.csv
python docs/experiments/a100-stage1/run_logged.py \
  --log docs/experiments/conv-dinput-pruning/tests/archive-01.log -- \
  python docs/experiments/fused-sgd-cuda/scripts/archive_results.py \
  --destination docs/experiments/conv-dinput-pruning/results/raw \
  runs/conv_dinput_cpu_tests_01 runs/conv_dinput_gpu_01 \
  runs/conv_dinput_trace_analysis_01 runs/conv_dinput_monitor_01 runs/conv_dinput_monitor_02
```

复用 4B 归档工具，未改旧文件。超过 1 MB 的原始文本无损 gzip，原文件全部保留；索引只记录路径、大小和副本位置，不计算内容哈希。Nsight/SQLite/NPZ 和原图稿不进 Git。

## Git

远端只读确认基线仍为指定提交，目标分支此前不存在。检查本轮代码、数据、链接和修改范围后，按授权提交并普通推送 `experiment/conv-backward-dinput-pruning`；不 force push，不合并 main。最终提交 ID 在交付消息和 Git 历史中提供。本轮到此停止。
