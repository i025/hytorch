# 阶段4G执行命令

工作目录 `/home/xymo/projects/hytorch`（规范路径`/srv/lab-disks/disk2/users/xymo/projects/hytorch`）。下列为本次实际目录；重现时需换新的输出后缀，脚本拒绝覆盖原始实验。命令准确argv、环境、时间、返回码以[logs](logs)中的`*.command.json`及GPU[commands.json](raw-records/conv_dweight_window_02/results/commands.json)为准；不执行哈希校验。

## A：CPU准备与编译

```bash
cd /home/xymo/projects/hytorch
source docs/experiments/a100-stage2a-gemm/environment/activate.sh
# 此时 CUDA_VISIBLE_DEVICES=''；CUDA_PATH=/usr/local/cuda-12.2
python docs/experiments/conv-dweight-fusion/scripts/history.py
python -m pytest \
  tests/test_conv_dweight_host.py tests/test_conv_dinput_opt_host.py \
  tests/test_conv_shape_dispatch.py tests/test_conv_backward_fusion_host.py \
  tests/test_conv_dinput_pruning.py tests/test_cuda_host.py \
  tests/test_conv2d_gemm_cpu.py tests/test_gemm_blocked_packed.py \
  -q -ra --junitxml=docs/experiments/conv-dweight-fusion/results/host-tests-03.xml
python docs/experiments/conv-dweight-fusion/scripts/check_cpu.py \
  --output runs/conv_dweight_cpu_01
python scripts/build_cuda.py --arch 80 --resource-usage \
  --output runs/conv_dweight_build_01
cuobjdump --dump-resource-usage runs/conv_dweight_build_01/libhytorch_cuda.so
HYTORCH_CUDA_LIBRARY=/home/xymo/projects/hytorch/runs/conv_dweight_build_01/libhytorch_cuda.so \
  python -c 'from hytorch.backends.cuda.runtime import load_library; lib=load_library(); print("ABI",lib.hy_cuda_abi_version(),"dWeight symbol",bool(lib.hy_conv_dweight))'
python docs/experiments/conv-dweight-fusion/scripts/analyze_micro_trace.py \
  --input runs/conv_dinput_opt_traces_wait_06/attempt-01/results/micro/kernel-history-fused-r1.sqlite \
  --output runs/conv_dweight_parser_cpu_01 --expected-cases 100
python -m pytest tests/test_conv_dweight_gpu.py --collect-only -q
python -m compileall -q docs/experiments/conv-dweight-fusion/scripts \
  tests/test_conv_dweight_host.py tests/test_conv_dweight_gpu.py
python docs/experiments/conv-dweight-fusion/scripts/run_window.py \
  --output runs/conv_dweight_window_02/results \
  --gpu-uuid GPU-f099fdbd-adf2-6157-3462-4145bc2a8359 --physical-gpu 0 \
  --library runs/conv_dweight_build_01/libhytorch_cuda.so \
  --plan docs/experiments/conv-dweight-fusion/results/gpu-plan-02.json
```

CPU回归01/02/03是逐次新增profiling/释放接口测试后的74/75/76通过，全部独立XML和日志保留。历史Trace解析比较是CPU离线验证，不能冒充4G GPU性能。

每条关键命令通过既有日志器保存完整stdout/stderr及命令元数据，例如：

```bash
python docs/experiments/a100-stage1/run_logged.py \
  --log docs/experiments/conv-dweight-fusion/logs/build-01.log -- \
  python scripts/build_cuda.py --arch 80 --resource-usage --output runs/conv_dweight_build_01
```

## B：GPU0完整补测窗口

选择前只读记录8卡状态与调度环境：[gpu-selection-01.json](environment/gpu-selection-01.json)。首次窗口错误和例外授权见[window-01-failure.json](results/window-01-failure.json)。用户明确允许同GPU0一次完整补测，统一入口从头连续执行所有任务：

```bash
source docs/experiments/a100-stage2a-gemm/environment/activate.sh
python docs/experiments/a100-stage1/run_logged.py \
  --log docs/experiments/conv-dweight-fusion/logs/gpu-window-02.log -- \
  python docs/experiments/conv-dinput-optimization/scripts/run_gpu.py \
    --gpu GPU-f099fdbd-adf2-6157-3462-4145bc2a8359 --idle-memory-mib 4 \
    --record runs/conv_dweight_window_02/guard -- \
    python docs/experiments/conv-dweight-fusion/scripts/run_window.py \
      --output runs/conv_dweight_window_02/results \
      --gpu-uuid GPU-f099fdbd-adf2-6157-3462-4145bc2a8359 --physical-gpu 0 \
      --library runs/conv_dweight_build_01/libhytorch_cuda.so
```

Guard将UUID写入CUDA_VISIBLE_DEVICES，控制器只用逻辑0，跨子进程保持初始化context，按76项顺序执行；全部结束才reset。Guard检测其他用户进程冲突时停止本轮，不换卡，不预留显存、不做忙循环。

完整序列：

1. 新增GPU277项，再既有回归446项，均要求0跳过。
2. 三种dWeight实现×三重复，每次全部132形状、5预热+30调用。
3. 两模型×throughput/phases×A/F/I/E×三重复，共48进程，每次10预热+256正式step。
4. 两模型×四配置训练Nsight，共8份；再三实现×三重复微基准Nsight，共9份。
5. 写window.json，`cudaDeviceReset=0`，释放后再做CPU分析。

所有子任务逐条实际argv、起止、返回码、Native库路径见[76项commands.json](raw-records/conv_dweight_window_02/results/commands.json)，无需猜测或省略参数。代表性子命令：

```bash
python -m pytest tests/test_conv_dweight_gpu.py -q -ra \
  --junitxml=runs/conv_dweight_window_02/results/correctness-new.xml
python -m pytest tests/test_conv_dinput_opt_gpu.py tests/test_conv_shape_dispatch_gpu.py \
  tests/test_conv_backward_fusion_gpu.py tests/test_conv_dinput_pruning.py \
  tests/test_fused_sgd_gpu.py tests/test_cuda_device.py -q -ra \
  --junitxml=runs/conv_dweight_window_02/results/correctness-regression.xml
python docs/experiments/conv-dweight-fusion/scripts/bench_shapes.py \
  --method fused --repeat 1 --output runs/conv_dweight_window_02/results/micro-fused-r1
python benchmarks/bench_cuda_training.py --device cuda --gpu 0 \
  --monitor-id GPU-f099fdbd-adf2-6157-3462-4145bc2a8359 \
  --model donkey --batch 4 --warmup 10 --seed 42 --lr 0.001 --workers 0 \
  --no-monitor --no-fingerprints --save-comparison-inputs --samples 256 --epochs 4 \
  --mode throughput --framework hytorch --matmul shared --conv im2col \
  --pool cupy --tile 16 --fused-sgd --prune-input-grad --conv-backward fused \
  --conv-dweight fused --output runs/conv_dweight_window_02/results/training-donkey-throughput-I-r1
```

`samples256 × epochs4 / batch4 = 256step`，每次实测1024samples。A只有`--framework hytorch`，其Native库环境被设为不存在的路径；F与I只差`--conv-dweight fused`；E为`--framework torch`。Spatial只替换模型。顺序按重复轮换AFIE→FIEA→IEAF。

Nsight统一外层为`nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none -o OUTPUT`；训练内部改用`scripts/profile_training.py`，`--samples 64 --epochs 1 --mode throughput --nvtx --operator-nvtx`，HyTorch另有`--memory-stats`。微基准内部增加`--nvtx`。完整命令见commands.json，Nsight执行完不在GPU窗口导出SQLite。

## C：释放后离线分析与归档

```bash
source docs/experiments/a100-stage2a-gemm/environment/activate.sh
# 恢复 CUDA_VISIBLE_DEVICES=''
nvidia-smi --id=GPU-f099fdbd-adf2-6157-3462-4145bc2a8359 \
  --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv
python docs/experiments/conv-dweight-fusion/scripts/postprocess.py \
  --raw runs/conv_dweight_window_02/results --analysis runs/conv_dweight_analysis_01 \
  --output docs/experiments/conv-dweight-fusion/results/acceptance-01
```

首版导出后发现PyTorch worker Kernel漏入Backward窗口的问题，原始数据和旧统计不覆盖。修正归因后仅复用CPU导出/已解析事件：

```bash
python docs/experiments/conv-dweight-fusion/scripts/postprocess.py \
  --raw runs/conv_dweight_window_02/results --analysis runs/conv_dweight_analysis_01 \
  --output docs/experiments/conv-dweight-fusion/results/acceptance-03 --reuse-analysis
python docs/experiments/conv-dweight-fusion/scripts/summarize_acceptance.py \
  --raw runs/conv_dweight_window_02/results \
  --results docs/experiments/conv-dweight-fusion/results/acceptance-03
python docs/experiments/conv-dweight-fusion/scripts/plot_results.py \
  --results docs/experiments/conv-dweight-fusion/results/acceptance-03 \
  --output docs/experiments/conv-dweight-fusion/figures/final-03
python docs/experiments/fused-sgd-cuda/scripts/archive_results.py \
  --destination docs/experiments/conv-dweight-fusion/raw-records \
  runs/conv_dweight_build_01 runs/conv_dweight_cpu_01 runs/conv_dweight_parser_cpu_01 \
  runs/conv_dweight_window_01 runs/conv_dweight_window_02 runs/conv_dweight_analysis_01
```

首版汇总/绘图也保存在acceptance-01/final-01。postprocess-02因拒绝覆盖旧日志退出，失败日志保留，最终输出03未覆盖此前文件。GPU未重跑。

17份Nsight导出及解析的每条真实命令在[analysis commands.jsonl](raw-records/conv_dweight_analysis_01/commands.jsonl)：`nsys export --type sqlite --output DATABASE TRACE`，微基准用本轮`analyze_micro_trace.py`，训练复用4A `analyze_trace.py`。完整归因CSV、每个子分析日志保留；最终重汇总命令另存[acceptance-03/commands.jsonl](results/acceptance-03/commands.jsonl)。

## 发布

GPU释放后检查差异，仅加入本轮源码、测试、脚本和4G记录及根README链接。分支`experiment/conv-dweight-fusion`；普通commit和push，不force、不合并main。认证只通过已有Git凭据缓存，凭据脚本/Token不进入Git。提交完成后停止，等待人工审查，不运行4H。
