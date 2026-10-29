# 执行记录与待执行命令

仓库 `/home/xymo/projects/hytorch`，真实路径 `/srv/lab-disks/disk2/users/xymo/projects/hytorch`。所有日志均新建；复现用新的输出目录。本轮未执行任何 GPU Kernel，不计算内容哈希。

## 实际执行：分支、环境与 CPU

确认工作区干净、当前 HEAD 为 `549f2ee4f9c046b3eeab2de1a54a47a5a2a16310` 后：

```bash
git switch -c experiment/conv-backward-fusion
source docs/experiments/a100-stage2a-gemm/environment/activate.sh
```

激活现有独立 .venv，CUDA_VISIBLE_DEVICES 为空，CUDA_PATH=/usr/local/cuda-12.2，OMP/OPENBLAS 线程数 4，CUPY_TF32=0，NVIDIA_TF32_OVERRIDE=0；未安装依赖。

环境完整 argv 在 [system.txt.command.json](environment/system.txt.command.json)，完整输出在 [system.txt](environment/system.txt)。其 bash 命令依次执行 date -u、hostname、id、/etc/os-release、lscpu、以下两条查询、Python/pip/GCC/nvcc/nsys 版本：

```bash
nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.total,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
```

后续空闲检查的实际查询为：

```bash
nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits
```

结果以 CSV 解析为 JSON，空闲要求无计算客户端、memory.used=0、utilization.gpu=0；实际全部不满足，selected_gpu=null。保存于 [gpu-availability-02.json](environment/gpu-availability-02.json)、[终端日志](environment/gpu-availability-02.log)。这只是读取占用，不占用 GPU 或改变任务。

提交前还保存了 [gpu-final.log](environment/gpu-final.log) 和 [完整 argv](environment/gpu-final.log.command.json)：date -u、index/uuid/memory.used/utilization.gpu 查询及同一 compute-apps 查询，仍无空闲卡。

```bash
python docs/experiments/a100-stage1/run_logged.py \
  --log docs/experiments/conv-backward-fusion/tests/host-01.log -- \
  python -m pytest tests/test_conv_backward_fusion_host.py tests/test_conv_dinput_pruning.py \
  tests/test_fused_sgd_host.py tests/test_cuda_host.py tests/test_conv2d_gemm_cpu.py \
  tests/test_gemm_blocked_packed.py -q -ra \
  --junitxml=docs/experiments/conv-backward-fusion/tests/host-01.xml
```

添加插桩还原测试后定向执行，不重跑无关回归：

```bash
python docs/experiments/a100-stage1/run_logged.py \
  --log docs/experiments/conv-backward-fusion/tests/host-02.log -- \
  python -m pytest tests/test_conv_backward_fusion_host.py -q -ra \
  --junitxml=docs/experiments/conv-backward-fusion/tests/host-02.xml
python docs/experiments/a100-stage1/run_logged.py \
  --log docs/experiments/conv-backward-fusion/tests/gpu-collection.log -- \
  python -m pytest tests/test_conv_backward_fusion_gpu.py tests/test_conv_dinput_pruning.py \
  tests/test_fused_sgd_gpu.py tests/test_cuda_device.py --collect-only -q
```

最后一条只收集 92 项测试，没有执行或报告它们通过。实际 GPU fixture 需 HYTORCH_REQUIRE_CUDA=1，执行时库缺失/设备故障应失败。

## 实际执行：编译与主机加载

```bash
python docs/experiments/a100-stage1/run_logged.py \
  --log docs/experiments/conv-backward-fusion/tests/build-01.log -- \
  python scripts/build_cuda.py --arch 80 --resource-usage \
  --output runs/conv_backward_fusion_build_01
python docs/experiments/a100-stage1/run_logged.py \
  --log docs/experiments/conv-backward-fusion/tests/library-load-01.log -- \
  python -c 'import ctypes; lib=ctypes.CDLL("runs/conv_backward_fusion_build_01/libhytorch_cuda.so"); print("ABI", lib.hy_cuda_abi_version()); print("hy_conv_dinput symbol", bool(lib.hy_conv_dinput)); print("Host library load only; no CUDA kernel submitted")'
```

nvcc 完整展开命令在 [build.json](results/raw/conv_backward_fusion_build_01/build.json)。没有使用需要空闲 GPU 自动检测架构的 --device 模式。

## 实际执行：CPU 方程、历史分析、计划

```bash
python docs/experiments/a100-stage1/run_logged.py \
  --log docs/experiments/conv-backward-fusion/tests/cpu-equation-01.log -- \
  python docs/experiments/conv-backward-fusion/scripts/check_gather_cpu.py \
  --output runs/conv_backward_fusion_cpu_equation_01
python docs/experiments/a100-stage1/run_logged.py \
  --log docs/experiments/conv-backward-fusion/tests/history-analysis-01.log -- \
  python docs/experiments/conv-backward-fusion/scripts/analyze_history.py \
  --output docs/experiments/conv-backward-fusion/results/history
python docs/experiments/a100-stage1/run_logged.py \
  --log docs/experiments/conv-backward-fusion/tests/plan-01.log -- \
  python docs/experiments/conv-backward-fusion/scripts/run_experiments.py \
  --output runs/conv_backward_fusion_plan_01 \
  --library runs/conv_backward_fusion_build_01/libhytorch_cuda.so \
  --gpu-uuid UNALLOCATED --plan-only
```

check_gather_cpu 为 8 组固定 FP32 输入、边界和 seed42 随机抽样的位置，与原 NumPy / PyTorch CPU 比较。计划文件所有 64 个 argv 在 [planned-commands.json](results/raw/conv_backward_fusion_plan_01/planned-commands.json)，不是成功实验。计划中的 UNALLOCATED 不能用于实际执行。

脚本语法解析、CPU ATen schema 烟测和 CLI 检查有 [script-review.log](tests/script-review.log) 和其 command.json；CPU ATen 实际验证了三个梯度的输出形状，不涉及 CUDA。

## 实际执行：归档与绘图

```bash
python docs/experiments/a100-stage1/run_logged.py \
  --log docs/experiments/conv-backward-fusion/tests/archive-01.log -- \
  python docs/experiments/fused-sgd-cuda/scripts/archive_results.py \
  --destination docs/experiments/conv-backward-fusion/results/raw \
  runs/conv_backward_fusion_build_01 runs/conv_backward_fusion_cpu_equation_01 \
  runs/conv_backward_fusion_plan_01
```

CPU 图从归档后的 summary.json 绘制，历史图复制自 history 分析产物；执行的内联绘图代码已整理为 [plot_cpu_results.py](scripts/plot_cpu_results.py)。下面命令会在新目录重建这两张实际数据图：

```bash
python docs/experiments/conv-backward-fusion/scripts/plot_cpu_results.py \
  --experiment docs/experiments/conv-backward-fusion --figures NEW_CPU_FIGURES_DIRECTORY
```

已用该脚本重建到 `runs/conv_backward_fusion_figures_01`，实际命令及输出在 [plot-reproduction.log.command.json](tests/plot-reproduction.log.command.json) 和 [log](tests/plot-reproduction.log)，没有覆盖初次生成的图。

## 待执行：空闲 GPU 的整批实验（NOT_RUN）

只有取得空闲且获准的 UUID 后，沿用现有 guard。以下为复现模板，**本轮未执行**；不应在当前已占用设备上运行。

```bash
source docs/experiments/a100-stage2a-gemm/environment/activate.sh
python docs/experiments/donkeynet-backward-profiling/scripts/run_gpu.py \
  --gpu ALLOCATED_IDLE_UUID --record NEW_MONITOR_DIRECTORY -- \
  python docs/experiments/conv-backward-fusion/scripts/run_experiments.py \
  --output NEW_GPU_OUTPUT_DIRECTORY \
  --library runs/conv_backward_fusion_build_01/libhytorch_cuda.so \
  --gpu-uuid ALLOCATED_IDLE_UUID
```

guard 两次确认空闲，通过 UUID 设置 CUDA_VISIBLE_DEVICES，逻辑设备为 0，持续检查其他计算客户端；有外部客户端时停止本轮进程。全部子进程 argv、开始/结束和退出码保存在新输出的 commands.json，任一步失败停止，日志保留。

A 的 Native 库路径故意指向不存在的文件；误调 Native 会明确失败。后处理还会检查 A 的 dispatch 和全 Trace Kernel 名。F 必须在每次 256 Step 训练记录 1024 次 conv_dinput_fused，D/F 首层均不应出现 dInput。GPU 测试的全部数组保存在 correctness/，旧回归数组分开保存。

三方法微基准各 3 次：每次 Conv2–5、dinput/full 两范围、5 warmup +30 正式；训练 A/B/C/D/F/E 各 3 次吞吐和 3 次阶段计时，10 warmup +256正式Step；顺序随 repeat 轮换。9 次 Nsight 是另起进程，参数 --trace=cuda,nvtx --sample=none --cpuctxsw=none；训练 Trace 16 正式 Step，微基准 Trace 每层/范围 3 次正式调用，仅用于归因。

## 待执行：真实结果分析（NOT_RUN）

```bash
python docs/experiments/conv-backward-fusion/scripts/analyze_results.py \
  --raw NEW_GPU_OUTPUT_DIRECTORY --analysis NEW_TRACE_CSV_DIRECTORY \
  --output NEW_GPU_RESULT_DIRECTORY
python docs/experiments/conv-backward-fusion/scripts/plot_results.py \
  --results NEW_GPU_RESULT_DIRECTORY --analysis NEW_TRACE_CSV_DIRECTORY \
  --figures NEW_GPU_FIGURES_DIRECTORY
```

实际长训练/微基准的比较失败先保留 CSV，再抛错；不改变容差。Nsight 使用旧 correlation 解析器，复制/布局/Kernel/APIs/NVTX 原始 CSV 和 schema 保留；不通过 GPU 异步时间重叠猜测父节点。

最后按授权检查源码和数据，git commit、普通 git push 到 experiment/conv-backward-fusion；不 force push、不合并 main。最终提交 ID 在交付消息与 Git 历史中提供。
