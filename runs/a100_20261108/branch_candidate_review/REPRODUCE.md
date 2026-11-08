# 复现

先确认GPU空闲且有权限。本次使用物理GPU1，结束时0MiB；禁止影响GPU0上的服务。以下GPU命令只在再次确认GPU1可用后运行。

```bash
cd /home/xieminhui_pubuse/hytorch
nvidia-smi
export CUDA_VISIBLE_DEVICES=1 NVIDIA_TF32_OVERRIDE=0
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=.
export PATH=/usr/local/cuda/bin:$PATH
PY=/home/xieminhui_pubuse/hytorch-env/bin/python
$PY scripts/build_cuda.py --arch 80
HYTORCH_REQUIRE_CUDA=1 $PY -m pytest -q tests
HYTORCH_REQUIRE_CUDA=1 HYTORCH_PRUNING_DEVICE=cuda $PY -m pytest -q tests/test_relu_fusion.py tests/test_conv_dinput_pruning.py tests/test_backward_schedule.py
/usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 99 $PY -m pytest -q tests/test_relu_fusion.py
/usr/local/cuda/bin/compute-sanitizer --tool racecheck --error-exitcode 99 $PY -m pytest -q tests/test_relu_fusion.py
```

## 推荐开启方式

```bash
$PY benchmarks/bench_cuda_training.py --device cuda --gpu 0 --model donkey   --samples 256 --batch 4 --epochs 3 --final-config   --relu-backward fused --no-input-grad   --monitor --monitor-components disabled --output runs/candidate_training_fresh
```

--monitor --monitor-components disabled使用统一异步训练helper，但不启用监控；不传--monitor则使用旧StepProfiler入口，其默认phase同步会改变性能口径。需要资源监控时选resources/all并单独报告监控开销。heap可通过--backward-schedule heap显式试验，DonkeyNet不推荐默认加上。输入梯度用于对抗训练/解释等时，不要指定--no-input-grad。

Python API：`Variable(x, requires_grad=False)`；CUDA配置上下文新增`backend(..., relu_backward="fused")`；调度`using_config("backward_schedule", "heap")`。上下文恢复原配置，不缓存设备指针。

## 完整配对实验

不要覆盖现有记录。将diagnostics复制到同深度新目录（例如runs/a100_20261108/branch_candidate_review_rerun/diagnostics），创建raw子目录后运行：

```bash
$PY NEW/diagnostics/experiment.py > NEW/raw/training.log 2>&1
$PY NEW/diagnostics/micro.py > NEW/raw/micro.log 2>&1
```

脚本从自身路径推导仓库和输出；NEW需替换成上述同深度目录。experiment默认12轮DonkeyNet六模式、6轮TinyNet三模式；固定数据/seed/init，每组10step预热，不修改门限。结果完整读回CPU在正式计时外。

精确原版对照需原始4317bf5工作区及其CUDA库。本次保留在/home/xieminhui_pubuse/hytorch-before-candidate-review。新机器可`git worktree add --detach /path/to/baseline 4317bf5`，用相同CUDA编译器/flags在该目录构建（原版build脚本--arch80）。修改paired_processes.py中的roots为实际两个工作区，再运行该脚本。其子进程设置HYTORCH_REVIEW_ROOT，保持同一数据与初始化；该环境变量也可用于独立experiment.py --worker baseline。baseline .so的原SHA见baseline/hashes.json。

Nsight完整命令见diagnostics/gpu_validation.sh；原版和候选各20steps，cudaProfilerApi截取，NVTX+CUDA correlation归因。旧系统/usr/bin sanitizer不可用，本次使用/usr/local/cuda/bin；Nsight使用/opt/nvidia/nsight-systems/2025.1.3/bin/nsys。

GPU实验结束后再独立运行heap_cpu.py，避免CPU负载干扰正式训练；它不使用GPU。已有CSV与SQLite齐全后运行analyze.py、figures.py。CSV单位在列名中：wall/delta/CI为秒，CPU/event/micro为μs，HEAP_CPU为ms。micro是完整分配+调用的Event区间，不是纯kernel时间；PROFILE来自独立诊断不能替代正式wall。

CUDA运行环境与原构建参数沿用目标分支；PyTorch在CuPy首次设备调用之前导入，避免本机混合CUDA runtime加载顺序问题。二进制trace/SQLite、权重和.so只在服务器，不随Git发布；EXTERNAL_ARTIFACTS.json列出路径/SHA。
