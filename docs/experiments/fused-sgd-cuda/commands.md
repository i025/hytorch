# 阶段 4B 执行命令

工作目录 `/home/xymo/projects/hytorch`（真实路径 `/srv/lab-disks/disk2/users/xymo/projects/hytorch`）。以下是已执行命令；复现时必须选择当时空闲且获准的 GPU，并使用新的输出目录，不能覆盖本轮数据。没有执行哈希/fingerprint 验证、sudo 或 GPU 权限修改。

## 分支、环境和编译

从干净的基线 `8e803e1007ad56d3255a2ea75f78517b62cbdbe1` 创建：

```bash
git switch -c experiment/fused-sgd-cuda
source docs/experiments/a100-stage2a-gemm/environment/activate.sh
```

该激活脚本使用仓库独立 `.venv`，CUDA Toolkit 12.2，`OPENBLAS_NUM_THREADS=4`、`OMP_NUM_THREADS=4`、`CUPY_TF32=0`、`NVIDIA_TF32_OVERRIDE=0`，初始 `CUDA_VISIBLE_DEVICES=''`。完整环境采集的 argv/stdout 保存在 [system command](environment/server-system.txt.command.json)、[system stdout](environment/server-system.txt)；GCC/PyTorch/cuDNN 为 [extra command](environment/extra-versions.txt.command.json)、[extra stdout](environment/extra-versions.txt)。

```bash
python docs/experiments/a100-stage1/run_logged.py \
  --log docs/experiments/fused-sgd-cuda/logs/build-01.log -- \
  python scripts/build_cuda.py --arch 80 --output runs/fused_sgd_cuda_build_01
```

此步骤仅编译，不需初始化 GPU。完整 nvcc argv 见 [build.json](results/raw/fused_sgd_cuda_build_01/build.json)。额外离线 PTX/cubin 检查命令为 [PTX](results/raw/fused_sgd_cuda_ptx_01/command.txt)、[cubin](results/raw/fused_sgd_cuda_ptx_01/cubin-command.txt)；编译器输出和资源数同目录保留。

## 主机回归

```bash
python docs/experiments/a100-stage1/run_logged.py \
  --log docs/experiments/fused-sgd-cuda/logs/host-regression-01.log -- \
  python -m pytest tests/test_cuda_host.py tests/test_conv2d_gemm_cpu.py \
  tests/test_gemm_blocked_packed.py -q -ra \
  --junitxml=docs/experiments/fused-sgd-cuda/logs/host-regression-01.xml

CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m pytest tests/test_fused_sgd_host.py -q \
  --junitxml=docs/experiments/fused-sgd-cuda/logs/fused-host-tests-01.xml \
  > docs/experiments/fused-sgd-cuda/logs/fused-host-tests-01.log 2>&1
```

形状查询完整 Python stdin 记录在 [parameter-shapes-cpu.command.json](logs/parameter-shapes-cpu.command.json)。

## 单卡 GPU 验证、计时与 Trace

```bash
source docs/experiments/a100-stage2a-gemm/environment/activate.sh
python docs/experiments/donkeynet-backward-profiling/scripts/run_gpu.py \
  --gpu GPU-158426a4-6684-7863-8cbe-ed6595f97683 \
  --record runs/fused_sgd_cuda_monitor_01 -- \
  python docs/experiments/fused-sgd-cuda/scripts/run_experiments.py \
  --output runs/fused_sgd_cuda_01 \
  --library runs/fused_sgd_cuda_build_01/libhytorch_cuda.so \
  --gpu-uuid GPU-158426a4-6684-7863-8cbe-ed6595f97683
```

guard 将该 UUID 设置为 `CUDA_VISIBLE_DEVICES`，进程逻辑 0 对应物理 2。执行器设 `HYTORCH_REQUIRE_CUDA=1`、`HYTORCH_CUDA_LIBRARY=<本轮新库绝对路径>`、`HYTORCH_SGD_TEST_OUTPUT=<rawroot>/correctness`。首次 GPU 子命令是 18 项真实测试，失败会停止后续测量，未将 unavailable 变成 skip。

**40 条实际子命令、开始/结束 UTC、返回码、日志路径均在 [commands.json](results/raw/fused_sgd_cuda_01/commands.json)**；全部 returncode=0。顺序为：GPU tests；9 次独立 Update；18 次训练；6 次 Nsight capture + 6 次 SQLite export。按 repeat 轮换 baseline/fused/torch 顺序，单卡串行运行。

核心参数如下，实际展开 argv 以上述 JSON 为准：

```bash
# 独立 Update，每方法每次独立进程；method=baseline/fused/torch，r=1/2/3
python docs/experiments/fused-sgd-cuda/scripts/bench_update.py \
  --method fused --warmup 5 --updates 100 --repeats 1 --output NEW_DIRECTORY

# HyTorch Fused 匹配训练；原 SGD 去掉 --fused-sgd
python benchmarks/bench_cuda_training.py \
  --framework hytorch --device cuda --gpu 0 \
  --monitor-id GPU-158426a4-6684-7863-8cbe-ed6595f97683 \
  --model donkey --samples 256 --batch 4 --epochs 4 --warmup 10 \
  --seed 42 --lr 0.001 --workers 0 --no-monitor --no-fingerprints \
  --save-comparison-inputs --matmul shared --conv im2col --pool cupy --tile 16 \
  --fused-sgd --mode throughput --output NEW_DIRECTORY
```

每方法独立运行 throughput/phases 两种计时，各重复 3 次。PyTorch 设置 `--framework torch`，去除 HyTorch 后端和 fused 参数，其余匹配。训练脚本默认 DonkeyNet dropout=0；PyTorch foreach SGD、TF32关闭和权重布局映射复用既有逻辑。

Nsight 实际调用格式：

```bash
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  -o NEW_PREFIX python APPLICATION.py APPLICATION_ARGUMENTS --nvtx
nsys export --type sqlite --output NEW_PREFIX.sqlite NEW_PREFIX.nsys-rep
```

独立 Update 的 APPLICATION 为 `bench_update.py`，每方法各两个模式 100 次更新。训练为本目录 `scripts/profile_training.py`，使用 `--samples 64 --epochs 1 --warmup 10 --mode throughput --nvtx`，其余相同。本轮没有运行 Nsight Compute。新插桩仅包裹 SGD metadata/native/ctypes 主机路径，不额外同步；trace 的吞吐数字不用于正式性能比较。

## CPU 分析、作图和数据归档

六份 SQLite 逐一运行既有 `docs/experiments/donkeynet-backward-profiling/scripts/analyze_trace.py`，未修改旧脚本。准确 argv/returncode 见 [analysis-commands.jsonl](results/raw/fused_sgd_cuda_analysis_01/analysis-commands.jsonl)，包括新汇总和独立核查命令：

```bash
python docs/experiments/donkeynet-backward-profiling/scripts/analyze_trace.py \
  --input runs/fused_sgd_cuda_01/trace-training-fused.sqlite \
  --output runs/fused_sgd_cuda_analysis_01/trace-training-fused --label trace-training-fused
python docs/experiments/fused-sgd-cuda/scripts/analyze_sgd_traces.py \
  --analysis-root runs/fused_sgd_cuda_analysis_01 \
  --output docs/experiments/fused-sgd-cuda/results/trace
python docs/experiments/fused-sgd-cuda/scripts/check_trace_results.py \
  --analysis-root runs/fused_sgd_cuda_analysis_01 \
  --summary-root docs/experiments/fused-sgd-cuda/results/trace \
  --correctness-root runs/fused_sgd_cuda_01/correctness \
  --output docs/experiments/fused-sgd-cuda/results/trace/independent-review.json
python docs/experiments/fused-sgd-cuda/scripts/analyze_fused_results.py \
  --raw runs/fused_sgd_cuda_01 --trace-analysis runs/fused_sgd_cuda_analysis_01 \
  --output docs/experiments/fused-sgd-cuda/results \
  --figures docs/experiments/fused-sgd-cuda/figures
```

图稿复核后只调整 phase 图的 legend 位置，旧六图保留于 rawroot 的 `figure-review-01/`；重绘记录见 [analysis-review.log](logs/analysis-review.log)，不覆盖测量数据。

```bash
python docs/experiments/a100-stage1/run_logged.py \
  --log docs/experiments/fused-sgd-cuda/logs/hardware-summary-01.log -- \
  python docs/experiments/fused-sgd-cuda/scripts/summarize_hardware.py \
  --commands runs/fused_sgd_cuda_01/commands.json \
  --monitor runs/fused_sgd_cuda_monitor_01/gpu-load.jsonl \
  --output docs/experiments/fused-sgd-cuda/results/hardware-process-windows.csv
python docs/experiments/a100-stage1/run_logged.py \
  --log docs/experiments/fused-sgd-cuda/logs/archive-01.log -- \
  python docs/experiments/fused-sgd-cuda/scripts/archive_results.py \
  --destination docs/experiments/fused-sgd-cuda/results/raw \
  runs/fused_sgd_cuda_01 runs/fused_sgd_cuda_analysis_01 \
  runs/fused_sgd_cuda_monitor_01 runs/fused_sgd_cuda_build_01 runs/fused_sgd_cuda_ptx_01
```

归档复制可管理的原始文本；大于 1 MB 的文本无损 gzip，仅压缩未计算内容校验。二进制、数组和 profiler 原文件不进 Git，文件索引只记录路径/字节数/副本位置。归档后新增的图稿索引另列在数据索引中。

## 人工复用此后端

在重新确认空闲并分配 GPU 后，使用上面的 activation、选定 UUID 和本轮新库路径；Python 接口为 `SGD(lr=0.001, fused=True)`。普通训练 CLI 加 `--fused-sgd --no-fingerprints`。无需重编译或替换已有实验使用的旧库。

## Git 发布

发布前检查代码、文档链接、全部新增文件大小、原始数据索引及差异；原 CUDA kernels、历史报告和精度原型没有变更。远端只读检查基线仍为 `8e803e1007ad56d3255a2ea75f78517b62cbdbe1`，目标分支此前不存在。按用户授权提交并以普通 `git push -u origin experiment/fused-sgd-cuda` 发布，不 force push，不合并 main；最终提交 ID 在交付消息和 Git 历史中提供，不为写入自身 ID 再创建递归提交。

发布文档检查最初发现主报告 3 处相对路径多一层父目录，已修正；失败记录 `logs/publication-check-01.json` 和修正后 `logs/publication-check-02.json` 均保留。这是文档链接修正，与 GPU 数值/性能实验无关。
