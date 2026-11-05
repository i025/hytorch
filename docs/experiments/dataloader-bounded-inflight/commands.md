# 命令与复现

工作目录`/home/xymo/projects/hytorch`。既有`.venv`，不安装依赖、不调用GPU。
每项CPU计算沿用以下命令环境，并使用`taskset -c 0-3 nice -n 10`：

```bash
export CUDA_VISIBLE_DEVICES=''
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=.
```

输出目录必须唯一。保留本轮所有原始数据，复跑时不要覆盖以下路径。

```bash
git switch -c experiment/dataloader-bounded-inflight
git show dc77d06:hytorch/dataloaders.py > runs/dataloader_bounded_01/baseline_dataloaders.py
```

原实现4项失败复现，在修改DataLoader前执行：

```bash
.venv/bin/python -m pytest tests/test_dataloader_bounded.py -k slow_first -q -ra --junitxml=docs/experiments/dataloader-bounded-inflight/results/bound-before-01.xml
```

实现后，初次专项36通过（随后加入2项非法窗口测试，包含于最终139项回归）：

```bash
.venv/bin/python -m pytest tests/test_dataloader_bounded.py tests/test_dataloader_indices.py tests/test_dataloader_lifecycle.py -q -ra --junitxml=docs/experiments/dataloader-bounded-inflight/results/loader-after-01.xml
```

完整CPU回归：

```bash
CUDA_PATH=/usr/local/cuda-12.2 .venv/bin/python -m pytest \
 tests/test_cuda_host.py tests/test_conv2d_gemm_cpu.py tests/test_gemm_blocked_packed.py \
 tests/test_fused_sgd_host.py tests/test_conv_dinput_pruning.py tests/test_conv_dweight_host.py \
 tests/test_conv_dinput_opt_host.py tests/test_udacity_dataset.py tests/test_udacity_experiment.py \
 tests/test_dataloader_indices.py tests/test_relu_backward_host.py tests/test_conv_backward_fusion_host.py \
 tests/test_conv_shape_dispatch.py tests/test_dataloader_lifecycle.py tests/test_cpu_pipeline_plan.py \
 tests/test_dataloader_bounded.py -q -ra \
 --junitxml=docs/experiments/dataloader-bounded-inflight/results/cpu-regression-01.xml
```

CUDA_PATH只让一项既有CPU测试解析libcudart符号，不调用设备函数。
这些源码与脚本在正式性能前本地提交为`29fa065a5fd136221f760e764b05f8a22171696a`，
正式性能每次记录该Git版本，最终报告提交不会改变测试过的生产代码。

以下命令均加公共CPU环境及affinity/nice；使用既有日志包装器保存真实argv、
stdout/stderr、时间和返回值，例如：

```bash
.venv/bin/python docs/experiments/a100-stage1/run_logged.py --log docs/experiments/dataloader-bounded-inflight/logs/benchmark-01.log -- \
 .venv/bin/python docs/experiments/dataloader-bounded-inflight/scripts/benchmark.py \
 --prepared runs/udacity_real_prepared_01 \
 --baseline runs/dataloader_bounded_01/baseline_dataloaders.py \
 --output runs/dataloader_bounded_01/benchmark-01
```

132个正式子命令及4个长度观察均保存在`benchmark-01/commands.json`，逐次summary
记录开始/结束时间、版本、CPU环境与内存。各配置三重复，seed42改变配置执行顺序，
真实图像顺序仍为原protocol，正式计时不验证数组或注入Queue探针。

```bash
.venv/bin/python docs/experiments/dataloader-bounded-inflight/scripts/diagnose.py --prepared runs/udacity_real_prepared_01 --baseline runs/dataloader_bounded_01/baseline_dataloaders.py --output runs/dataloader_bounded_01/diagnostics-01
.venv/bin/python docs/experiments/dataloader-bounded-inflight/scripts/check_real.py --prepared runs/udacity_real_prepared_01 --baseline runs/dataloader_bounded_01/baseline_dataloaders.py --output docs/experiments/dataloader-bounded-inflight/results/real-correctness-01
.venv/bin/python docs/experiments/dataloader-bounded-inflight/scripts/analyze.py --raw runs/dataloader_bounded_01 --report docs/experiments/dataloader-bounded-inflight
.venv/bin/python docs/experiments/fused-sgd-cuda/scripts/archive_results.py --destination docs/experiments/dataloader-bounded-inflight/results/raw-01 runs/dataloader_bounded_01/benchmark-01 runs/dataloader_bounded_01/diagnostics-01
.venv/bin/python -m compileall -q hytorch/dataloaders.py tests/test_dataloader_bounded.py docs/experiments/dataloader-bounded-inflight/scripts
```

绘图使用Agg及`MPLCONFIGDIR=/tmp/hytorch-bounded-inflight-mpl`。
环境记录包括OS/CPU、Python/NumPy/Pillow/pytest/matplotlib和完整pip list。
初次环境采集误用系统python3，查询NumPy元数据失败；保留`system-attempt-01.txt`，
重新以`.venv/bin/python`采集为`system.txt`，没有安装或静默替代依赖。

不做任何文件哈希或Checksum/Fingerprint验证。Git ID只用于版本追溯。
最后正常git commit、非force push到`experiment/dataloader-bounded-inflight`，
核对远程提交后停止，不合并main。
