# 完整命令与复现边界

工作目录 `/home/xymo/projects/hytorch`，基线`9909e546d9314e342663d75c4a10d1f46b4b7351`。
没有哈希/Checksum/Fingerprint步骤。Git提交ID仅记录源码版本。

所有CPU实验使用既有`.venv`。以下环境只影响当前命令：

```bash
export CUDA_VISIBLE_DEVICES=''
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=.
# 对每项计算命令加：taskset -c 0-3 nice -n 10
```

## 分支与原始源码

```bash
git switch -c experiment/cpu-training-pipeline
git show 9909e54:hytorch/dataloaders.py > runs/cpu_pipeline_01/baseline_dataloaders.py
git show 9909e54:donkeycar_hy/udacity_dataset.py > runs/cpu_pipeline_01/baseline_udacity_dataset.py
```

基线文件已经保存。重放实验应选择新的唯一输出目录，不能覆盖本轮记录。

## 实验命令

以下省略公共前缀`.venv/bin/python`及上述affinity/nice；实际完整argv和环境
可见对应日志的`.command.json`，逐进程性能子命令另存各grid的`commands.json`。

```bash
docs/experiments/cpu-training-pipeline/scripts/reproduce_lifecycle.py --module runs/cpu_pipeline_01/baseline_dataloaders.py --output docs/experiments/cpu-training-pipeline/results/lifecycle-before.json
docs/experiments/cpu-training-pipeline/scripts/reproduce_lifecycle.py --module hytorch/dataloaders.py --output docs/experiments/cpu-training-pipeline/results/lifecycle-after.json

docs/experiments/cpu-training-pipeline/scripts/bench_loader.py --prepared runs/udacity_real_prepared_01 --baseline runs/cpu_pipeline_01/baseline_dataloaders.py --output runs/cpu_pipeline_01/loader-grid-01

docs/experiments/cpu-training-pipeline/scripts/profile_cpu.py --prepared runs/udacity_real_prepared_01 --output runs/cpu_pipeline_01/profile-01

docs/experiments/cpu-training-pipeline/scripts/bench_loader.py --prepared runs/udacity_real_prepared_01 --baseline runs/cpu_pipeline_01/baseline_dataloaders.py --baseline-dataset runs/cpu_pipeline_01/baseline_udacity_dataset.py --variants repaired stream --narrow --repeats 5 --output runs/cpu_pipeline_01/stream-grid-01

docs/experiments/cpu-training-pipeline/scripts/bench_loader.py --prepared runs/udacity_real_prepared_01 --baseline runs/cpu_pipeline_01/baseline_dataloaders.py --baseline-dataset runs/cpu_pipeline_01/baseline_udacity_dataset.py --variants repaired stream --narrow --repeats 3 --output runs/cpu_pipeline_01/stream-integrated-01

docs/experiments/cpu-training-pipeline/scripts/profile_cpu.py --prepared runs/udacity_real_prepared_01 --open-stream --output runs/cpu_pipeline_01/profile-stream-01

docs/experiments/cpu-training-pipeline/scripts/check_cpu.py --prepared runs/udacity_real_prepared_01 --baseline runs/cpu_pipeline_01/baseline_dataloaders.py --output docs/experiments/cpu-training-pipeline/results/exact-data-03

docs/experiments/cpu-training-pipeline/scripts/check_pending.py --output docs/experiments/cpu-training-pipeline/results/pending-observation.json

docs/experiments/cpu-training-pipeline/scripts/prepare_gpu.py --prepared runs/udacity_real_prepared_01 --output docs/experiments/cpu-training-pipeline/results/future-gpu-02

docs/experiments/cpu-training-pipeline/scripts/future_training.py --method E-prefetch --model spatial --task io --prepared runs/udacity_real_prepared_01 --output docs/experiments/cpu-training-pipeline/results/future-dry-run-02 --no-fingerprints --plan

docs/experiments/cpu-training-pipeline/scripts/analyze_stream.py --raw runs/cpu_pipeline_01 --report docs/experiments/cpu-training-pipeline
docs/experiments/cpu-training-pipeline/scripts/analyze_cpu.py --raw runs/cpu_pipeline_01 --report docs/experiments/cpu-training-pipeline
```

初始grid用同一preprocessing，比较existing/repaired/inplace，10配置×3版本×3重复。
句柄原型实验5重复，集成API再3重复，均只测workers0/2/4及prefetch2，匹配对照不跨批拼接。
随机种子42仅用于运行顺序，全部图像顺序来自原protocol，无在线调优。

当前脚本`stream`直接调用公开`open_stream=True`。第一批30次句柄原型在加入公开接口前
使用等价Dataset子类，内容为`with self.paths[index].open('rb') as stream: value=preprocess_udacity(stream)`
及原标签copy；集成后18次重新测量公开接口。原型与集成结果分开保存。
初始90次时正式Dataset尚未增加可选参数；其默认读取路径至今未变。

## CPU回归

```bash
CUDA_PATH=/usr/local/cuda-12.2 .venv/bin/python -m pytest \
  tests/test_cuda_host.py tests/test_conv2d_gemm_cpu.py tests/test_gemm_blocked_packed.py \
  tests/test_fused_sgd_host.py tests/test_conv_dinput_pruning.py tests/test_conv_dweight_host.py \
  tests/test_conv_dinput_opt_host.py tests/test_udacity_dataset.py tests/test_udacity_experiment.py \
  tests/test_dataloader_indices.py tests/test_relu_backward_host.py \
  tests/test_conv_backward_fusion_host.py tests/test_conv_shape_dispatch.py \
  tests/test_dataloader_lifecycle.py tests/test_cpu_pipeline_plan.py -q -ra \
  --junitxml=docs/experiments/cpu-training-pipeline/results/host-regression-03.xml
```

CPU公共前缀仍需沿用。CUDA_PATH仅供一项既有CPU符号解析测试找到libcudart，
它不调用cudaDeviceReset或任何设备函数。首轮漏设变量的122 passed/1 failed完整保留，
复测123 passed，再加入文件关闭测试后124 passed。没有修改容差、跳过或改写失败测试。

真实数据校验先使用`exact-data-01`，因参考NPZ被重复读取而中断；
一次性载入后`exact-data-02`30配置通过，最终`exact-data-03`40配置通过。
未来入口在`future-gpu-01`/`future-dry-run-01`已先做CPU验证；加入统一open_stream后
另存`-02`，不覆盖初始记录。所有`--run-gpu`任务仅出现在计划JSON中，未执行。

## 日志、归档与静态检查

主要命令使用既有`docs/experiments/a100-stage1/run_logged.py --log UNIQUE.log -- ...`
保存stdout/stderr、实际argv、环境、开始/结束时间和返回值。
初始loader-grid直接将stdout/stderr重定向到`logs/loader-grid-01.log`，90个子命令和
每次开始/结束/耗时/RSS/CPU均在grid原始JSON/CSV中。生命周期两次复现以原始JSON为准。

```bash
.venv/bin/python docs/experiments/fused-sgd-cuda/scripts/archive_results.py --destination docs/experiments/cpu-training-pipeline/results/raw-01 runs/cpu_pipeline_01/loader-grid-01 runs/cpu_pipeline_01/profile-01
.venv/bin/python docs/experiments/fused-sgd-cuda/scripts/archive_results.py --destination docs/experiments/cpu-training-pipeline/results/raw-02 runs/cpu_pipeline_01/stream-grid-01 runs/cpu_pipeline_01/stream-integrated-01 runs/cpu_pipeline_01/profile-stream-01
.venv/bin/python -m compileall -q hytorch/dataloaders.py donkeycar_hy/udacity_dataset.py tests/test_dataloader_lifecycle.py tests/test_cpu_pipeline_plan.py docs/experiments/cpu-training-pipeline/scripts
```

绘图环境`MPLCONFIGDIR=/tmp/hytorch-cpu-pipeline-mpl`，Agg后端。无新增依赖或系统配置。
`results/raw-01`与`raw-02`复制完整可移植数值/日志，二进制pstats只列索引留服务器。
本轮结束执行正常git commit和非force push到新分支，不合并main、不启动GPU计划。
