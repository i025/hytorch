# 执行命令、环境与复现

工作目录 `/home/xymo/projects/hytorch`，从 `experiment/dataloader-bounded-inflight`
的 `4530176dfdbd320bbd73c18293fabd391def86e4` 创建本分支。
没有哈希、Checksum、Fingerprint步骤；Git提交ID仅记录源码版本。

```bash
git switch -c experiment/cuda-reduction-foundation
export CUDA_VISIBLE_DEVICES=''
export CUDA_PATH=/usr/local/cuda-12.2
export PATH=/usr/local/cuda-12.2/bin:$PATH
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.
```

以下Python计算命令均加 `taskset -c 0-3 nice -n 10 .venv/bin/python` 前缀，控制4个逻辑CPU和
BLAS线程。CPU Torch训练对照设4线程。使用既有独立`.venv`，没有安装依赖或修改共享环境。
主要命令用 `docs/experiments/a100-stage1/run_logged.py --log UNIQUE.log -- COMMAND...` 包装，
完整stdout/stderr、实际argv、开始/结束、环境和返回码见[logs](logs)。

## 离线编译与CPU检查

```bash
.venv/bin/python scripts/build_cuda.py --arch 80 --resource-usage \
  --output runs/cuda_reduction_01/build-01
.venv/bin/python -m pytest tests/test_reduction_host.py -q -ra \
  --junitxml=docs/experiments/cuda-reduction-foundation/results/reduction-host-01.xml
.venv/bin/python docs/experiments/cuda-reduction-foundation/scripts/cpu_check.py \
  --output runs/cuda_reduction_01/cpu-reference-03
HYTORCH_CUDA_LIBRARY=/home/xymo/projects/hytorch/runs/cuda_reduction_01/build-01/libhytorch_cuda.so \
  .venv/bin/python docs/experiments/cuda-reduction-foundation/scripts/check_host.py \
  --output runs/cuda_reduction_01/check-host-01
```

`--arch 80`明确指定目标，避免build脚本默认自动探测GPU。ABI检查只CDLL装载、解析新符号并调用
纯主机`hy_cuda_abi_version`，没有调用CUDA设备入口。
cpu-reference-01首先保存失败样本和统计，-02新增全部输入/输出NPZ保留，-03将数值失败返回码改为1。
三次算术/阈值相同，均284/288通过、4失败；旧日志没有覆盖。最终脚本有失败返回1，不是成功验收。

```bash
.venv/bin/python -m pytest \
 tests/test_cuda_host.py tests/test_conv2d_gemm_cpu.py tests/test_gemm_blocked_packed.py \
 tests/test_fused_sgd_host.py tests/test_conv_dinput_pruning.py tests/test_conv_dweight_host.py \
 tests/test_conv_dinput_opt_host.py tests/test_udacity_dataset.py tests/test_udacity_experiment.py \
 tests/test_dataloader_indices.py tests/test_relu_backward_host.py tests/test_conv_backward_fusion_host.py \
 tests/test_conv_shape_dispatch.py tests/test_dataloader_lifecycle.py tests/test_cpu_pipeline_plan.py \
 tests/test_dataloader_bounded.py tests/test_reduction_host.py -q -ra \
 --junitxml=docs/experiments/cuda-reduction-foundation/results/regression-01.xml
```

结果196 passed，0 failed，0 skipped。与单独57项、CPU三步312项和288项算术诊断不是同一计数范围，
不能相加冒充一个GPU测试集合。

## 仅生成GPU计划：已执行，无GPU调用

```bash
.venv/bin/python -m pytest tests/test_reduction_gpu.py --collect-only -q
.venv/bin/python docs/experiments/cuda-reduction-foundation/scripts/gpu_micro.py \
 --plan --output runs/cuda_reduction_01/micro-plan-01
.venv/bin/python docs/experiments/cuda-reduction-foundation/scripts/gpu_plan.py \
 --output runs/cuda_reduction_01/gpu-plan-01 --gpu-output runs/cuda_reduction_gpu_future_01 \
 --prepared runs/udacity_real_prepared_01
```

对A/I/S/R/SR/E×DonkeyNet/SpatialNet共12组逐一执行：

```bash
.venv/bin/python docs/experiments/cuda-reduction-foundation/scripts/future_training.py \
 --plan --method S --model donkey --no-fingerprints --prepared runs/udacity_real_prepared_01 \
 --output runs/cuda_reduction_01/training-plans-01/donkey-S
```

12条完整实际命令在[training-plans-01.json](results/training-plans-01.json)。
GPU集合仅collect，103项不是103 passed。准备123个顺序子任务，任何正确性失败停止性能测试。

## 下一轮显式GPU启动命令：本轮未执行

**需要下一轮授权及空闲设备后再运行**，将`GPU-实际UUID`替换为获准设备。

```bash
.venv/bin/python docs/experiments/cuda-reduction-foundation/scripts/run_gpu_plan.py \
 --run-gpu --uuid GPU-实际UUID --plan runs/cuda_reduction_01/gpu-plan-01/plan.json
```

启动入口记录资源快照、每任务开始/结束/完整输出，固定UUID，无自动换卡。详见[验收方案](gpu-acceptance.md)。
已存在输出目录需使用新唯一目录重新生成plan，不覆盖历史。旧ReLU基准入口仅从该未来窗口调用。

## 数据可视化和归档：CPU

```bash
MPLCONFIGDIR=/tmp/hytorch-reduction-matplotlib .venv/bin/python \
 docs/experiments/cuda-reduction-foundation/scripts/plot_cpu.py \
 --input runs/cuda_reduction_01/cpu-reference-03 \
 --output docs/experiments/cuda-reduction-foundation/figures/cpu-01
.venv/bin/python docs/experiments/fused-sgd-cuda/scripts/archive_results.py \
 --destination docs/experiments/cuda-reduction-foundation/results/raw-01 runs/cuda_reduction_01
git diff --check
```

环境记录使用uname/lscpu/nvcc/gcc、Python发行包元数据；未查询GPU驱动或设备。
暂存区源码检查使用 `git -c core.whitespace=cr-at-eol diff --cached --check -- hytorch scripts tests docs/experiments/cuda-reduction-foundation/scripts 'docs/experiments/cuda-reduction-foundation/*.md'`。
原始CSV使用Python标准csv的CRLF换行，原始system-info保留工具输出结尾空行；不为消除全量diff的格式提示而改写原始记录。
已知旧scipy未安装提示原样保留，不新增依赖、不静默忽略失败。
最后只提交本轮源码/测试/脚本/记录，正常push新分支，不合并、不启动未来GPU计划。
