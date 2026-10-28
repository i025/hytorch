# A100 阶段 2A：GEMM 实验准备（GPU 测量待执行）

**状态：尚未完成正式实验。** 2026-10-28（Asia/Shanghai）多次检查发现 8 张 A100 都有其他计算进程，未选择或使用 GPU。没有将瞬时 0% 利用率当成空闲，没有沿用阶段 1 的物理 GPU 1。GPU 小规模试运行、1024/2048/4096 正式测量、数值正确性结论、性能排名和六张 GPU 结果图均待执行。

本分支 `experiment/a100-stage2a-gemm` 从阶段 1 提交 `a3ae73a93fcac7148d42dcc294bc361a80389343` 创建。阶段 1 的报告、实验数据、CUDA 源码和训练代码保持不变。未运行 Conv2d、Pooling 性能实验、Nsight 或模型训练。本轮未进行任何哈希校验。

## 已完成工作

- 阅读 README、CUDA 开发与验收指南、阶段 1 报告及 GEMM benchmark、报告器、runtime、CUDA GEMM kernel 源码。
- 保留 CUDA Event 计时、同步、FP32 同输入、预分配输出以及数值误差比较。
- GEMM 默认改为 5 次预热、10 次正式测量；新增 `--warmup`。
- 增加 P95、GFLOPS、每行 atol/rtol、hardware 类型、逐次 H2D/D2H 和长表 `raw-measurements.csv`。
- Basic 只执行一次，CSV Tile=N/A；Blocked/Shared 各执行 16/32。
- 增加 cuBLAS 版本、handle 数学模式前后值、CuPy FP32 compute policy 和相关环境变量记录。GPU 字段尚无实测值。
- 每完成一个矩阵规模保存一次 CSV，后续规模失败时也保留已完成数据；原始失败仍报错，不静默忽略。
- 提供只接收实测 GPU 数据的六图生成器 `scripts/plot_gemm.py`，以及单卡占用检查和运行期时钟/功耗/进程记录器 `scripts/run_monitored.py`。绘图脚本未获得 GPU 数据，因此尚未生成六张图，不使用伪造数据测试绘图。

## 当前环境和限制

Ubuntu 20.04.6；2 × AMD EPYC 7763；8 × A100-PCIE-40GB（40960 MiB/卡）；驱动 535.247.01；nvcc 12.2.140；Python 3.8.10；NumPy 1.24.4；CuPy 12.3.0；Git 2.25.1。NumPy 链接 OpenBLAS ILP64，本轮设置 OMP/OPENBLAS 线程数为 4。CUDA Runtime 12.2 已在阶段 1 验证；本轮 GPU runtime 实测尚待空闲设备。

占用、GPU 时钟、功耗、温度、驱动等原始快照在 `environment/`。两次归档快照均显示每张卡有计算进程。现阶段 `CUDA_VISIBLE_DEVICES=''`，没有物理到逻辑 GPU 的实际实验映射。待选中空闲 UUID 后只暴露该 UUID，进程内部使用逻辑 0。

精度配置准备使用 `CUPY_TF32=0` 和 `NVIDIA_TF32_OVERRIDE=0`。核对 [CuPy 12.3 官方精度选择源码](https://github.com/cupy/cupy/blob/v12.3.0/cupy/_core/_routines_linalg.pyx)：环境变量 CUPY_TF32>0 才选择 TF32 policy；默认 FP32 policy 的 CUDA 11+ 路径选择 CUBLAS_COMPUTE_32F，TF32 policy 则选择 FAST_TF32。最终还需记录实际 handle mode 与 policy 值；这些配置不能冒充 kernel 指令级精度验证。没有运行 Nsight。

## 计时口径审查

- NumPy 是 CPU OpenBLAS 基准，使用 `perf_counter` 墙钟时间；GPU 使用同一当前 stream 上 CUDA Event，预热后同步，每次结束 Event 同步。
- 每次 GPU Event 对函数调用包围计时，包含该 API 提交之间可能存在的设备空隙，不能称为通过 profiler 分离出的纯 kernel 指令时间。
- NumPy/CuPy/CUDA 均计算 FP32 `C=A@B`，无转置语义改变。输入由 `default_rng(seed + size)` 生成，同一规模各实现复用同一输入。
- CuPy 和 CUDA 共用预分配 `dy`，各结果分别复制到独立 host target。Basic launch 固定 256 threads，tile 参数不影响 Basic kernel，因此报 N/A，不额外运行 Basic-32。
- CPU/GPU compute、H2D（两个输入矩阵）和 D2H（一个输出）分别计时。传输使用 pageable host 内存和同步墙钟计时，包含同步/API 开销。
- H2D 每个规模测量一组、各 GPU 实现共享该组；长表标记为 `implementation=shared_inputs`，不伪装成每个实现独立的 H2D 样本。D2H 每个实现独立测量。
- `transfer_plus_compute_ms` 是三个独立均值之和，**不是实测端到端延迟**。
- Std 使用样本标准差（ddof=1）；P95 使用 NumPy 线性分位数；GFLOPS=`2*N**3/(mean_ms*1e6)`。10 个样本的 P95 统计稳定性有限，保留全部原始数值。
- 默认 atol=rtol=1e-4；未为了通过而改变容差。只有 `correct=True` 的配置参与最快实现选择。

## 已实际执行的检查

CPU 格式试运行（不是用户要求的 GPU 试运行）：64、128 方阵，seed=42，每组 5 次预热、10 次重复，共 2 行 NumPy 汇总和 20 次原始测量。记录于 `checks/cpu-format/`，原始 runs 保留于 `/home/xymo/projects/hytorch/runs/gemm_stage2a_cpu_format_01/`。

`checks/record-check.log` 确认 mean/std/median/P95/GFLOPS 与原始时间一致，GPU 状态为 NOT_EXECUTED，未伪造 CuPy 加速比或 GPU 胜者。NumPy 的参考就是自身结果，CPU-only 的 correct=True 不代表 GPU 数值验证。

主机回归执行 `tests/test_cuda_host.py`：**14 passed / 0 failed / 2 skipped**。两项跳过为原有 tiny/donkey 的可选 PyTorch 对照，原因是未安装 torch，见 `checks/host-regression.log` / `.xml`。测试范围不同于阶段 1 的三个测试文件总数，不能直接把 14 与阶段 1 的 21 比较为回退。

## GPU 空闲后的待执行命令

下列为准备好的命令，**尚未执行**。`GPU_UUID` 必须先替换为当时真正空闲且允许使用的 UUID；不要直接沿用任何历史编号。记录器在启动前再次检查 0 MiB、0% 和无计算进程，未满足会退出，保留检查记录。它不提供调度器级独占保障；运行期间每秒保存所选卡进程及硬件快照，出现其他进程时保留受干扰实验，并决定是否用新目录重测。

```bash
cd /home/xymo/projects/hytorch
source docs/experiments/a100-stage2a-gemm/environment/activate.sh

# 先小规模 GPU 试运行，成功后检查结果格式/误差再运行正式实验。
python docs/experiments/a100-stage2a-gemm/scripts/run_monitored.py \
  --gpu GPU_UUID --record docs/experiments/a100-stage2a-gemm/checks/gpu-pilot-01 -- \
  python benchmarks/bench_cuda_gemm.py --device 0 --sizes 128 256 \
  --tiles 16 32 --repeats 10 --warmup 5 --seed 42 --output runs/gemm_stage2a_pilot_01

python docs/experiments/a100-stage2a-gemm/scripts/run_monitored.py \
  --gpu GPU_UUID --record docs/experiments/a100-stage2a-gemm/results/record-01 -- \
  python benchmarks/bench_cuda_gemm.py --device 0 --sizes 1024 2048 4096 \
  --tiles 16 32 --repeats 10 --warmup 5 --seed 42 --output runs/gemm_stage2a_01

python docs/experiments/a100-stage2a-gemm/scripts/plot_gemm.py \
  --input runs/gemm_stage2a_01 --output docs/experiments/a100-stage2a-gemm/figures
```

已有输出必须改新名称，不覆盖原记录。正式结果将有 21 行汇总（3 个规模 × 7 个实现），共 210 次计算测量；传输原始数据另列。全部数值误差和失败配置均须保留，不只保留最快配置。

## 待生成图像与原始数据对应关系

六张图由 `gemm.csv` 的实际字段自动生成：

| 图 | 数据来源 |
| --- | --- |
| gemm-latency.png | 各 GPU 配置 mean_ms，对数 Y 轴 |
| speedup-vs-cupy.png | speedup_cupy，包含 1× 线 |
| gemm-gflops.png | gflops |
| tile-comparison.png | Blocked/Shared 的 mean_ms 与 std_ms |
| transfer-vs-compute.png | h2d_ms、mean_ms、d2h_ms，分项显示 |
| latency-variability.png | repeats_ms 全部独立测量，箱线图与原始点 |

未通过数值检查的配置保留并标记，不作为有效胜者。没有真实 GPU 数据时绘图器直接报错，不创建 GPU 结果图。图表将是数据可视化，不冒充终端截图。

## 文件、提交和下一步

当前所有原始记录根目录：`/home/xymo/projects/hytorch/docs/experiments/a100-stage2a-gemm/`。[完整文件索引](files.md)、[已执行命令](commands.md)、[分析状态与源码机制](results/analysis.md)。阶段 1 数据位于原目录，未修改。

实际修改生产代码仅 `benchmarks/bench_cuda_gemm.py`；未修改 `scripts/report_cuda.py`、CUDA kernel、runtime、Conv2d、Pooling、DonkeyNet 或训练框架。原有计时 helper 的默认 warmup=1 保持不变，GEMM 显式传入新参数，避免改变导入这些 helper 的其他 benchmark 行为。

本次提交保存可执行准备工作和真实 CPU 检查，不宣称阶段 2A 完成。剩余阻塞为没有空闲 GPU；待空闲后补 GPU 试运行、正式测量、数值检查、六图和实测分析，再提交推送最终结果。完成前后均不自动进入阶段 2B。
