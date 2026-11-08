# HyTorch 算子优化前源码与性能采集

本轮只做诊断，没有修改任何生产 CUDA/自动求导/优化器算法，没有启用 TF32/fast-math，没有改变 dtype、测试门限或提交代码。
实验日期为 2026-11-06（Asia/Shanghai）；原始机器时间戳保留 UTC。本报告优先读取既有实验，新增 GPU 工作仅为一个短采集窗口。

## 1. 最重要结论

1. **实际运行不是全程 FP32。** 输入、权重和 native 前向是 FP32；`MeanSquareError.backward` 里的 Python 浮点标量被 `core.as_array` 转为 CuPy FP64 标量，第一次混合乘法把梯度提升为 FP64。`Tanh.backward` 的 `1-y*y` 还存在 int64 标量引起的提升。之后 dInput、dWeight、dBias、全连接梯度和 SGD 中间量均为 FP64；SGD 写回 FP32 参数。时间线出现真实 FP64 cuBLAS kernel 和每步 16 次 FP32→FP64 转换。原训练配置 `dtype="float32"` 是声明，不是运行期验证。以前的四组训练 wall 数据仍保留，但**不能继续描述为完全相同精度路径的公平 FP32 框架对比**。
2. **GPU 累计执行时间最大的反向路径是卷积 dWeight**：496.125 µs/step，占全部 kernel 时间 26.55%；其中 im2col 98.247、FP32→FP64 转换 191.134、同 dtype/layout copy 112.023、BLAS/其辅助计算 94.721 µs。只优化 GEMM 主计算无法消除前面的转换和 packing。
3. **dInput 次之**：344.038 µs/step，其中 col2im 215.211 µs，占 dInput 的约 62.55%。conv2 col2im 单次 87.803 µs；它处理 FP64 非连续列张量，逐像素遍历卷积核并做边界/整除检查。
4. **64×64 优化实际只覆盖每步 2 次 GEMM（conv4、conv5）**。覆盖 18,579,456 FLOPs，占 native 前向 GEMM 的 8.641%，占实测前后向所有 GEMM 的 2.880%。没有覆盖卷积/全连接 backward。两个小矩阵只发出 4 和 2 个 block，而不是大方阵工作负载。
5. **还有明显的提交/调度间隙**。每步 161 个 kernel，其中平均 126 个 kernel 的单次耗时 <10 µs（10 步总计 1,260/1,610）。较轻插桩 trace 中，kernel 累计 1.868 ms/step，而 CPU step wall 5.915 ms；kernel+copy 时间区间的并集占 step wall 32.24%。这是已记录 CUDA activity 的时间覆盖率，不是硬件 occupancy 或 nvidia-smi 利用率。不能将其余 67.76% 全部称为 Python 时间。

## 2. 版本与源码状态

- 分支 `main`，HEAD `edb6fbe3bfaa9eed461addd29e6943b22e28216e`，working tree **不干净**，保留用户此前的工作。
- 相对该提交已有 tracked 修改：README 状态说明、`kernels.cu` 的通用寄存器分块和 dispatch（46 行增加、1 行删除）、`test_cuda_device.py` 的 8 个参数化测试（20 行增加）。
- 已有 untracked：`bench_cuda_revisions.py`、A100 复现报告、Udacity 准备记录。本轮仅在被忽略的 `runs/` 下新增诊断和交付文件。
- 原始状态见 `raw/git_status.txt`、`raw/git_head.txt`、`raw/working_tree.patch`。`source_manifest.json` 对照当前生产文件的 SHA-256，最终校验见 `raw/final_integrity.json`。
- CUDA 编译：nvcc 12.4.99，`-O3 -std=c++14 -lineinfo --shared --cudart shared -Xcompiler -fPIC`，sm_80 + compute_80 PTX；ABI=1。本轮未重编译。
- GPU：物理 1，A100-PCIE-40GB，UUID `GPU-0bc09ee6-967a-b53d-bf80-38379585cf08`，PCI `0000:E1:00.0`，108 SM。`CUDA_VISIBLE_DEVICES=1` 后逻辑设备 0。GPU 0 的 VLLM 服务未停止或改动。

## 3. 完整实际调用链与后端边界

### 入口、CPU 与同步

`benchmarks/bench_cuda_training.py:124 main` → 创建 `SyntheticDataset`（:35，CPU NumPy）→ `initialize`（:76，CPU 初始化与规范 FP32 权重）→ `model.to_gpu()` → `DataLoader`（`hytorch/dataloaders.py:14 _SyncDataLoader`，CPU NumPy 拼 batch）→ `cp.asarray`（每步输入/标签 H2D）→ `F.mean_square_error(model(x), t)` → `model.cleargrads(); loss.backward()` → `SGD.update()`。

`hytorch/profiling.py:94 StepProfiler` 在 begin/end 同步，phases 模式每个阶段还同步；throughput 模式不做内部阶段同步。
训练入口每步 `scalar(loss)`（:201）执行 `cuda.as_numpy`，造成 D2H 与等待；每步 JSONL 写入在 `StepProfiler.end`（:133），不包含在其 step timer 内，但包含在 epoch wall 中。
结尾 snapshot 把权重搬回 CPU，不在 epoch wall 内。数据/hash/初始化均有 NumPy 参与；实测 CUDA 算术没有退回 NumPy。

### Conv 前向

`donkeycar_hy/model.py:56 DonkeyNet.forward` → `hytorch/layers.py:135 Conv2d` → `hytorch/functions_conv.py:56 Conv2d.forward`
→ `runtime.selected(:184)` → `runtime.conv2d(:144)`：

1. `_arrays(:87)` 校验 CuPy FP32/当前设备并使输入连续；为 y/col 创建逻辑数组。
2. `_call(:75)` 传入 ctypes 指针和当前 stream → `kernels.cu:125 hy_im2col` → `im2col_kernel(:117)`，生成 `(N*OH*OW, IC*KH*KW)`。
3. `runtime.gemm(:105)` 接收 `col` 与 `w.reshape(OC,-1).T`；后者非连续，`cp.ascontiguousarray` 每次 packing。
4. `_call(hy_gemm)` → `kernels.cu:93 hy_gemm` 按 variant/tile dispatch；满足 M/N/K≥64 才走 `gemm_register(:53)`。
5. `_call(hy_conv_pack)` → `conv_pack` 将矩阵输出变回 NCHW 并加 bias。

默认配置全是 CuPy。`backend(matmul,conv,pool,tile)` 使用 ContextVar 临时选择；CPU 或非 FP32 输入使 selected 回退，并记计数。
`conv=cupy` 使用 `conv2d_cpu_tensordot_array(:405)`，**函数名有 cpu 但 GPU 输入实际使用 CuPy**；路径为 CuPy im2col→tensordot→bias→rollaxis。

### Conv 反向：没有 native backward

`hytorch/core.py:126 Variable.backward` 的 Python 图遍历 → `functions_conv.py:70 Conv2d.backward`，按下面顺序执行：

- **dInput**：`deconv2d(:135)` → `Deconv2d.forward(:95)` → `cp.tensordot(Weight, gy, (0,1))` → `cp.rollaxis(gcol,3)` → `col2im_array(:442)` → `_col2im_gpu(:507)` → CuPy `ElementwiseKernel('col2im')`。无 native hy_gemm 调用。
- **dWeight**：`Conv2DGradW.forward(:147)` → `im2col_array(:376)` → `_im2col_gpu(:468)` → CuPy `ElementwiseKernel('im2col')` → `cp.tensordot(gy,col,((0,2,3),(0,4,5)))` → cuBLAS。转换/packing 由 CuPy 内部触发。
- **dBias**：`gy.sum(axis=(0,2,3))` → `hytorch/functions.py:194 Sum.forward` → CuPy reduction。

没有 requires_grad 剪枝；dInput 和 dWeight 都计算。conv1 的 leaf 输入梯度不用于 SGD，但仍被计算；不能在不设计通用梯度需求机制的情况下为第一层硬编码跳过。

### FC、activation、pooling、update

- 实际 `layers.Linear.forward(:112)` 调用 `F.linear_simple(:317)`，不是 `F.Linear`：`MatMul.forward(:267)` → native GEMM，然后 Add。`MatMul.backward(:275)` 显式用 `MatMul(native=False)`，故走 CuPy ndarray.dot/BLAS。
- ReLU/Tanh/loss 为 CuPy elementwise/reduction；反向 dtype 证据见 `raw/workload/operators.json` 的 id 83–91，首次升型为 id 86。
- native 仅有池化 forward：`Pooling.forward(:184)` → `runtime.max_pool2d(:172)` → `hy_pool`。`Pooling2DGrad.forward(:214)` 是 CuPy zeros/arange/scatter→col2im，没有 native backward。本次 DonkeyNet **0 次 pooling**，因此不虚构池化 profile。
- `optimizers.py:18 Optimizer.update` 用 Python 循环逐参数调用 `SGD.update_one(:91)`；16 个参数每次各发 multiply 和 in-place subtract，共 32 kernel/step。FP64 gradient 中间量最终写回 FP32 参数。
- ctypes 绑定、ABI 类型和 current-stream 都在 `runtime.py:48 load_library/:75 _call`，CUDA C ABI 不执行内存分配、H2D/D2H 或主动同步。

## 4. 实测 workload 与 GEMM 覆盖

一个完整 batch=4 training step 的所有包装调用、完整 shape、dtype、字节 strides、连续性、输入输出、参数、phase、调用次数、分配和 stream 见 `WORKLOAD.csv`；原始逐调用记录及内存 hook 为 `raw/workload/`。
记录包含嵌套包装层，不能把 wrapper 与它的 native.gemm 子调用重复计入 FLOPs。`raw/gemm_workload.csv` 已去掉这种重复。
raw summary 的 dispatch_counts 包含预热/一致性验证，不是单步计数；本文单步次数来自记录区间。所有实际调用使用 stream pointer 0（默认流）；首步输入为 `(4,3,66,200)` FP32 NCHW，标签 `(4,1)` FP32。

| 层 | 前向 M,N,K | kernel | dInput 逻辑 M,N,K | dWeight 逻辑 M,N,K |
|---|---|---|---|---|
| conv1 | 12152,24,75 | shared16 | 75,12152,24 | 24,75,12152 |
| conv2 | 2632,36,600 | shared16 | 600,2632,36 | 36,600,2632 |
| conv3 | 440,48,900 | shared16 | 900,440,48 | 48,900,440 |
| conv4 | 240,64,432 | register16 | 432,240,64 | 64,432,240 |
| conv5 | 72,64,576 | register16 | 576,72,64 | 64,576,72 |
| fc1 | 4,100,1152 | shared16 | 4,1152,100 | 1152,100,4 |
| fc2 | 4,50,100 | shared16 | 4,100,50 | 100,50,4 |
| fc3 | 4,1,50 | shared16 | 4,50,1 | 50,1,4 |

每种 GEMM 每步调用 1 次。前向总计 215,007,056 FLOPs，前后向共 645,021,168 FLOPs（以 2MNK 计，不计 packing/归约/激活）。
CuPy tensordot 的 M/N/K 是按真实 axes/shape 计算的逻辑矩阵尺寸，**不是声称已截获 cuBLAS transa/transb**；输入转置可从 CSV 的 axes/strides 复核。底层 cuBLAS API 参数未单独捕获。

## 5. 时间分解与测量口径

使用三份短 trace：`native_trace` 和 `cupy_trace` 有逐函数 NVTX；`native_plain` 只保留 phase/step NVTX。
逐函数诊断明显增加 Python 开销，因此不使用其 15.190 ms/step wall 作为训练性能结论。
`native_plain` 的 1,610 个 kernel 与详细 trace 的顺序、phase、名称、grid/block、register/shared-memory 签名 **1,610/1,610 全匹配**，据此把 shape 与 dInput/dWeight 标签映射到较低扰动的真实 kernel duration。该映射是可验证派生归因，不是 plain trace 自带的细粒度 NVTX。

以下 GPU 时间均为 `native_plain` 的 10 步 CUPTI activity duration 累加后除以 10；百分比分母为全部 kernel duration 1,868.328 µs/step。CPU phase 为异步调用范围的 elapsed wall，可能与 GPU 重叠。

| 阶段 | kernel 次数/step | GPU µs/step | GPU kernel 占比 | CPU phase elapsed µs/step |
|---|---:|---:|---:|---:|
| Forward | 38 | 689.541 | 36.91% | 1458.568 |
| Backward | 91 | 1062.229 | 56.85% | 3524.438 |
| SGD update | 32 | 116.558 | 6.24% | 430.184 |

CPU step wall=5,914.868 µs，GPU kernel 累计=1,868.328 µs，GPU memcpy=38.919 µs。
三者**不能相加**。详细算子归因的分组如下；与上表是同一批 kernel 的不同汇总层级，也不能相加。

| 路径 | kernel 次数/step | GPU µs/step | 占全部 kernel |
|---|---:|---:|---:|
| SGD_update | 32 | 116.558 | 6.24% |
| activation_backward | 18 | 69.668 | 3.73% |
| conv_dBias | 5 | 60.254 | 3.23% |
| conv_dInput | 20 | 344.038 | 18.41% |
| conv_dWeight | 27 | 496.125 | 26.55% |
| conv_forward | 20 | 554.212 | 29.66% |
| linear_backward | 12 | 65.338 | 3.50% |
| linear_bias_backward | 3 | 10.151 | 0.54% |
| other_backward | 6 | 16.656 | 0.89% |
| other_forward | 18 | 135.329 | 7.24% |

### 每层卷积分解

| 层 | forward µs | dInput µs | dWeight µs | dBias µs |
|---|---:|---:|---:|---:|
| conv1 | 70.477 | 87.239 | 129.716 | 28.909 |
| conv2 | 133.630 | 128.264 | 220.171 | 13.104 |
| conv3 | 79.223 | 64.640 | 76.100 | 6.410 |
| conv4 | 119.815 | 33.978 | 40.842 | 6.013 |
| conv5 | 151.067 | 29.917 | 29.296 | 5.818 |

### 反向内部拆分

| 路径 | 类别 | GPU µs/step |
|---|---|---:|
| conv_dInput | col2im | 215.211 |
| conv_dInput | CuPy_BLAS_compute | 85.754 |
| conv_dInput | copy_layout_or_dtype | 23.843 |
| conv_dInput | cast_FP32_to_FP64 | 19.229 |
| conv_dWeight | cast_FP32_to_FP64 | 191.134 |
| conv_dWeight | im2col | 98.247 |
| conv_dWeight | copy_layout_or_dtype | 112.023 |
| conv_dWeight | CuPy_BLAS_compute | 94.721 |

同 dtype `cupy_copy` 发生在 tensordot 的 layout/packing 路径；未声称每一次 copy 都可安全删除。
FC、loss、activation 和 SGD 的全部 kernel、调用数、单次均值/中位数、总时间与 shape 见 `PROFILE.csv` 和 `raw/native_plain_kernel_summary.csv`。

### CUDA API、同步与传输

- CPU CUDA API 累计 898.126 µs/step，其中 launch API 776.173 µs/step，161 次 kernel launch/step；这些 API elapsed 可能与 kernel 重叠。
- `cudaStreamSynchronize` 21 次/10 步（包含窗口末尾），累计仅 8.866 µs/step；此轻量 runner 每步末尾同步，原 StepProfiler 还有 begin 同步，不能混淆次数。
- 每步 H2D 2 次、633,616 bytes、36.410 µs GPU copy；D2H 1 次、4 bytes（loss）、2.509 µs GPU copy。
- 78.26% 的 kernel <10 µs；launch API 起点到 kernel 起点中位间隔为 7.789 µs。该间隔含 API/排队，不可全部叫“launch overhead”。
- CPU sampling/context-switch sampling 未启用，没有 Python 调用栈采样证据；不将 CPU-GPU 空隙全部归因于 Python。

### 日志与同步的独立对照（同 GPU 窗口，未启用 Nsight）

固定 seed=42，64 样本、batch=4、3 epochs/48 steps、10-step warmup，5 次重复，交替顺序，初始权重每次恢复。
三种模式最终 loss 和所有参数 SHA-256 一致。均保留 CPU DataLoader 和 pageable H2D。所有 wall timer 都包含 loader。

| 模式 | 平均 wall ms/step | 5 次样本标准差 |
|---|---:|---:|
| logged | 4.9127 | 0.1083 |
| no_json | 4.7660 | 0.0864 |
| boundary_only | 4.7023 | 0.0659 |

logged 为 StepProfiler throughput/JSONL + 每步 loss；no_json 去掉 JSON/阶段 context，保留每步同步/标量读；boundary_only 再去掉每步显式同步和标量读，最终统一同步。
GPU event 从循环开头到结尾的 span 也已保存，但包含 CPU 提交空隙，**不是纯 kernel 时间**。此 runner 不启用 ResourceMonitor，且 loader 在 StepProfiler 子区间外，与原入口并非逐行等价；仅用于受控观察测量开销，不替换已有训练基线。

## 6. 内存与热点 kernel

预热后完整一 step：147 次 CuPy pool malloc request，申请字节累计 124,776,518（约 119 MiB），0 次底层 device alloc hook；这是累计请求，不是同时占用或显存峰值。warmup 后 trace 无 cudaMalloc 记录。pool 重用内存块，不等于逻辑 workspace 被跨算子缓存复用。

conv2 dWeight 的 im2col 工作区 `(4,24,5,5,14,47)` FP32=6,316,800 bytes，随后混合 dtype tensordot 触发转换/packing。
conv2 dInput 先生成 `(24,5,5,4,14,47)` FP64=12,633,600 bytes，再 rollaxis 成非连续 `(4,24,5,5,14,47)`；实测 strides 为 `(5264,526400,105280,21056,376,8)` bytes。col2im 输出 FP64 `(4,24,31,98)`。
原生前向和 CuPy 反向各自重新生成 im2col，没有共享同一布局的列工作区。权重 `.T` packing 也每步执行。

以下 grid/block、register 和 shared-memory 来自 CUPTI launch records，不是估算 occupancy：

| 层 | kernel | 单次均值 µs | grid xyz | block xyz | registers/thread | shared bytes/block |
|---|---|---:|---|---|---:|---:|
| conv5 | void gemm_register<(int)16>(const float *, const float *, float *, int, int, int) | 136.350 | 1,2,1 | 16,16,1 | 64 | 8256 |
| conv4 | void gemm_register<(int)16>(const float *, const float *, float *, int, int, int) | 103.261 | 1,4,1 | 16,16,1 | 64 | 8256 |
| conv2 | cupy_copy__float32_float64 | 94.020 | 12338,1,1 | 128,1,1 | 16 | 0 |
| conv2 | col2im | 87.802 | 2279,1,1 | 128,1,1 | 33 | 0 |
| conv2 | void gemm_shared<(int)16>(const float *, const float *, float *, int, int, int) | 76.154 | 3,165,1 | 16,16,1 | 32 | 2048 |
| fc1 | void gemm_shared<(int)16>(const float *, const float *, float *, int, int, int) | 67.063 | 7,1,1 | 16,16,1 | 32 | 2048 |
| conv3 | void gemm_shared<(int)16>(const float *, const float *, float *, int, int, int) | 55.127 | 3,28,1 | 16,16,1 | 32 | 2048 |
| conv1 | col2im | 55.120 | 1238,1,1 | 128,1,1 | 33 | 0 |
| conv1 | cupy_copy__float32_float64 | 51.687 | 7121,1,1 | 128,1,1 | 16 | 0 |
| conv2 | im2col_kernel(const float *, float *, int, int, int, int, int, int, int, int, int, int, int, int) | 45.783 | 6169,1,1 | 256,1,1 | 23 | 0 |
| conv2 | im2col | 44.202 | 12338,1,1 | 128,1,1 | 32 | 0 |
| conv3 | col2im | 42.461 | 741,1,1 | 128,1,1 | 33 | 0 |
| conv2 | cupy_copy__float64_float64 | 39.702 | 12338,1,1 | 128,1,1 | 16 | 0 |
| conv2 | sm80_xmma_gemm_f64f64_f64f64_f64_nt_n_tilesize32x64x16_stage4_warpsize2x2x1_tensor8x8x4_execute_kernel__5x_cublas | 31.767 | 19,42,1 | 128,1,1 | 94 | 49152 |
| conv2 | void cutlass::Kernel2<cutlass_80_tensorop_d884gemm_32x64_16x4_nt_align1>(T1::Params) | 29.434 | 8,3,17 | 128,1,1 | 80 | 49152 |

静态访存依据：
- `_im2col_gpu` 以最后一维 out_x 连续展开，滑窗导致输入像素重复读取，并写出完整 6D 列矩阵。
- `_col2im_gpu` 每线程一个输出像素，循环 KH×KW，做 padding/stride 边界与余数检查，读取上述非连续 gcol。源码可确认 gather 与整数索引工作；未测硬件事务，不能直接断言带宽利用率或缓存命中率。
- `gemm_register` 使用 4×4 寄存器微块、共享内存加载和双同步；conv4/conv5 分别只有 4/2 block，对 108 SM 的 A100 可用网格并行度很小。这是结构性证据，不是实测 occupancy。
- CuPy 的同 dtype copy 和 FP32→FP64 copy 有真实 kernel 证据，但闭源 cuBLAS 内部路径无法仅凭名字还原完整实现。

本轮未运行 Nsight Compute。驱动报告 `RmProfilingAdminOnly: 0`，不是因权限不足而放弃；已有 Nsight Systems 能提供本轮所需 launch 配置，为避免额外 GPU replay 优先使用它。因此 bandwidth/occupancy/warp-stall/内存事务指标没有测量，也没有填写猜测值。

## 7. 优化候选与证据边界

最多三个候选详见 `OPTIMIZATION_CANDIDATES.md`：先明确标量 dtype 合同，随后考虑 dWeight 的工作区/packing 路径和 dInput col2im。
小矩阵 native GEMM 的 grid 利用问题同样明确，但本轮未与原 native 对这些真实 shape 做配对微基准，不能宣称寄存器版本在这些 shape 上比原版更快或更慢。

已测：运行 dtype、shape/strides、每步调用、kernel/API/copy 时长、launch 配置、pool 请求、诊断前后逐位一致。
推测：修复 dtype 后可节省多少时间、消除 packing 是否抵消布局转换成本、col2im 重写后的带宽/occupancy、缩小 tile 是否改善端到端。未给未经测量的加速比。

## 8. 交付与停止点

`OPERATOR_SOURCE.md` 含完整关键函数和行号；`source/` 为 32 个真实源码文件快照；`WORKLOAD.csv`、`PROFILE.csv` 含完整 JSON 形状字段和统计口径。
详细可复现命令见 `REPRODUCE.md`，原始 trace 位于相邻 `operator_handoff_traces/`，不放入上传包。
原先 45 项 GPU 测试及 sanitizer 结果被复用；本轮只验证 observation hooks 的 loss/gradients/weights 一致性，不重复整套测试。
信息采集完成后停止，等待 ChatGPT 的优化方案；本轮没有应用任何优化补丁。
