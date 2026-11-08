# HyTorch 最终算子三方评测（2026-11-08）

## 结论与实验边界

本轮仅评测，不修改生产源码、CUDA kernel 或优化算法，不运行真实驾驶训练、不提交代码。所有主性能表格来自本轮重新运行；原库只用于同进程消融，并非拼接历史计时。Native 并不普遍优于 CuPy：**GEMM 的大方阵仍明显慢于 cuBLAS；Im2col、Col2im 和 MaxPool forward 的最终完整接口表现较好；Packed dWeight 的收益来自 Native Im2col 与 cuBLAS 的组合以及减少布局复制。**

共 122 个算子 case、两个接口口径；36 个 GEMM shape（一次真实 DonkeyNet 前后向中的 24 次不同 GEMM + 12 个扩展），14 个 Im2col、14 个 Conv forward、14 个 dWeight、36 个 Col2im、各 4 个 Pool forward/backward。真实 workload 使用一个固定 seed 的 batch=4 合成 step，记录前后向及 SGD；没有重新做全训练时序验收。因此本轮不能量化新增训练加速，也不能用微基准时间相加代替训练 wall time。

## 环境、开关与公平性

- 物理 GPU 1：A100-PCIE-40GB，UUID GPU-0bc09ee6-967a-b53d-bf80-38379585cf08；驱动 580.95.05。开始、接口补测前均确认为 0 MiB、无计算进程；结束回到 0 MiB。GPU 0 VLLM 未触碰，没有终止他人任务。
- NumPy 2.2.6 / OpenBLAS 0.3.29，实际线程查询为 4；CuPy 14.2.0；PyTorch 2.9.1+cu128（独立 CPU 参考/环境加载）；nvcc 12.4。环境全记录 raw/environment.json 和 static_environment.txt。
- TF32：所有进程 NVIDIA_TF32_OVERRIDE=0、PyTorch matmul/cuDNN TF32=False；无 fast-math，保留当前 .so。未锁频。当前 native matmul=shared + 128 CTA dispatch、conv=im2col、col2im=native（POD+metadata）、dweight=packed、pool=native；F1 metadata_cache=True，O1 compact_function=True，tile=16。此设置仅实验上下文，未修改全局默认。
- 同一 case 固定 seed=20261108+case 哈希、完全相同逻辑 FP32 输入；值分布 N(0,0.25²)，Pooling 另有全负/相等值例。数据在计时前驻留各自设备，不计 H2D/D2H，也不将 CPU/GPU 的存储边界混为一谈。
- 每种先预热 4 次，短算子每个样本连续调用 5 次、长算子 1 次；常规 18 次、五路 GEMM array 消融 30 次正式重复，平衡轮换并交替正反顺序。CPU 每 case/mode/scope 预声明 30 秒预算，本次没有触发降重复次数。全部原始样本保留 RAW_TIMINGS.csv。
- **sync_us**：调用前 stream synchronize 不计时，perf_counter_ns 包含 Python 提交、GPU 执行及调用后 synchronize；短算子为 5 次调用共享末尾同步的摊销成本，不能称“每次都同步的单次响应延迟”。NumPy 完整同步调用同样循环摊销。主比值只用相同 sync_us。
- **submit_us**：上述正常提交循环到同步前的 CPU elapsed，可能包括 runtime 内部阻塞，不能称纯 Python 指令时间。
- **event_us**：独立调用循环两 CUDA Event 之间的时间，可能含 CPU 提交造成的 GPU 空洞，不等于纯 kernel。
- **kernel_us**：另一轮 Nsight Systems/CUPTI、每模式 15 次 NVTX 采集，通过 correlationId 将实际 kernel 归属到 CPU launch。CUDA API CPU elapsed、NVTX CPU elapsed、kernel 累计各自列出，**不相加**；这些带 profiler 的数据不替换正式 wall time。
- GEMM array 所有实现预分配输出；Function GEMM 包含输出分配。其他算子均包括必要输出/临时分配、packing、layout copy；输入准备、误差计算在计时外。内存池已预热；ALLOCATIONS.csv 是分配请求量，不是 cudaMalloc 次数或峰值显存。
- NumPy Conv/dWeight 为向量化窗口+NumPy BLAS，生产 CPU 自研 C GEMM 没混入这个基线。单独 Native Im2col 是诊断 ABI 适配器；完整接口例外详见 CORRECTNESS.md。
- GPU 实验在同一张卡集中完成；主批次结束复核时 GPU 已释放，随后为纠正 Function 构造口径，重新检查空闲后补测 dWeight/Pool backward。三种实现始终在各自同进程配对内比较；首批/补测不交叉计算同一算子的比值。此实际流程包含一次短暂释放后补测，非人为宣称全程持有 GPU。

## 主要时间结果

以下为每个 case 的重复均值再作等权算术平均，**不是训练调用次数加权，也不是某个固定 shape 的延迟**。GEMM 平均明显受 4096³ 影响；应结合逐 shape CSV、方阵曲线和优势区域图。每行 CPU/Native 是跨硬件实现比较，不能作为自研算法比 cuBLAS 好的证据。

### 数组接口：resident-data，同步完成 μs

| 算子 | case 数 | NumPy CPU μs | CuPy GPU μs | Native GPU μs | CPU/Native | CuPy/Native |
|---|---|---|---|---|---|---|
| GEMM | 36 | 8176.832 | 281.329 | 454.043 | 18.009× | 0.620× |
| Im2col | 14 | 538.412 | 68.048 | 31.174 | 17.271× | 2.183× |
| Col2im | 36 | 654.388 | 50.899 | 27.276 | 23.992× | 1.866× |
| Conv forward | 14 | 694.762 | 134.061 | 90.301 | 7.694× | 1.485× |
| dWeight（混合） | 14 | 861.686 | 125.821 | 77.339 | 11.142× | 1.627× |
| MaxPool forward | 4 | 259.987 | 108.015 | 34.197 | 7.603× | 3.159× |
| MaxPool backward（混合） | 4 | 111.690 | 125.248 | 109.617 | 1.019× | 1.143× |

### Function 接口：包含正常包装/构图，同步完成 μs

| 算子 | case 数 | NumPy CPU μs | CuPy GPU μs | Native GPU μs | CPU/Native | CuPy/Native |
|---|---|---|---|---|---|---|
| GEMM | 36 | 8442.271 | 330.577 | 540.532 | 15.618× | 0.612× |
| Im2col | 14 | 546.450 | 75.091 | 35.338 | 15.464× | 2.125× |
| Col2im | 36 | 663.298 | 56.186 | 32.294 | 20.539× | 1.740× |
| Conv forward | 14 | 699.389 | 149.732 | 105.686 | 6.618× | 1.417× |
| dWeight（混合） | 14 | 868.186 | 137.161 | 89.031 | 9.751× | 1.541× |
| MaxPool forward | 4 | 267.176 | 113.914 | 38.115 | 7.010× | 2.989× |
| MaxPool backward（混合） | 4 | 116.326 | 131.594 | 114.227 | 1.018× | 1.152× |

完整均值、中位数、样本标准差、最小/最大、重复次数分别在各 *_RESULTS.csv；每种 scope、metric 独立成行。SPEEDUP_SUMMARY.csv 同时给逐对差值的近似 95% t 区间，使用保守系数 2.12；共享机顺序样本不保证独立同分布，区间仅作局部稳定性诊断。

## 正确性

当前测试 **319 passed / 2 CPU-specific skipped**；全部 804 组诊断对独立 FP64 参考的混合容差失败元素数为 0（包括大矩阵完整参考）。测试门限未改变。最大绝对误差和相对 L2 的 worst case 不必来自同一个 case；不能把最大绝对误差单独与 atol 比较而忽略 rtol。

| 算子 | Native MaxAbs | Native RelativeL2 | 混合容差失败元素 |
|---|---|---|---|
| GEMM | 0.000133982 | 2.08534e-06 | 0 |
| Im2col | 0 | 0 | 0 |
| Col2im | 9.80217e-07 | 9.24238e-08 | 0 |
| Conv forward | 8.76938e-06 | 5.3811e-07 | 0 |
| dWeight（混合） | 6.75029e-06 | 3.33045e-07 | 0 |
| MaxPool forward | 0 | 0 | 0 |
| MaxPool backward（混合） | 1.82539e-07 | 4.76533e-08 | 0 |

Native GEMM 最大绝对误差约 1.34e-4、最大相对 L2 约 2.09e-6；相对误差很小但不是逐位相等。全三方、旧库和各接口的细项见 ACCURACY.csv；历史严格失败副本仅作追溯，不被当前诊断容差替代。

## GEMM：自研优化有效，但不能等同于超过 cuBLAS

| M×N×K | NumPy | CuPy | 旧 Shared | 旧 Register dispatch | 最终 dispatch | Shared/最终 | 旧 register/最终 |
|---|---|---|---|---|---|---|---|
| 240×64×432 | 60.24 | 24.24 | 25.47 | 79.68 | 25.74 | 0.99× | 3.10× |
| 72×64×576 | 46.18 | 24.29 | 30.87 | 102.70 | 31.11 | 0.99× | 3.30× |
| 256×256×256 | 141.05 | 23.99 | 19.91 | 38.02 | 20.34 | 0.98× | 1.87× |
| 512×512×512 | 802.76 | 34.99 | 72.30 | 69.73 | 72.82 | 0.99× | 0.96× |
| 1024×1024×1024 | 4950.30 | 138.78 | 472.90 | 261.35 | 261.14 | 1.81× | 1.00× |
| 2048×2048×2048 | 33991.69 | 1217.82 | 4288.41 | 1913.88 | 1895.89 | 2.26× | 1.01× |
| 4096×4096×4096 | 249701.01 | 7920.33 | 31275.88 | 12782.65 | 12672.43 | 2.47× | 1.01× |

上表均为同步完成 μs、同一最终 Python wrapper 装载不同冻结 ABI，30 轮同进程平衡交替。当前源码/库 SHA 见 SOURCE_STATUS.md。实际 kernel、grid/block、registers 从 Nsight Systems 读取，见 DISPATCH_DECISIONS.csv，非仅根据 if 条件猜测。

- 240×64×432、72×64×576 最终恢复 Shared，输出 64×64 Register 分块仅 4、2 个 CTA；原 Shared 对应 60、20 个 CTA。最终完整调用相对旧 register dispatch 分别约 3.10×、3.30×；相对旧 Shared 约 0.99×，没有宣称超过 Shared 算法本身。
- 1024³/2048³/4096³ 仍走 Register，256/1024/4096 个 CTA，相对旧 Shared 约 1.81×/2.26×/2.47×，保留大矩阵收益；但相对 CuPy 只有约 0.53×/0.64×/0.63×。CuPy 实际为 ampere_sgemm 等 cuBLAS kernel。
- 窄矩阵和长 K 的最差例为 24×75×12152，最终自研仅约 0.076× CuPy。这个形状来自真实 **dWeight**，但生产 Packed dWeight 实际用 cuBLAS，本 microbenchmark 是“将该形状分别送给所有 GEMM”的诊断，不能误读成当前训练在这里调用了慢 Native GEMM。
- 256³/512³ 的 register 输出 CTA 太少，最终 dispatch 回退 Shared；128 是保守通用门限，当前数据不能证明它对所有矩形最优。
- GEMM 单算子统一使用连续合成 A/B，只复用真实 M/N/K；训练原始 strides/实际后端另有记录，完整 Conv 包含其权重 packing。真实形状比大型方阵更窄，最终 dispatch 和利用率不同；而整个 Conv 还包括 Im2col、权重 transpose packing 与输出 pack。大方阵更快不能直接推出卷积或训练更快。

## Im2col、Col2im、Conv 与 dWeight 的证据

所有 Im2col 输出统一为 (T=N×OH×OW,D=C×KH×KW)，T 顺序(n,oh,ow)，D 顺序(c,kh,kw)，三方布局与数值一致。CuPy 的 6D 输出必须转置再物化矩阵，已计入完整时间。独立 raw/IM2COL_LAYOUT_SAMPLES.csv 分离 6D、仅 layout、完整 CuPy、完整 Native；这些独立测量不能直接相加为整条路径。

以下是五层 array 口径的 **CUPTI 纯 kernel 累计 μs**，每模式 15 次。与前述无 profiler wall time 分开看：


### Im2col

| 层 | CuPy kernel μs | Native kernel μs | CuPy/Native | kernel 次数 | 名称含 copy 的 kernel 次数 |
|---|---|---|---|---|---|
| conv1 | 28.111 | 18.409 | 1.527× | 2→1 | 1→0 |
| conv2 | 49.952 | 29.107 | 1.716× | 2→1 | 1→0 |
| conv3 | 18.558 | 10.065 | 1.844× | 2→1 | 1→0 |
| conv4 | 10.728 | 5.645 | 1.901× | 2→1 | 1→0 |
| conv5 | 8.853 | 4.570 | 1.937× | 2→1 | 1→0 |

### Col2im

| 层 | CuPy kernel μs | Native kernel μs | CuPy/Native | kernel 次数 | 名称含 copy 的 kernel 次数 |
|---|---|---|---|---|---|
| conv1 | 33.389 | 11.823 | 2.824× | 1→1 | 0→0 |
| conv2 | 53.813 | 17.592 | 3.059× | 1→1 | 0→0 |
| conv3 | 26.340 | 9.395 | 2.804× | 1→1 | 0→0 |
| conv4 | 9.446 | 6.298 | 1.500× | 1→1 | 0→0 |
| conv5 | 16.646 | 8.700 | 1.913× | 1→1 | 0→0 |

### Conv forward

| 层 | CuPy kernel μs | Native kernel μs | CuPy/Native | kernel 次数 | 名称含 copy 的 kernel 次数 |
|---|---|---|---|---|---|
| conv1 | 43.420 | 47.687 | 0.911× | 4→4 | 1→1 |
| conv2 | 69.378 | 85.626 | 0.810× | 4→4 | 1→1 |
| conv3 | 35.991 | 52.373 | 0.687× | 5→4 | 1→1 |
| conv4 | 26.110 | 31.590 | 0.827× | 5→4 | 1→1 |
| conv5 | 24.230 | 35.305 | 0.686× | 5→4 | 1→1 |

### dWeight（混合）

| 层 | CuPy kernel μs | Native kernel μs | CuPy/Native | kernel 次数 | 名称含 copy 的 kernel 次数 |
|---|---|---|---|---|---|
| conv1 | 93.302 | 75.475 | 1.236× | 5→4 | 2→1 |
| conv2 | 140.479 | 105.709 | 1.329× | 5→4 | 2→1 |
| conv3 | 69.701 | 52.610 | 1.325× | 5→4 | 2→1 |
| conv4 | 47.465 | 38.042 | 1.248× | 5→4 | 2→1 |
| conv5 | 36.410 | 28.423 | 1.281× | 4→3 | 2→1 |

Copy 的识别基于实际 kernel 名包含 copy，详细名称、grid、寄存器和次数在 KERNEL_DETAILS.csv，可人工审计，不把所有带宽型 kernel 都归为 copy。dWeight 的五层每次均少一次 copy 和一次分配请求；conv2 申请字节由 13,099,008 降至 6,782,208、请求从 4 降为 3。减少的是临时布局拷贝，不是本轮实施了 forward workspace reuse。

Native **dWeight 是 Native Im2col + CuPy/cuBLAS GEMM**；最终路径并没有用自研 GEMM 计算 dW。代码证据为 runtime.py::packed_dweight 中 gm @ col，不能把该乘法速度归为自研 C++ GEMM。Native Pool backward 也是 CuPy scatter + Native Col2im 的混合路径。

Native Conv forward 的五层 kernel 总时间都比 CuPy 长，然而正常完整调用在除 conv2 外的四层更快。这符合减少/简化 Python 提交路径可抵消某些 kernel 差距的解释，但本轮没有 Python 栈采样来作独立因果拆分；PROFILE_SUMMARY.csv 里的 CPU NVTX 与 CUDA API 数据来自 profiler，只是辅助证据。conv2 完整 array 比值 0.929×，属于必须展示的回退。

### tiny、holes、非连续与规模

- Col2im 36 个完整 array 调用中 Native 都更快，范围 1.675–2.761×。但 **holes 的 kernel 本身慢约 14–18%**：例如 contiguous 的 CuPy/Native 为 7.403/8.747 μs；完整同步为 42.009/23.773 μs。tiny kernel 基本相当，Native 完整调用约快 1.8×。这说明 kernel 更快与 binding 更快是不同结论，不能用一项替代另一项。
- tiny/holes 的 CPU Col2im 是 NumPy 竞争力很强的例子：tiny contiguous 6.448 μs、holes contiguous 9.944 μs，均快于相应 GPU 完整调用；当输入本就在 GPU，额外 D2H/H2D 不在本表中，不能直接据此切到 CPU。
- 单独 Native Im2col 必须把非连续 NCHW 输入 pack 成连续后交给 ABI，CuPy 原 ElementwiseKernel 可以按逻辑 strides 读取。tiny_sliced 的 Native/CuPy优势降至约 1.21×；不能从连续 shape 的结果外推所有 view。
- Col2im 的 training、contiguous、negative 三种真实物理布局全部测量；Native 使用实际 signed strides，没有把 view 当连续数组。结果在 COL2IM_RESULTS.csv 全部保留。
- GEMM 从 256³ 到 4096³，CPU 时间从约 0.141 ms 增至约 250 ms（以逐 shape 表为准），Native 在大型矩阵相对 CPU 的优势放大，但始终不等于优于 cuBLAS。卷积相关曲线同时改变几何/形状/元素数，是覆盖性趋势，不是控制其他变量不变的 scaling law。

## 哪些场景更合适

| 算子 | Native 均值更快 cases | 配对差值区间全为正 | 区间全为负 |
|---|---|---|---|
| GEMM | 16/36 | 11 | 18 |
| Im2col | 14/14 | 14 | 0 |
| Col2im | 36/36 | 36 | 0 |
| Conv forward | 13/14 | 13 | 1 |
| dWeight（混合） | 14/14 | 14 | 0 |
| MaxPool forward | 4/4 | 4 | 0 |
| MaxPool backward（混合） | 4/4 | 3 | 0 |

Im2col、Col2im 和 MaxPool forward 的完整接口在本覆盖内优势明显；Packed dWeight 的“布局优化+cuBLAS”组合也有稳定收益。MaxPool backward 只有中等收益，其中 asymmetric 的区间跨零，不宜宣称所有形状都稳定提升。

CuPy/cuBLAS 更适合大型/长 K GEMM；真实 Conv2 的完整调用也应选择 CuPy 作为比较优势方。少数小 GEMM 的 Native 完整调用略快，部分差异在波动内，且 GPU kernel 时间常不更快，不能宣称通用 GEMM 算法超过 cuBLAS。tiny/holes CPU Col2im 保持竞争力。

## 与此前训练收益的关系

1. 新数据确认 shape-aware dispatch 避免低 CTA 的 register 回退，支持此前设计动机；但真实 dI/dW 多数走 cuBLAS，不能把所有 GEMM microbenchmark 的收益都加进训练。
2. 本轮最终 Col2im kernel 与同步完整调用分开测，能说明“kernel 收益可能被 CPU 路径抵消”的机制；此前最初版本训练退化来自历史报告，本轮未重跑旧 binding + 全训练，不将历史数值与本轮数据拼接。
3. dWeight 少一次 copy/分配是本轮直接测得，能支持其减少数据搬运的解释；没有重新训练，所以不能给出本轮端到端提升百分比。
4. O1/F1 本轮一直显式开启但没有再做独立消融，不能把本轮 Native/CuPy差值完全归因于 O1/F1。

## 局限与交付索引

共享服务器未锁频，GPU时钟/CPU调度会变化；无 NCU occupancy/stall 计数器，不能断言精确占用率或 stall 原因。无 transfer-inclusive 或真实训练实验。正式 sync 时间的 short-op 末尾同步是摊销口径；不同 metric 不混算。只比较 FP32；高阶、非默认 stream、FP64 fallback 等生产语义由现有 GPU回归覆盖。没有增加本轮新的 sanitizer；本轮未修改 kernel。

七组图见 figures/01...07，每组 PNG/SVG；数字、图例、误差条、硬件与口径均标注。PPT_SUMMARY.md 给五页安排。完整复现命令在 REPRODUCE.md；环境/源码状态在 SOURCE_STATUS.md；原始样本、kernel详情、分配请求和所有回退 case 均保留。traces 的原始 Nsight 二进制与冻结动态库只记录服务器路径/SHA，不进入压缩包。报告生成后对源码和库逐一复核，未发现改变；不继续进行下一轮优化。
