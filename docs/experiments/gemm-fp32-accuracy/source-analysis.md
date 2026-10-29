# GEMM 累加机制：源码、已编译指令与实测证据

基线：`aad4d0562fd968083a727099e478e7db64d35dd0`，分支 `experiment/a100-gpu-complete`。本轮仅新增独立诊断文件，未修改下面任何生产文件或历史报告。

## 1. 从源码直接确认

| 对象 | K 维顺序及状态 | 精度与含义 |
|---|---|---|
| [Basic](../../../hytorch/backends/cuda/kernels.cu#L12) | `float v=0`，`q=0..k-1`，`v += a*b` | 每输出元素一个逻辑 FP32 累加器 |
| [Blocked](../../../hytorch/backends/cuda/kernels.cu#L24) | `base+=T`，内层 `q=0..T-1`；`v` 跨所有块保留 | 分块访存/循环，不是分块部分和归约 |
| [Shared](../../../hytorch/backends/cuda/kernels.cu#L35) | FP32 shared-memory tile；`v` 在 tile 外初始化，跨 tile 保留；内层 unroll | 同样是按 K 递增的单累加链，T=16/32 不改变这一点 |
| [Python runtime](../../../hytorch/backends/cuda/runtime.py#L85) | 要求 CuPy FP32 输入，转为 contiguous，FP32 输出，传当前 stream | 没有 FP64 转换或更高精度隐藏累加 |
| [原 benchmark](../../../benchmarks/bench_cuda_gemm.py#L93) | `default_rng(42+N)`，先生成 A 后 B，`standard_normal(..., dtype=float32)`；NumPy `matmul` 参考 | `atol=rtol=1e-4` 原样保留 |

Shared 在尾块会补零；本轮三个 N 均整除 16/32，没有尾块问题。这里的“单累加器”指一个输出元素的一条逻辑依赖链，不意味着编译器必须始终使用同一个寄存器编号。

复用了历史 [diagnose_gemm.py](../a100-gpu-window/scripts/diagnose_gemm.py) 的输入和异常位置，以及 [diagnose_accumulation.py](../a100-gpu-window/scripts/diagnose_accumulation.py) 的 `libm.fmaf` 计算语义。旧脚本及其输出没有重新写入。

## 2. 从已有编译产物确认 FMA

本轮只读取已有动态库，没有重新编译。已有 build.json 使用 CUDA 12.2 的 nvcc、`-O3 -std=c++14 -lineinfo`、`compute_80 / sm_80`，没有 `--use_fast_math` 或 `--fmad=false`。其副本为 [native-build.json](evidence/native-build.json)。

只靠 `v += a*b` 源码不能断言编译后的 FMA；本轮额外保存 [PTX](evidence/cuda-ptx.log)、[SASS](evidence/cuda-sass.log) 和 [GEMM 指令摘要](evidence/gemm-instruction-summary.json)。五个 GEMM specialization 均有 `fma.rn.f32` / `FFMA`，没有 `HMMA` 或 `DMMA`。静态 FFMA 条数分别为 Basic 29、Blocked16/32 各 31、Shared16 16、Shared32 32；这是展开后的静态指令数，**不是每个输出实际执行的 K 次计算数量**。

在 FMA 中，输入和累加器是 binary32；乘积与累加数学上合并后只舍入一次到 binary32，不是“先把乘积舍入到 FP32，再单独做一次 FP32 加法”。最终存储为 FP32，没有 FP64 累加。NVIDIA 对这种单次舍入及非结合性的解释见 [CUDA 12.2.2 浮点指南](https://docs.nvidia.com/cuda/archive/12.2.2/floating-point/index.html)；默认允许 FMA 合并见 [nvcc 12.2.2 文档](https://docs.nvidia.com/cuda/archive/12.2.2/cuda-compiler-driver-nvcc/index.html)。

独立 CPU `fmaf` 按 K 递增计算，在全部 1,548 个抽样位置都与五个自研 GPU specialization 数值完全相等。这是额外的数据证据；没有在 CPU 上对整个 N=4096 矩阵逐元素模拟 FMA。

## 3. NumPy FP32 与 CUDA 的顺序差别：已知与未知

环境实测为 NumPy 1.24.4、OpenBLAS 0.3.21、ILP64、`DYNAMIC_ARCH`，选中 `Zen` core，四个 BLAS 线程。NumPy 返回 FP32 数组。本轮没有追踪其 SGEMM 微内核的具体寄存器分组和每次求和顺序，因此**不把 NumPy 等同于本轮的 Pairwise、block32 或任何特定归约树**。

数据事实是 NumPy FP32 与自研顺序 FMA 大量元素不同，而且二者均相对独立 FP64 dot 有误差。结合浮点非结合性，归约分组/顺序不同是符合证据的解释，但不能仅凭输出倒推出完整 OpenBLAS 计算图。NumPy 的实际微内核仍是未追踪项。

CuPy/PyTorch 在 N=2048/4096 全矩阵与自研一致；N=1024 时 CuPy/PyTorch 相互一致，却与自研有 996,036 个元素数值不同。库可能随形状选择不同算法，这是合理推测；本轮未分析 cuBLAS 内部 kernel，不能把其全部形状都描述为顺序 FMA。

## 4. TF32 与实际执行配置

记录见 [GPU 元数据](results/gpu-01/gpu-summary.json) 和 [启动记录](evidence/gpu-monitor-01/launch.json)：

- `CUPY_TF32=0`，`NVIDIA_TF32_OVERRIDE=0`。
- CuPy FP32 compute type 为 1，cuBLAS handle math mode 在计算前后均为 0。
- `torch.backends.cuda.matmul.allow_tf32=False`，`torch.backends.cudnn.allow_tf32=False`。
- 自研 GEMM 的已编译指令为 FP32 FFMA，不是 TF32 Tensor Core 指令。
- CuPy 报告 CUDA runtime 12020；PyTorch wheel 的 build CUDA 版本为 12.1。这是不同含义的版本记录，不把 wheel 标记当成 Toolkit 版本。

这些配置与禁用 TF32 的预期一致。没有对闭源 cuBLAS 做逐指令性能分析；公开配置与实测结果是这里的证据范围。尤其 N=2048/4096 的 GPU 抽样输出完全复现 IEEE FP32 `fmaf`，没有观察到需要 TF32 截断来解释的误差。

## 5. 独立候选实现及未来 CUDA 成本

[accuracy.py](scripts/accuracy.py) 明确控制每次运算的精度：

| 策略 | 实际计算 |
|---|---|
| 顺序 FMA | ctypes 绑定 `fmaf(float,float,float)->float`，K 递增，每步舍入到 FP32 |
| Pairwise | `np.multiply(..., dtype=np.float32)` 先舍入乘积；相邻两项 `np.add(..., dtype=np.float32)`，逐层合并 |
| Block32/64/128 | 每个连续 K 块内从零开始 FMA；块部分和存为 FP32，再做上述平衡树 FP32 加法 |
| FP64 acc → FP32 | FP32 输入精确转为 binary64，逐项 binary64 乘加，最后一次转换 FP32 |
| 独立参考 | 取同一位置的 FP64 `np.dot`；额外用 `math.fsum` 对 binary64 乘积交叉检查 |

本轮输入的 FP32 有限数乘积可由 FP64 精确表示，但 FP64 求和仍有舍入。最后输出 FP32 又引入输出量化误差，因此 E 方案也不等同于精确实数答案。

Pairwise 把长度 K 的串行归约深度变为约 log2(K)，但先舍入乘积，可能丢掉 FMA 能保留的信息。语义测试包含一个 FMA 保留 `-2^-46`、先舍入乘积得到零的反例；另一个 `[1e8,1,-1e8,1]` 点积中真值 2，顺序 FMA 得 1、Pairwise 得 0。因此没有逐点精度单调改善的保证。

未来最实用的首个 CUDA 候选是 **连续 32 项局部 FMA + FP32 平衡树合并部分和**，同时保留纯 Pairwise 作为精度对照。块长 32 的优先级来自此次各规模平均误差，而不是恰好等于 warp 长度。K=4096 时每输出有 128 个部分和；如果全部持有，会增加寄存器或共享内存压力。可以考虑确定顺序的逐级合并，或者将局部点积分配给不同线程，但需要额外存储、加法、shuffle/同步以及布局设计。平衡树合并 128 个部分和需 127 次加法；纯 Pairwise 对 4096 个已舍入乘积需 4095 次加法及对应乘法。具体寄存器占用、并行效率和速度均未实测，本轮没有实现 CUDA 候选或做性能结论。
