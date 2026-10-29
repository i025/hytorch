# Block32 + Tree：语义、编译与资源分析

基线为阶段 3A 提交 `3bdd482f1c8e1cece33ff90f8f895feb80151d62`。本轮只编译一个独立 Kernel 版本，没有修改生产 `kernels.cu`、CUDA ABI 或默认 dispatch。

## 1. 原实现与计时方式

已阅读 [生产 Kernel](../../../hytorch/backends/cuda/kernels.cu)、[runtime](../../../hytorch/backends/cuda/runtime.py)、[原 benchmark](../../../benchmarks/bench_cuda_gemm.py)、[3A 报告](../gemm-fp32-accuracy/README.md)、[源码分析](../gemm-fp32-accuracy/source-analysis.md) 和 [CPU 语义脚本](../gemm-fp32-accuracy/scripts/accuracy.py)。

Basic 按 K 递增使用一个 `float v`；Blocked 把循环分段，但不重置 `v`；Shared 用 shared-memory tile，仍保留同一 `v` 跨越全部 K。已有 sm_80 二进制中的 FFMA 证据和编译参数见 3A 原始 PTX/SASS，本轮只读复用。输入 A 为 row-major `(M,K)`，B 为 row-major `(K,N)`，输出为 row-major FP32 `(M,N)`，通过当前 CUDA stream 提交。

原 benchmark 预分配 GPU 缓冲区，以同一 stream 上的 CUDA Event 计时，在 end event 上同步；CPU/GPU 拷贝单独计时。本轮沿用 event 方法，5 次预热、10 次测量，使用已准备好的 native ABI 参数及 CuPy/PyTorch `out=`，不在计时区间分配输入/输出或传输数据。未改变生产 benchmark。

## 2. 独立算法及确定顺序

新实现为 [kernel.cu](../../../experiments/gemm_block32_tree/kernel.cu)，仍使用 32×32 输出 Tile、32×32 线程和两个 FP32 shared tile，以减少访存布局变化对比较的干扰。真正改变的是累加结构：每个连续 32 项块从 `partial=0` 开始，用 `__fmaf_rn` 计算；块末做在线二叉进位合并，节点使用 `__fadd_rn`。这些 intrinsics 显式要求 nearest-even 舍入；`__fadd_rn` 不合并进 FMA，见 [CUDA 12.2.2 单精度 intrinsics](https://docs.nvidia.com/cuda/archive/12.2.2/cuda-math-api/group__CUDA__MATH__INTRINSIC__SINGLE.html)。

设已经处理的块数为 b，二进制位为 1 的层保存对应的已完成子树。新 partial 作为右子树，与已有同层左子树相加，向上一层进位。结束时按低层到高层合并：较低层代表较新的右侧尾部，因此必须用 `levels[level] + total`。不能把剩余子树从左向右相加。

例如七个部分和的结果是：

```text
((p0+p1)+(p2+p3)) + ((p4+p5)+p6)
```

这与 3A 的相邻两项归约、奇数尾项直接带到下一层完全一致。CPU 测试还包含一个反例：错误地左折叠剩余子树会得到 1，正确的归约树得到 0。没有为了接近 NumPy 而改变归约顺序。

原型明确支持 `1 <= K <= 4096`，K 不整除 32 时仅计算实际存在的项，不做尾部无意义 FMA；M/N 的 Tile 边界通过加载零和条件写出处理，所有线程都参加屏障。K=4096 有 128 个部分和，在线状态最多 8 个层槽位，不长期保存 128 项。每个输出总共做 K 次 FMA 和 `ceil(K/32)-1` 次树加法；K=4096 是 4096 次 FMA + 127 次加法。

这是有明确 K 上界的实验原型，不是可替换生产所有形状的通用实现。固定 8 个槽位覆盖实验范围；逻辑上同时存活的子树数量至多为块数二进制表示的位数。

## 3. 编译事实与资源

[编译记录](environment/build.json) 使用 CUDA 12.2 nvcc、`-O3 -std=c++14 -lineinfo`、sm_80/PTX compute_80 和 `--ptxas-options=-v,-warn-spills`。未使用 fast-math。`__launch_bounds__(1024)` 约束最大线程数。二进制保存在服务器 `runs/gemm_block32_tree_build_01/libblock32_tree.so`，没有提交 Git。

| 指标 | 原 Shared32 | 新 Block32 + Tree | 来源 |
|---|---:|---:|---|
| 寄存器/线程 | 32 | 29 | CUDA function attributes、二进制资源、NSYS launch |
| Shared Memory/块 | 8192 B | 8192 B | 同上 |
| Local Memory/线程 | 0 B | 0 B | 二进制与 function attributes |
| 新 Kernel Spill stores / loads | — | 0 / 0 B | ptxas 原始日志 |
| 理论驻留块/SM | 2 | 2 | `cudaOccupancyMaxActiveBlocksPerMultiprocessor` |
| 理论驻留线程比例 | 100% | 100% | 2×1024 / 2048 |
| 实际 achieved occupancy | 未测得 | 未测得 | NCU 权限 BLOCKED |

PTX 中仍能看到中间形式的 `.local` 声明，但最终 ptxas 资源为 0-byte stack、0-byte spill，sm_80 SASS 中没有 LDL/STL。不能用中间 PTX 声明单独判断最终发生 Spill。也不能把 occupancy calculator 的理论驻留上限称为实际运行 occupancy；该 API 的含义见 [CUDA 编程指南](https://docs.nvidia.com/cuda/archive/12.2.2/cuda-c-programming-guide/index.html#occupancy-calculator)。

## 4. 额外开销的证据与推断边界

[静态指令统计](results/instruction-counts.csv) 来自两个实际执行版本的 SASS：

| 指标 | Shared32 | Block32 + Tree |
|---|---:|---:|
| 静态指令数 | 134 | 331 |
| FFMA | 32 | 32 |
| FADD | 0 | 14 |
| shared-load 指令 | 40 | 64 |
| shared-store 指令 | 2 | 2 |
| BAR 指令 | 2 | 2 |
| BRA 指令 | 3 | 18 |

这些是包含循环与尾部路径的**静态代码条数**，不是动态执行次数，也不能直接换算访存流量。两者源码均在每个 K tile 用两次线程块屏障，树归约是线程内的，没有新增跨线程归约同步。

实测没有新增寄存器压力、Local Memory 或理论 occupancy 降低。新增 FADD、位判断/进位分支、尾块谓词，以及编译后的共享加载与调度变化，都可能贡献开销。K=4096 时，每个输出在 4096 次 FMA 外增加 127 次加法；按 FMA=2 FLOPs 计，额外算术工作量约为 1.55%。仍测得约 39%–52% 的 event 延迟增加，不能仅按 FLOP 数预测耗时。

没有做第二版 Kernel 或消融实验，也没有可用的 NCU stall/bandwidth/achieved-occupancy 计数器，因此不能把全部延迟差定量归因于其中某一项。一次成功的 NSYS capture 保留 12 个 Kernel：Shared32 三次平均 26.929325 ms，新 Kernel 三次平均 40.601263 ms，CuPy/PyTorch 六次 SGEMM 合计平均 7.783165 ms。它验证了实际 GPU Kernel 区间和资源信息，但插桩值不混入正式 benchmark。

## 5. TF32 与参考精度

GPU 运行均设置 `CUPY_TF32=0`、`NVIDIA_TF32_OVERRIDE=0`、Torch matmul/cuDNN `allow_tf32=False`。CuPy FP32 compute type=1，cuBLAS math mode=0。NSYS 对照实际 Kernel 名为 `ampere_sgemm_128x64_nn`。新 Kernel 的显式 FP32 FMA/加法、FP32 存储及编译指令相符。

FP64 参考来自精确转换的 FP32 输入做 GPU FP64 GEMM，另用 FP64 GEMM 计算 `abs(A)@abs(B)` 作为消去条件分子。4,758 个抽查位置使用 CPU `math.fsum` 与 FP64 dot 交叉核查，涵盖全部原种子历史失败位置、历史最差坐标、近零和随机位置。另对唯一新增 NumPy 失败位置单独验证。高精度参考也不是精确实数：两处极近零结果在转回 FP32 后，与 fsum 的舍入结果相差 1 和 7 ULP，绝对差约 1e-13，具体数据单独保留。
