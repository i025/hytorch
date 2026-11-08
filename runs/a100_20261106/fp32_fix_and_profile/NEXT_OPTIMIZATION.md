# 下一轮候选 CUDA 算子（按预期端到端影响排序）

下列顺序是基于已测 GPU 工作量与可行性的判断，不是已经得到的训练加速保证。CPU/GPU 异步重叠和主机提交间隙使 kernel 节省不能直接换算成 wall time。

## 1. Native 小矩形 GEMM：通用输出分块与 dispatch

位置：`hytorch/backends/cuda/kernels.cu:33`（gemm_shared）、`:53`（gemm_register）、`:93`（hy_gemm），调用入口 `hytorch/backends/cuda/runtime.py:105`。

事实：修复后训练 conv4/5 register GEMM 每步分别 113.268 / 148.583 μs，合计 261.851 μs，约占全部 GPU kernel 14.95%。实际 block 为 4 / 2，A100 有 108 SM；原 shared 则 60 / 20 blocks。相同随机 FP32 输入微基准 CUPTI：原/现分别 25.248 / 95.253 μs、32.365 / 124.936 μs，当前分别慢 3.77 / 3.86 倍。两 native 输出逐位相同。其他前向 native GEMM 也有开销，卷积 GEMM 总计 436.271 μs/step。

假设与方向：大输出 tile 的复用优势抵不过少量输出 block 的并行度限制；基于通用 M/N/K 工作量选择较小输出 tile、不同线程分块或通用 dispatch 值得设计，不应硬编码 conv4/5。没有测 occupancy、SM 活跃率、带宽，不能将结构推断冒充硬件计数器结论。此次没有改 dispatch。

风险/测试：尾块、非方阵、长 K 累积误差、非默认 stream、输入视图和输出别名。维持严格门限，保留当前 long-K 失败。需要在跨形状训练与配对端到端测试中确认收益。

## 2. dInput col2im

位置：`hytorch/functions_conv.py:95`（Deconv2d.forward）、`:442`（col2im_array）、`:507`（_col2im_gpu）。

事实：dInput 341.710 μs/step；其中 col2im 233.735 μs，占 dInput 68.40%、全部 GPU kernel 13.35%。conv2/1/3 分别 95.066 / 59.565 / 46.371 μs。FP64 已消失，该热点仍然存在。

假设与方向：现实现每个输出像素循环 kernel 窗口、做 stride 整除/边界判断并 gather；可设计通用索引计算、连续访存和工作划分优化。禁止以减少 dtype 精度换速度。未测带宽、stall，因此不断言它纯属带宽瓶颈。

风险/测试：stride/padding、非连续 col 布局、边界尺寸、batch、累积次序、FP32/FP64，以及卷积数值和梯度对齐。候选最大涉及上述 233.735 μs GPU 工作，实际训练收益待测。

## 3. dWeight im2col 与布局整理

位置：`hytorch/functions_conv.py:147`（Conv2DGradW.forward）、`:376`（im2col_array）、`:468`（_im2col_gpu）；CuPy tensordot 的外部实现也已随包提供。

事实：dWeight 340.299 μs/step；im2col 107.469 μs、FP32 布局复制 107.700 μs、BLAS 与辅助计算 125.130 μs。FP64 cast 已消失，但显式中间 col 和 tensordot 的布局复制仍在。当前 dWeight 走 CuPy/cuBLAS，不会自动使用 native GEMM。

假设与方向：以 cuBLAS 需要的布局直接生成 col、通用 workspace 复用、减少重复物化。先明确实际转置与 stride，不能仅将逻辑 M/N/K 当作 cuBLAS transa/transb。该候选主要影响约 215.169 μs 的 im2col/布局工作，是否值得与 BLAS 融合需另外测量。

风险/测试：多 stream 并发、buffer 生命周期、不同 kernel/stride/padding/batch、布局正确性、高阶梯度与 FP64 保留。此次只观察到 memory pool 请求，不能把逻辑新数组一律认定为每步 cudaMalloc。
