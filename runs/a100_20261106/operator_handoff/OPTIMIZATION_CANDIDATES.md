# 最多三个优化候选（仅供 ChatGPT 设计，不是已实施补丁）

本轮没有优化或修改生产算子。以下影响范围使用现状耗时，不代表可实现的加速比，也不能把重叠范围相加。
DonkeyNet 仅是观测 workload；任何后续方案需覆盖一般 batch、通道、空间尺寸、kernel、stride、padding、layout 和支持的 dtype。

## 1. 优先澄清标量类型合同，阻止非预期 FP64 梯度传播

**源码位置**：`hytorch/core.py:170 as_array`、`:264 mul`、`:300 rsub`；`hytorch/functions.py:475 MeanSquareError.backward`、`:53 Tanh.backward`。

**现状问题与证据**：Python `2./len(diff)` 经 as_array→CuPy 0D float64 数组，实际 `Mul.forward` 输入 FP32/FP64、输出 FP64；`1-y*y` 的 int64 标量还导致另一次升型。`raw/workload/operators.json` id 83–91 可直接复核。FP64 传播到全部 5 层卷积和 3 层 FC 的梯度；16 个 FP32→FP64 copy kernel/step 累计 **238.469 µs**，占全部 GPU kernel 时间 **12.76%**。其中 dWeight 的列张量转换占 **191.134 µs/step**。cuBLAS 名称中的 d884 / f64 和实际输出 dtype 相互印证；不是仅凭 kernel 名字推测。

**可能方向**：先由 ChatGPT 定义与验证标量转换、混合 dtype 和自动求导的精度合同，再决定如何传递标量或融合 backward elementwise。对原本应保持 FP32 的计算消除非预期扩宽，可能同时减少转换、临时内存和 double GEMM。不能把所有输入强制 cast 为 FP32，也不能破坏用户明确要求的 FP64 计算。此候选是继续 CUDA 优化前的语义问题，应先独立处理和重新建立基线。

**正确性风险**：NumPy/CuPy 标量提升规则、整数常量、真正的 mixed dtype、广播、高阶梯度、loss 的缩放/归约和 SGD 舍入都会受影响；不能把改变后的结果与现状“逐位相等”当作唯一门限。

**需补充测试**：FP32/FP64 tensor × Python float/int × NumPy/CuPy scalar/0D array 的矩阵；MSE/Tanh/ReLU 梯度 dtype 断言；CPU/GPU 一致性、有限差分/高精度参考、二阶导数；固定初始化的完整训练权重/loss 对齐。原有误差门限不得随意放宽。

**影响范围**：广泛的反向 GEMM、copy、activation、SGD 中间量；238.469 µs 是现状转换 kernel 总时间，不是修复后必然节省量。未预测 speedup。

## 2. Conv2D dWeight 的 im2col、dtype/layout packing 与 GEMM 工作区

**源码位置**：`hytorch/functions_conv.py:147 Conv2DGradW.forward`、`:376 im2col_array`、`:468 _im2col_gpu`；CuPy 源码 `source/external/cupy/linalg/_product.py:358 tensordot`。

**现状问题与证据**：每层重新生成 6D im2col；CuPy 先把归约轴移到前部，再由 tensordot_core 处理 reshape/packing/dtype。实测 dWeight 合计 **496.125 µs/step**，分解为 im2col **98.247**、FP32→FP64 **191.134**、同 dtype/layout copy **112.023**、BLAS/其辅助计算 **94.721 µs**。conv2 最突出：dWeight **220.171 µs**；其 FP32 列工作区 **6,316,800 bytes**。完整形状、stride 与请求分配记录在 WORKLOAD/raw 中。

**可能方向**：先在候选 1 的 dtype 合同明确后，设计面向 dWeight 的通用矩阵布局，尽量让 im2col 输出被 GEMM 直接消费；评估前向列矩阵的受控复用、融合 packing，或通用 implicit-GEMM。不能仅把现有 native GEMM 接到 backward：目前输入 dtype/布局、transpose 和归约维度均不同。

**正确性风险**：NCHW/OIHW 轴顺序、stride/pad 边界、non-contiguous view、归约数值顺序、工作区生命周期、训练 step 间 stale cache、高阶求导的保存张量语义。

**需补充测试**：随机/不规则 N/IC/OC/H/W/KH/KW、非方形卷积、非整齐尾块、非对称 stride/pad、bias 有无；dWeight 与 FP64 参考、CuPy/框架对齐；non-default stream、连续/非连续输入、重复 step 参数更新；多并发调用与 memory sanitizer。不要硬编码本次 conv2 的 shape。

**影响范围**：dWeight 路径占全部 kernel **26.55%**，其中转换部分与候选 1 重叠。CPU 调度可能仍限制端到端；去掉 workspace 也可能付出新的访存/计算代价。无已测加速比。

## 3. Conv2D dInput 的 col2im gather 与列工作区布局

**源码位置**：`hytorch/functions_conv.py:95 Deconv2d.forward`、`:442 col2im_array`、`:507 _col2im_gpu`。

**现状问题与证据**：dInput tensordot 生成 `(IC,KH,KW,N,OH,OW)`，rollaxis 后成为非连续 `(N,IC,KH,KW,OH,OW)`；col2im 每线程一个输入像素，循环 KH×KW 并检查 stride 整除/边界。实测 dInput **344.038 µs/step**，col2im 占 **215.211 µs**（约 **62.55%**）。conv2 col2im 单次 **87.803 µs**，grid=(2279,1,1)，block=(128,1,1)，33 registers/thread、0 shared bytes。其输入 FP64 工作区为 **12,633,600 bytes**，strides 明确见 REPORT。

**可能方向**：通用边界/索引计算优化、使列布局更适合合并读取、评估融合 dInput GEMM 与重叠累加或 tile 内局部归约。若考虑省略不需要的 leaf dInput，应先设计通用梯度需求传播，不能在第一层按名字硬编码。

**正确性风险**：重叠窗口累加、padding 边缘、stride>1 的空位置、非连续列张量、FP64/FP32 语义、原子累加的非确定性、输出尺寸推导、二阶梯度与异步流。

**需补充测试**：跨 batch/通道/空间/kernel/stride/padding 的 dInput 参考测试，含重叠和不重叠窗口、奇数尺寸、非对称参数、non-default stream、非连续输入；如果改变归约次序需用原门限/高精度参考验证；racecheck/memcheck。

**影响范围**：col2im 占全部 kernel **11.52%**；不代表能缩短 wall 同样比例。未采集 bandwidth/occupancy/事务指标，所以“内存带宽瓶颈”仍是待验证假设，不能写成已测事实。
