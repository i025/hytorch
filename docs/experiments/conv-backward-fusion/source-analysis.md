# 实际调用路径与实现边界

基线 `549f2ee4f9c046b3eeab2de1a54a47a5a2a16310`；保留阶段 4C 的显式梯度需求、4B Fused SGD、原 Forward 和原 dWeight/dBias。

## 旧路径

[`Conv2d.backward`](../../../hytorch/functions_conv.py) 在 `x.requires_grad` 时调用 `deconv2d(gy,W,outsize=x.shape[2:])`。`Deconv2d.forward` 执行 `xp.tensordot(W,gy,(0,1))`，产生 `[C,KH,KW,N,OH,OW]` 的完整 gcol，再 `rollaxis` 为逻辑 `[N,C,KH,KW,OH,OW]`，交给 Col2im。

CuPy tensordot 把归约轴移到前面，再调用其 core GEMM；历史 Trace 的该分支每层实际有一个 CuPy Copy、一个 cuBLAS SGEMM 和一个 Col2im。rollaxis 本身可为视图，不能把所有 Copy 简单说成 rollaxis 的独立设备拷贝。具体布局 materialization 在 Tensordot/底层矩阵准备内。

gW 仍由 `Conv2DGradW(x,gy)` 的 Im2col + Tensordot 计算；gb 仍为 `gy.sum((0,2,3))`。本轮未改变这些代码。

## 新路径与 ABI

```text
Conv2d.backward
  x.requires_grad=False → gx=None（不调用 native）
  x.requires_grad=True
    conv_backward=original → 原 Deconv2d
    conv_backward=fused
      create_graph=True → 原可微 Deconv2d
      create_graph=False → runtime.conv2d_dinput(gy.data,W.data,x.shape)
        → ctypes hy_conv_dinput(...,current_stream.ptr)
        → conv_dinput_gather<<<ceil(N*C*H*W/256),256,0,stream>>>
        → Variable(gx)
  原 gW 与 gb
```

[`backend(..., conv_backward="fused")`](../../../hytorch/backends/cuda/runtime.py) 为新增关键字参数，默认 `original`。现有四个位置参数保持兼容，ContextVar 退出后恢复。

新实现位于独立 [`conv_dinput.cu`](../../../hytorch/backends/cuda/conv_dinput.cu)，编译进原 `libhytorch_cuda.so`。未修改原 `kernels.cu`、`fused_sgd.cu` 或自动求导调度。ABI 版本仍为 1，新 symbol 在选择该函数时绑定，旧默认路径不要求新 symbol。明确启用 fused 却使用缺少 symbol 的旧库会报错，不静默降级。

runtime 支持同一当前 CUDA 设备上的 **C-contiguous CuPy FP32 gy/weight**，NCHW/OIHW、groups=1、dilation=1、正 kernel 尺寸、每轴 stride 1/2 和 padding 0/1。检查类型、布局、尺寸匹配和 int 索引范围是该 ABI 的实际支持约束。输出只分配一个连续 gx；不调用 ascontiguousarray、不偷偷复制不支持的布局。CPU/FP64/非连续数组明确启用 native 会报错。

## 计算和精度

对每个 `(n,c,h,w)`，枚举 kh/kw，仅当 `(h+pad_h-kh)` 和 `(w+pad_w-kw)` 可被 stride 整除且落在 gy 范围内才取贡献。每个空间位置先按 OC 递增用 `__fmaf_rn` 累加，再用 `__fadd_rn` 加入该输出的空间和。乘法、FMA 累加、空间和及输出均为 FP32。没有 TF32、Tensor Core、FP64、fast-math 或原子加。

这与原来的“先 channel dot、再 spatial sum”分组一致，但不保证等于 cuBLAS/cuDNN 内部归约顺序，须使用已有数值容差验证，不能要求逐位一致。

## 高阶、剪枝与 SGD

普通 `backward(create_graph=False)` 时 Config.enable_backprop=False，可用无梯度图的 native gx；`create_graph=True` 时保留原 Variable 运算来构造高阶图。这是明示的可微路径要求，不是失败后 fallback。高阶图后续在不建图的一阶求值中仍可能调用 native，GPU 测试覆盖此情形；目前该 GPU 项目 NOT_RUN。

CPU 测试已确认：显式开关路由、剪枝不会调用 native、dWeight/dBias 不变、默认上下文恢复、高阶权重/混合梯度正确、插桩前后结果一致。路由测试用 CPU stub，不能作为 CUDA 正确性的证据。4C 既有默认/输入梯度/多层/DonkeyNet/SGD CPU 回归通过。

## 编译事实与占用率边界

`nvcc 12.2.140 -O3 -std=c++14 --resource-usage` 已成功生成 SM80 + compute_80 PTX 动态库。ptxas 对新 Kernel 报告 **40 registers、0 stack、0 spill stores/loads、428 bytes cmem[0]**，源码无 shared memory。动态库已在主机加载并找到新 symbol。

这些是静态编译信息，不是实测 Occupancy，也不代表 Kernel 已在 A100 执行。本轮未申请 NCU 权限、未使用 sudo；GPU 被占用时没有初始化计算上下文或运行 Kernel。未来的资源上限分析应与 Nsight 实际调度数据区分。

## 测量接口

训练显式参数为 `--conv-backward fused`，与 `--prune-input-grad --fused-sgd` 组合为 F。A/B/C/D/E 保持原定义。训练入口只增加选项、传入 backend 和记录 metadata。

独立 benchmark 分开测量 dInput-only 和 **完整 Conv2d.backward 的 dInput/dWeight/dBias**。初始化/H2D/Forward 在计时外，gy/x/W 在设备端准备为相同连续布局；每次调用中的 gx/gW/gb 分配、runtime 检查、ctypes 提交、CuPy 布局准备计入。PyTorch 使用本地 2.4.1 的 `aten.convolution_backward` 原语（已在 CPU 验证 schema），避免把额外 Forward 计进一个框架。

CUDA Event 覆盖完整提交序列，可能包含主机提交不足造成的 GPU 空隙，不等于纯 Kernel 时长；host_submit、end_event_wait、sync_wall 分开保存。另一次不计时调用用 CuPy MemoryHook 或 PyTorch allocator 记录峰值和保留字节，输出张量与临时量分别说明，不能把内存池请求字节当实际 DRAM 流量。

新 NVTX wrapper 只补充 runtime dInput 标签，复用 4A 的逐层跟踪、MemoryHook 和 correlation 解析器；4A 文件未修改。训练吞吐不启用 NVTX 或 allocation hooks。GPU 时间线白区仅说明所选操作事件的间隙，不等于全卡空闲率。
