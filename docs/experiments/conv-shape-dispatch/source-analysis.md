# 调用路径、支持范围与选择器

基线为阶段 4D GPU 补测提交 `d26667f160bf3af772ad24685f73049f6267a11a`。本轮不修改任何 CUDA C/C++ Kernel、优化器或自动求导调度算法。

## 算法与选择分离

[`Conv2d.backward`](../../../hytorch/functions_conv.py) 先检查既有 `x.requires_grad`。无需输入梯度时直接保留 `gx=None`，不会调用选择器，dWeight/dBias 正常计算。

```text
需要输入梯度
  original → 原 Deconv2d
  fused    → 原 runtime.conv2d_dinput；建高阶图时用可微 Deconv2d
  auto     → runtime.conv_dinput_dispatch（只读元数据，记录实际选择和原因）
               CPU / create_graph → original
               其他 → select_conv_dinput(shape, weight_shape, stride, pad, dtype, contiguous)
                        满足受限规则 → fused
                        其他 → original
原 Conv2DGradW(x, gy) 与 gy.sum((0,2,3))
```

[`conv_dispatch.py`](../../../hytorch/backends/cuda/conv_dispatch.py) 不接收模型名、层号、训练脚本或 Tensor 数值，不访问 GPU，不分配工作区，不计时。没有在线 Autotuning、缓存搜索、异常后重试或隐式转换。`runtime.py` 负责当前数组元数据和计数；算法继续由原实现负责。

默认仍为 `original`。显式 `fused` 的原有支持检查和报错保留；`auto` 对未覆盖输入的 original 是预定义选择，而不是执行失败后静默降级。库缺失、ABI 错误或 Kernel 错误继续正常报错。

## 本轮 Auto Gather 的实际支持域

连续 CuPy FP32 NCHW 输入/梯度、OIHW 权重，三者 dtype 相同；当前 Conv2d API 本来就只有 groups=1、dilation=1。

- Kh=Kw=1，stride=(1,1)，padding=(0,0)。
- 1≤N≤16，3≤Cin≤64，16≤Cout≤64。
- 3≤H≤65，5≤W≤67，输入/权重 Cin 一致。
- `N*Cin*H*W*Cout <= 1,161,216`。

不满足任一条保留 original，计数原因为 geometry、dtype_layout、shape 或 work；CPU、高阶图另有 cpu、higher_order 原因。矩形空间、不整块边界、包络内未见 Batch/通道值均由同一条件处理。该规则只在本次 A100 上有性能证据，不宣称适用于其他 GPU。

FP16/BF16、其他 layout、groups/dilation 扩展应先扩展算子和算法能力、测量数值/性能，再扩展元数据策略；本轮没有虚构这些 API 或建立大型注册系统。3×3/5×5 可以由 Auto 处理，但始终选择原算法；不能描述为这些形状已获得可靠的 Gather 自动预测。

## 两种原有计算机制

Original 的 `Deconv2d.forward` 执行 `tensordot(W,gy,(0,1))`，产生完整 gcol，再 rollaxis 为逻辑 `[N,Cin,Kh,Kw,OH,OW]`，交给 Col2im。rollaxis 可以只是视图；设备 Copy 的归因应依 CUDA correlation 指向的 Tensordot/布局准备，而不能直接称为 rollaxis 拷贝。

原 [`conv_dinput.cu`](../../../hytorch/backends/cuda/conv_dinput.cu) 每线程计算一个 dInput 元素，枚举 kh/kw 和 Cout；通道和使用 `__fmaf_rn`，空间和使用 `__fadd_rn`。不创建完整 gcol，没有原子加、Tensor Core、TF32 或 FP64。权重/gy 重读、串行 Cout 循环可能降低纯设备计算效率。1×1 简化空间循环，但仍有通道归约成本；这只是规则的机制解释，不替代测量。

调用仍为 `ctypes → hy_conv_dinput → conv_dinput_gather`，传原 `_call` 获取的当前 Stream。没有新 Kernel、RawKernel 或 ElementwiseKernel。dWeight 保持 Im2col + CuPy Tensordot，dBias 保持归约，Forward 和 Fused SGD 未改变。

## 跨模型训练与发现的测试模型类型问题

训练脚本仅增加独立的 SpatialNet：3×3(3→16)、stride2 3×3(16→32)、1×1(32→32)、空间平均、Linear、Tanh。输入 `[4,3,16,16]`；匹配 PyTorch 模型共享初始权重。它包含不同空间/通道形状，不参与规则拟合。DonkeyNet 结构完全保留。

初版新模型用 `F.average`，该库路径把 Python 浮点归一化因子转成 CuPy FP64 零维数组，产生 FP64 参数梯度；CPU 用例没有暴露这一 GPU 类型提升。首次 GPU 运行 132 passed / 4 failed，Gather dInput 和 Fused SGD 分别正常拒绝 FP64。修复只在新增 SpatialNet 中使用 `F.sum(...) * np.float32(1/(H*W))` 表达相同空间平均，新增 dtype 断言；没有对优化器或通用 F.average 加 cast、catch 或 fallback，没有改正式容差。完整失败日志和已产生的比较结果仍保留；类型错误前未单独捕获该新模型现场输入，另有按固定seed离线重建的首批输入及明确 provenance，不能称为现场捕获。最终验收数字见 README 和原始 XML。

## 计时和归因

单算子完整操作包括每次分配、布局准备、Python/runtime/ctypes 提交和尾事件同步；输入 H2D/初始化在外。`auto` 在每个被计时调用中真正调用选择器与计数器，不把一次离线决定伪装为运行时性能。

CUDA Event 是完整流跨度，含主机提交不足造成的间隙，不等于 Kernel 执行和。Nsight 单独采集，使用 correlationId → 提交 API → 同线程 NVTX 父节点；PyTorch worker 的跨线程 phase 采用已有明确标记的规则。GPU Kernel 和、Host NVTX、API 时间、完整墙钟互不替代。MemoryHook 的请求/池内存不是 DRAM 流量，PyTorch allocator 口径也单独标注。

架构参考为 [PyTorch 的算法 benchmark 开关](https://docs.pytorch.org/docs/main/backends.html) 和 [cuDNN execution plan/heuristics](https://docs.nvidia.com/cudnn/latest/developer/overview/)。只借鉴算法与元数据选择分离；HyTorch 原路径与 Gather 没有转交 cuDNN。PyTorch 对照的 cuDNN benchmark 关闭、deterministic 开启，TF32 关闭。
