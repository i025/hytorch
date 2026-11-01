# 实现与边界

基线 `e2d60c26cdb28facd4a41dd3b443c3148e397b66`。旧 `conv_dinput.cu`、SGD、GEMM、Pooling、dWeight/dBias和4E选择规则均保持原文件内容。本轮只新增一个CUDA Kernel。

## 计算机制

旧Gather以256线程block线性覆盖NCHW输出；每线程一个dInput，按Kh/Kw遍历合法gy位置，Cout递增使用一条 `__fmaf_rn` 链；空间部分和通过 `__fadd_rn` 合并。相邻线程多数沿W合并读取gy，Weight相同地址可复用；是否命中缓存、是否受内存或指令延迟限制，没有NCU硬件证据，不能仅从源代码定性。

新 `conv_dinput_opt.cu` 保留全部索引/空间循环/线程映射，仅把Cout按4递增：p0/p1/p2/p3分别处理通道 `4j+0/1/2/3`，最后 `(p0+p1)+(p2+p3)`，不足4通道用原FMA尾循环。不使用共享内存、warp通信、原子、Tensor Core、FP16/TF32或FP64。通道使用FP32 FMA的单次舍入，空间合并和链间归约使用显式FP32加法；改变的是舍入分组，非卷积数学公式。

四路是固定的指令并行度方案，不是按模型选择算法，不是固定某组张量尺寸；N/Cin/Cout/H/W/Kh/Kw/OH/OW/stride/pad全部来自ABI参数。相邻线程的访问方向不变，没有宣称减少Global load字节数；增加寄存器/归约指令可能带来退化。

## Runtime 和自动求导

```text
Conv2d.backward
  x.requires_grad=False → gx=None，不调用选择器或native
  original → 原可微Deconv2d
  fused → 原conv2d_dinput
  fused_opt → conv2d_dinput_opt → 共用原输入检查/分配/Stream ABI
               → hy_conv_dinput_opt → conv_dinput_gather_opt
  auto → 原4E选择器；只可能original/旧fused，不会选择新Kernel
  create_graph=True → 原可微Deconv2d
  所有模式均保留原Conv2DGradW及gy.sum((0,2,3))
```

`backend(conv_backward="fused_opt")`和训练CLI `--conv-backward fused_opt`显式启用，默认original。新ABI符号沿用13个几何整数、3个指针和当前Stream指针。共同的 `conv2d_dinput` 新增keyword-only variant，旧调用默认仍用原symbol；薄包装避免复制检查/数学逻辑。缺库/符号和CUDA错误照常报错，不加catch或隐式fallback。

支持同一当前CUDA设备上的连续FP32 NCHW gy、OIHW权重，groups1/dilation1、正Kh/Kw、每轴stride1/2和padding0/1，沿用原INT_MAX尺寸约束；不支持的显式调用按原检查报错。输出只分配一个gx，不创建gcol或布局Copy。未改Forward、dWeight、dBias、SGD、网络结构或Auto阈值。

## 编译与CPU证据

Toolkit12.2的 `nvcc --resource-usage -O3 -std=c++14` 同时编译新旧版本，SM80和compute80 PTX，动态库ABI1及新旧symbol均可在CPU主机加载。ptxas和cuobjdump一致：旧40、新48寄存器；两者stack/local/spill/shared均0，常量参数428字节。寄存器增加20%可能减少驻留上限；这不是实测Occupancy，本轮不申请NCU权限。

`check_host_equations.py`读取新旧实际CUDA函数体，替换CUDA内建线程索引为串行host索引、显式FMA为C++ float `std::fma`、FP32加法为float加法，不复制另一套计算公式。以CPU编译体在124形状各约128个随机位置及首尾位置比较旧Gather和PyTorch CPU：32,076条逐位置比较通过，最大误差6.5565109e-7。不是全张量CUDA执行，也不能作为GPU性能证据。

## 首次真实GPU测试问题的归因

首轮52 failed / 394 passed：48个失败来自复用的4E测试helper假设stride/pad为标量，无法接受新集合的tuple；另外c23/v19的Forward与默认cuDNN参考各在两类Stream上失配。现场输入/实际输出/原始日志全部保留。

独立CPU FP64诊断显示：c23中CuPy最大绝对误差2.6154e-7、PyTorch原生FP32为2.9309e-7、cuDNN为0.00508193；v19分别3.0325e-7、3.4860e-7、0.00712423。TF32关闭时仍观测到cuDNN差异，具体内部算法原因本轮未确定；不是新dInput Kernel造成，因为该Forward在原路径也相同。

只改本轮测试：实现通用pair几何处理，对全部形状统一使用PyTorch原生CUDA FP32作为正确性参考，保留原 `atol1e-5/rtol1e-4`，不按形状择优换参考。dInput Benchmark计时、训练PyTorch及端到端对照仍用标准cuDNN，额外保留其实际输出和原生参考误差，不把正确性参考选择混进性能配置。之后完整446项GPU验收通过；第二次尝试因GPU0外部任务进入被guard中止，单独留档，最终成功窗口使用GPU2。

## 测量口径

普通完整dInput测量包括Python检查、分配、ctypes提交及尾事件同步；输入H2D、参考计算在计时外。另测预分配gx、预绑定ABI/参数的kernel范围以隔离准备成本；CUDA Event跨度依然可能包含主机提交间隙。

纯设备Kernel时间取独立Nsight CUPTI activity，按correlationId关联提交API与父NVTX；最终GPU1窗口已完成每形状30次、3个独立profiling进程，26份Trace解析通过。原先GPU2的2进程部分数据仍标记PARTIAL、独立保留。与GPU2已完成3重复的未插桩吞吐、Event跨度、Host提交、同步墙钟、MemoryHook/allocator口径分别记录，不跨设备混算收益。普通训练不启用NVTX、Nsight或内存hook。未将GPU活动区间并集叫作GPU/SM利用率。
