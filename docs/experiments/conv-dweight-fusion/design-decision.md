# 单一方案：Split-K Tiled Implicit GEMM

基线 `adaaa3cc4489e3ff86e7960e3290af381e61f8c8`，阶段4G。设计在新GPU实验前固定，不按模型、层号或测量胜负切换算法。

原路径的 `col[N,C,Kh,Kw,OH,OW]` 占 `4*N*C*Kh*Kw*OH*OW` 字节。`tensordot(gy,col,((0,2,3),(0,4,5)))` 将两者收缩维移到矩阵轴；跨batch的NCHW/六维col通常不能仅通过view成为所需矩阵，CuPy内部物化Copy。4A补测已按correlationId确认，每层dWeight含1次Im2col、2次Copy和矩阵计算；4F最终Trace中DonkeyNet五层dWeight分别约81.858、128.645、58.626、36.936、27.786 μs/call，合计333.851 μs/step。各层4–5个Kernel，末层矩阵后处理不同，不能一律写成5。

设计前提取的4F历史逐层数据如下，时间为μs/call，字节为完整col逻辑大小，均可追溯到`results/history-dweight.csv`中的原始Trace路径。它们只作为方案选择依据，不参与4G正式加速比。

| 历史诊断标签 | col字节 | Im2col核时 | 两次Copy核时 | cuBLAS GEMM核时 |
|---|---:|---:|---:|---:|
| Donkey conv1 | 3,645,600 | 29.714 | 22.774 | 20.690 |
| Donkey conv2 | 6,316,800 | 48.132 | 41.980 | 33.172 |
| Donkey conv3 | 1,584,000 | 15.482 | 17.630 | 20.208 |
| Donkey conv4 | 414,720 | 7.234 | 11.904 | 12.748 |
| Donkey conv5 | 165,888 | 6.076 | 9.346 | 12.364 |

计算量为`2*M*P*R`浮点运算。Im2col至少完整写出并再次读取col，布局物化还有额外写/读；这是数据量与流量下界，不是实测DRAM带宽。较大的空间归约与相对小的权重矩阵同时需要并行归约及矩阵数据复用，因此选择Split-K分块方案，而不是逐权重长串行dot。单凭减少Launch数不保证弥补cuBLAS计算效率差距。

以 `M=Cout, P=Cin*Kh*Kw, R=N*OH*OW` 表示 `dW[M,P]=gy[M,R] @ implicit_x[R,P]`。新CUDA C/C++计算块覆盖16×16输出，256线程协作装入两个16×16 shared tile，按通用几何即时计算x地址，每个加载的gy/x供16个输出复用。不是每个权重线程独立扫描整个R，也不调用cuDNN或现有GEMM Kernel。

R固定分成128项的split：每块最多8个K-tile，单FP32 FMA链最多128项。工作区为 `ceil(R/128)*M*P` 个FP32部分和；第二个CUDA Kernel以32个连续权重×8个归约lane协作，将split按8组短链累加再共享内存树形归约。不用全局原子，结果不依赖CTA执行次序。两Kernel构成同一个算法，不追加其他优化方案。

这个方案避免完整Im2col和其布局Copy，保留矩阵分块复用，并增加长R/小权重矩阵的并行CTA数量。成本是部分和工作区、第二次Launch、地址除余、边界判断和shared同步；工作区不保证所有形状都小于原col。16×16边界和不足128的尾段零填充，全部shape使用同一算法，不为1×1、小Cout或网络某层另加专用路径。

第一版支持当前设备连续FP32 NCHW x/gy，OIHW输出，groups=1、dilation=1，合法正kernel/stride与非负padding；数组、输出和工作区均沿用INT_MAX元素限制。测试覆盖每轴stride1/2、padding0/1及矩形kernel/空间，不宣称验证了所有合法整数参数。线程块线性索引避免grid.y/z的65535限制。

`backend(conv_dweight="fused")` 显式启用；默认original。仅普通一阶反向启用原生路径，创建高阶图仍使用原Conv2DGradW。既有dInput、Fused SGD、Forward、dBias不改。GPU正确性不通过则停止该窗口的性能任务，保存失败输入和结果，不放宽容差。

纯设备、完整算子、未插桩训练及Nsight主机范围分别统计。新算法即便减少Copy/Launch，也可能慢于cuBLAS；性能失败属于有效实验结果。所有GPU任务在准备完成后，由一个持续存活的控制程序固定UUID顺序执行，结束后释放context，再做CPU导出/分析/绘图/推送。
