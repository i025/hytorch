# 通用 Shape-Aware GEMM Dispatch：A100 验证

## 三个问题的回答

1. **目标小矩形恢复到 shared 水平**：tile16 无 profiler CUDA-event中位数，(240,64,432) 81.616→21.088 μs（3.870×），(72,64,576) 67.744→17.248 μs（3.928×）。独立直接调用CUPTI分别95.392→25.2645、124.976→32.320 μs，和强制shared的25.248/32.3045 μs一致到小于0.1%的差异。
2. **大方阵收益保留**：1024³/2048³/4096³在两个tile配置下全部仍用register，与旧dispatch接近，最大这次中位延迟变化约0.87%；相对forced shared仍有1.75–2.47×加速。
3. **端到端仅微小收益**：7轮交替同进程配对，wall均值0.834571→0.831816 s，仅减少0.330%。前向GPU省下约194 μs/step，但不能直接加到/减出wall。不能用约4×局部加速宣传显著训练加速。

**128 CTA不是已验证最优的通用阈值**。额外矩形测试发现120/127 CTA反例，延迟增加10.77–26.39%。实现严格保留用户指定第一版，交由ChatGPT审查，不自行重调启发式。

## 代码和可靠基线

main HEAD edb6fbe3bfaa9eed461addd29e6943b22e28216e，工作区开始已有前轮FP32修复、register kernel和测试/文档未提交修改。旧CUDA库在 lib_before.so、旧CUDA源码在 raw/kernels_before.cu，SHA随清单保存。

补齐最终整数dtype保护GPU回归后才将新策略作为验收候选。基线56GPU/31host通过；新版102GPU/31host通过。hytorch/functions.py和runtime.py字节不变。详见CORRECTNESS.md及SOURCE_CHANGES.md。

生产变化只有hy_gemm的shape判断：m/n/k>=64且ceil(M/64)*ceil(N/64)>=128才用register。其他variant、ABI、stream、kernel数学、精度门限全部不变，没有设备属性查询、额外同步、TF32、fast-math或网络专用shape特例。

## 实验环境与方法

物理GPU1 A100-PCIE-40GB，108SM，逻辑CUDA0；GPU0 VLLM服务保持。每次阶段启动前确认GPU1空闲，没有终止别人的进程或锁定时钟。北京时间约11:43–11:49同一实验会话，因两个诊断环境错误有短暂停顿，进程间均重新查空闲；最终GPU1无本轮进程/显存占用。完整UTC时间在命令JSON。

CUDA nvcc12.4.99，sm_80；驱动580.95.05；CuPy14.2.0、NumPy2.2.6、PyTorch2.9.1+cu128。具体构建argv、CPU线程数、环境、GPU快照均随包保存。未锁频，因此不将不同窗口绝对延迟直接比较。

四实现同进程切换已加载库/标准cuBLAS handle，同一随机输入和预分配输出。47组M/N/K：24组来自已有实际FP32 DonkeyNet instrumentation（同模型/配置未改，原完整记录已复制），另23组为方阵与通用边界矩形；两个tile，4实现，共376行。实际训练trace核对8个前向native调用，其中只conv4/conv5改变kernel。

seed=9000+shape_id；5次普通warmup并额外预上传/回放graph；12轮正反交替。事件包围CUDA Graph回放：大工作量图含1次调用，中等8次，小工作量32次，除以次数。排除H2D/D2H/输出验证和逐kernel的Python提交间隙。该结果代表重复同矩阵的热缓存吞吐；不把它冒充真实训练缓存状态或孤立单次launch延迟。

CuPy的cuBLAS wrapper不支持当前stream capture，实际cuBLAS组用同一库/handle的标准SGEMM接口，默认FP32 math，NVIDIA_TF32_OVERRIDE=0。它是cuBLAS参考，不是声称调用了cp.dot。每组实际kernel从graph DOT及另外2次直接调用的Nsight记录核对。主性能统计12次事件；2次profile仅辅助kernel/配置核验，不能冒充12次profile统计。

## 小矩形结果

| M,N,K | tile | 实现 | 无 profiler event μs | 独立 CUPTI kernel μs | CTAs |
|---|---|---|---|---|---|
| 240,64,432 | 16 | old | 81.616 | 95.392 | [4] |
| 240,64,432 | 16 | new | 21.088 | 25.264 | [60] |
| 240,64,432 | 16 | forced_shared | 21.088 | 25.248 | [60] |
| 240,64,432 | 16 | cublas | 13.568 | 17.840 | [96, 30] |
| 72,64,576 | 16 | old | 67.744 | 124.976 | [2] |
| 72,64,576 | 16 | new | 17.248 | 32.320 | [20] |
| 72,64,576 | 16 | forced_shared | 17.248 | 32.305 | [20] |
| 72,64,576 | 16 | cublas | 7.456 | 15.872 | [54, 10] |


Graph事件和直接CUPTI绝对时间不同源于缓存/提交/采样条件，分别比较同口径的旧/新/forced shared，不交叉计算加速。microbench中conv4/5旧register分别4/2CTAs，新shared为60/20CTAs（tile16）；两者输出逐位一致。

## 大型方阵

| size | tile | 旧 μs | 新 μs | forced shared μs | 新相对 shared 加速 |
|---|---|---|---|---|---|
| 1024 | 16 | 258.048 | 258.048 | 471.040 | 1.825× |
| 1024 | 32 | 245.760 | 245.760 | 430.592 | 1.752× |
| 2048 | 16 | 1627.136 | 1626.112 | 3694.080 | 2.272× |
| 2048 | 32 | 1584.640 | 1582.592 | 3343.872 | 2.113× |
| 4096 | 16 | 13171.200 | 13167.616 | 32526.337 | 2.470× |
| 4096 | 32 | 12109.824 | 12215.296 | 29223.936 | 2.392× |
| 65 | 16 | 13.248 | 4.384 | 4.384 | 1.000× |
| 65 | 32 | 14.304 | 7.264 | 7.264 | 1.000× |


1024³、2048³、4096³的register候选CTAs分别256、1024、4096，全部超过阈值，实际kernel为gemm_register<16/32>。大矩阵既有收益保留，未重写数学实现。

## 泛化检查发现的反例

| M,N,K | tile | register candidate CTAs | 旧 μs | 新 μs | 新版延迟增加 |
|---|---|---|---|---|---|
| 64,8128,65 | 16 | 127 | 16.640 | 20.992 | 26.15% |
| 64,8128,65 | 32 | 127 | 18.432 | 23.296 | 26.39% |
| 449,897,71 | 16 | 120 | 16.640 | 18.432 | 10.77% |
| 449,897,71 | 32 | 120 | 18.304 | 23.104 | 26.22% |


这些是测量事实。单一CTA门限忽略K长度、长宽比、每CTA工作量及tile，可能在120/127等阈值附近过于保守。这里的解释是性能假设，不是硬件计数器证明；本轮没有测occupancy/带宽/stall。报告完整披露反例，不能宣称“所有小矩形都更快”。

其他形状和波动（mean/std/min/max、12次原值）全部在GEMM_COMPARISON.csv。DISPATCH_DECISIONS.csv列实际名字/CTA/grid/block，new退回shared与forced_shared性能一致并不证明128是最佳界限。

## 端到端配对训练

固定seed42、相同初始权重/数据hash、256合成样本、batch4、3epochs、192steps、SGD .001、无dropout/momentum/weight decay。每模式10warmup，7轮交替且每次恢复初始状态；模型、FP32修复、SGD、数据、计时方式完全一样，唯一区别是CUDA library的dispatch。

wall包含CPU DataLoader、pageable H2D、前后向、更新和最终同步，无逐步日志/loss读取/显式同步；两模式同样phase NVTX。loss与权重读取在计时外。

| 实现 | wall s（均值±SD） | samples/s 均值 | 最终 loss | 与旧版最终权重 |
|---|---|---|---|---|
| old | 0.834571 ± 0.002830 | 920.24 | 0.000255663209828 | 逐位相同 |
| new | 0.831816 ± 0.002180 | 923.29 | 0.000255663209828 | 逐位相同 |


7个配对均新版略快，平均省2.755 ms/192steps；配对差值样本SD=2.221 ms。以t分布6自由度计算的描述性95%区间为[0.701, 4.809] ms。区间仅针对本窗口7次测量，不保证不同GPU负载/时钟/模型下持续收益。工程意义上很小，不应宣传成显著端到端提速。

所有14次训练的loss完全相同，最终权重hash均为2cf9c1a0745128aa7809d046befd5839b3727209d86f26d23d7f1c9e5791f094；相对旧版最大绝对权重差0。

## 时间线追溯（独立10-step trace，单位μs/step）

| 实现 | GPU kernel 总计 | Forward GPU | Backward GPU | SGD GPU | Native GEMM GPU | CUDA API CPU elapsed | 同步 API CPU elapsed |
|---|---|---|---|---|---|---|---|
| old | 1752.115 | 751.827 | 880.524 | 119.763 | 524.076 | 711.094 | 0.758 |
| new | 1556.413 | 557.484 | 880.705 | 118.224 | 329.966 | 689.204 | 0.788 |


Native GEMM是Forward中的子集，不能再与Forward相加；API CPU elapsed和GPU duration重叠，不能相加。TRAINING_COMPARISON.csv把各模式一份trace汇总附到7轮wall记录，每行明确注明不是7次profile。

真实训练conv4旧register113.226→shared30.093 μs；conv5 149.457→38.685 μs，合计省193.905 μs，解释了前向整体194.343 μs减少。其他6个native GEMM仍shared且时间接近；backward仍走CuPy/cuBLAS，880.524→880.705 μs基本不变，FP64转换kernel仍为0。kernel总数均147/step，主要是选择更快kernel而非减少launch。

GPU kernel+memcpy活动时间并集占profile墙钟：旧34.54%、新31.06%。这是trace时间覆盖比例，**不是occupancy或硬件利用率计数器**。GPU工作缩短后活动间隙比例增加，和wall收益较小一致；CPU逐算子提交、数据准备、异步重叠使GPU工作节省大部分未变成wall节省。CPU未开栈采样，不能把全部间隙精确归因于Python。

## 正确性限制与后续决定

所有old/new/forced_shared输出逐位相同，FP64误差指标无新增回退；但原严格NumPy FP32 allclose gate仍有22行失败（长K与2048/4096，包含cuBLAS的方阵失败）。门限和失败记录均保留，详情CORRECTNESS.md。不能以相对L2代替严格门限通过。

128 CTA启发式成功修复指定两形状并保留大方阵收益，但已经存在通用矩形反例，端到端仅0.33%改善。是否接受当前阈值、进一步设计通用dispatch，或转向col2im/dWeight，留给ChatGPT审查。本轮不进行这些后续优化，不提交/推送/创建PR。
