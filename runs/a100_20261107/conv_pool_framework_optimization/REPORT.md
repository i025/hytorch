# Conv2D Backward / Pooling 最终框架优化报告

## 结论

保留一个默认关闭的 O1：`Config.compact_function` 控制 `Function.__call__` 的单输出包装快路径。仅新增 11 行生产代码，不更改数学运算、CUDA kernel、GEMM dispatch、Col2im、dWeight 或 SGD。所有正常图连接和必要检查仍执行。

在已经开启 F1 的同进程 12 轮正式训练中，B0 **0.764135s → O1 0.732040s**，wall time 减少 **4.200%**，速度比 **1.0438×**，12/12 轮胜出。每对节省均值 **32.096ms**，Student-t 95% CI **[25.621, 38.570]ms**。loss、全部参数梯度和最终权重逐位一致。

HyTorch/PyTorch 训练耗时比由 **2.4344× → 2.3321×**。绝对差距从 450.242ms 降到 418.147ms，差距减少 7.13%。这不代表已经接近 PyTorch 端到端水平。

## 实验范围、资源与基线

- A100-PCIE-40GB，物理 GPU1，UUID `GPU-0bc09ee6-967a-b53d-bf80-38379585cf08`，`CUDA_VISIBLE_DEVICES=1`。每个任务启动前查询资源，无他人进程才执行；不终止进程、不改 GPU 时钟。
- 主实验窗口：2026-11-07 16:09–16:22 UTC。实际 TinyNet Pooling 补测：2026-11-08 07:39 UTC。会话跨日，因此**不是所有数据来自单一连续窗口**；补测明确独立列出，没有用它与主窗口绝对时间相减。主训练 12 轮均在一个进程与连续窗口内完成。
- 补测也重新检查 GPU1 空闲。最终 `raw/RESOURCE_RELEASED.json` 为 GPU1 0 MiB、0% 且无计算进程。资源检查是瞬时检查，不是排他预留。
- `baseline/` 保存本轮前 Git状态/HEAD/完整已有diff、83份源码/构建文件及SHA，包括实际动态库；既有 FP32 修复、128 CTA GEMM、Native Col2im POD/cache、packed dWeight 均保留。
- B0 显式 `metadata_cache=True`，不把 F1 换名再实现。O1 唯一变量为新 Function 开关。Legacy SGD，默认后端设置不变。
- Torch 2.9.1+cu128，CuPy 14.2.0，TF32 matmul/cuDNN均关闭，cuDNN正常启用且benchmark=False，torch.compile关闭，SGD foreach=None/fused=None（保留框架默认优化）。不启用fast-math。

## 实际瓶颈及证据

### Conv2D Full Backward

路径是 `Variable.backward → Conv2d.backward → Deconv2d / Conv2DGradW / Sum`。dInput仍用 CuPy tensordot、rollaxis、Native Col2im；dWeight仍生成 fresh im2col，并执行 gy transpose/reshape packing 和 CuPy GEMM；dBias仍单独reduce。完整图还包含输入/输出 Variable 包装、弱引用、拓扑排序和梯度累加。

30步真实训练 cProfile 记录 `Function.__call__` **1980次（66/step）**，优化前后次数不变；`np.isscalar` 从 **2836 → 856次**，减少1980次，保留必要的标量转换。Function self elapsed 8.259→7.601ms，isscalar self elapsed 2.335→0.807ms；这些是独立 profiler 数据，不作为正式加速率。额外输出tuple/list/listcomp/单元素循环在快路径中消除，Variable和图边没有消除。

conv4 代表性分层 CPU提交（B0，无profiler中位数）：预验证直调 178.67μs；带安全检查原始计算 190.32μs；Function反向子调用 221.38μs；完整autograd 260.24μs。各层均计算x/W/b三类梯度，GPU数学路径不变。差值是诊断上限的线索，不能宣称全部可安全删除。

conv4 Full Backward HyTorch实际8个kernel、7个CuPy分配请求、1,147,648 bytes请求；Torch4个kernel、3个分配请求、195,328 bytes。GPU kernel相近不代表工作分解/packing/内存行为相同。诊断cProfile里 reshape 的elapsed还可能含GPU背压，不能全算作Python解释器成本。

### Pool Forward / Backward

DonkeyNet没有Pooling，不把Pooling局部收益算入它的训练收益。TinyNet真实池化位于3→4通道卷积和ReLU之后，输入是 **(4,4,16,16)**，kernel2、stride2、pad0；`WORKLOAD_TINY_POOL.json`为运行时证据。历史探针名字`tiny_pool`使用(4,3,16,16)，本报告把它列为**通用3通道形状**，不冒充真实TinyNet。

真实4通道Pool：预验证且预分配的Native直调 CPU提交 6.34μs；正常runtime含校验/两次分配 25.10μs；完整Function 43.98μs；Torch 15.42μs。GPU kernel均约5μs。直调跳过检查并预分配，只是性能下限，不能直接拿它替换安全生产接口。

逐项微计时见 `CPU_OVERHEAD.csv`：backend selection、_arrays、_geometry（已启用F1）、stream查询、指针封装、输出与indexes分配。参数校验、两次数组分配、ctypes和Function包装共同构成完整调用，不能把全部差距都归为ctypes。Pooling backward还需要scatter/索引/col2im，真实4通道HyTorch4 kernels、Torch2 kernels，O1不改变这些工作。

## 修改选择与安全性

O1 对非tuple单输出直接构建一个Variable；已知ndarray跳过np.isscalar，其他标量/None仍走原as_array；同样设置generation、creator、inputs和weakref outputs。tuple返回（含一个元素与多输出）原路径不变。no_grad原有inputs/outputs保留语义不变。

未修改 `Variable.backward` 的遍历、排序、梯度累加顺序，也未绕过Function.forward。所有dtype/device/buffer/alias检查原封不动。无任何GPU地址、stream、数值workspace缓存；Config沿用原项目using_config的作用域规则。

接受O1作为可选开关，默认False。**不实现O2**：虽然Pooling binding有额外成本，但其不在DonkeyNet中，省略安全检查的诊断下限不足以证明安全POD重构可获得同等收益；新增ABI/workspace复用会增加风险。既有Fused SGD拒绝策略不变。本轮没有测试第二个生产候选，也不把“未实施”描述为“已实验证明无效”。

## 局部性能

GPU 时间来自 Nsight/CUPTI 的实际 kernel activity，15 次/shape/配置；完整同步调用和 CPU 提交来自无 profiler 的独立计时，18 次平衡排列。二者不是同一调用，不能相减解释为“等待时间”，也不能将 CPU、API、GPU 时间相加。submit 包含 Python/C 执行、CUDA API 与可能的队列等待，不等同于纯 Python 主动执行。cProfile 单独采集，self/cumulative elapsed 也不是 CPU 指令周期。

### 完整卷积反向（x/W/b梯度集合相同）

| shape / case | GPU B0/O1/Torch (μs) | 同步完成 B0/O1/Torch (μs) | CPU提交 B0/O1 (μs) | B0/O1 kernel数 |
| --- | --- | --- | --- | --- |
| conv1/backward | 136.577 / 136.576 / 611.650 | 263.863 / 255.526 / 667.489 | 282.869 / 264.994 | 8 / 8 |
| conv2/backward | 170.337 / 170.369 / 189.729 | 255.726 / 252.081 / 248.508 | 268.605 / 262.115 | 8 / 8 |
| conv3/backward | 88.384 / 88.320 / 72.736 | 243.406 / 229.815 / 100.459 | 264.162 / 255.970 | 8 / 8 |
| conv4/backward | 64.417 / 64.512 / 70.944 | 234.230 / 230.615 / 100.747 | 260.236 / 258.056 | 8 / 8 |
| conv5/backward | 50.752 / 50.496 / 52.961 | 231.416 / 221.409 / 100.362 | 262.213 / 245.240 | 7 / 7 |
| irregular/backward | 49.217 / 49.313 / 28.832 | 229.322 / 222.220 / 97.383 | 262.700 / 247.825 | 7 / 7 |
| asymmetric/backward | 44.769 / 45.408 / 50.176 | 190.726 / 181.974 / 93.017 | 209.995 / 199.881 | 5 / 5 |

五层真实卷积跨shape中位数：CPU提交降低 **3.10%**，同步完成降低 **3.16%**；完整调用H/P中位数 **2.306× → 2.206×**；GPU时间比约0.908×，几乎不变。

局部置信区间见 `LOCAL_PAIRED.csv`。conv2、conv4及irregular完整同步调用的CI包含零，不能宣称每个shape都稳定加速。直接调用路径不使用O1，但也出现测量波动，进一步说明必须使用正式训练配对结果。

conv1的PyTorch全梯度局部测量包括输入梯度，cuDNN在此形状较慢。真实训练PyTorch首层输入不requires_grad；HyTorch现有引擎仍计算它。本轮不改变梯度需求，因此不能把上述各层局部时间直接相加预测训练时间。

### Pooling前向

| shape / case | GPU B0/O1/Torch (μs) | 同步完成 B0/O1/Torch (μs) | CPU提交 B0/O1 (μs) | B0/O1 kernel数 |
| --- | --- | --- | --- | --- |
| tiny_pool/forward | 5.312 / 5.280 / 5.632 | 38.659 / 35.691 / 16.203 | 41.441 / 38.826 | 1 / 1 |
| irregular_pool/forward | 5.216 / 5.248 / 5.760 | 38.127 / 34.733 / 16.299 | 42.376 / 38.046 | 1 / 1 |
| overlap_pool/forward | 6.912 / 6.912 / 6.976 | 38.041 / 36.208 / 17.054 | 42.300 / 38.110 | 1 / 1 |
| wide_pool/forward | 5.408 / 5.440 / 5.408 | 38.462 / 35.782 / 16.532 | 42.309 / 38.900 | 1 / 1 |
| tiny_actual_pool/forward | 5.344 / 5.344 / 5.664 | 41.082 / 40.311 / 19.694 | 43.983 / 41.550 | 1 / 1 |

主窗口4个通用shape同步调用中位数降低 **7.32%**；真实TinyNet补测为 **1.88%**，但它的均值配对节省95%CI包含零（−0.347～3.102μs），仅作观察，不能称为稳定前向加速。其CPU提交中位数降低 5.53%。5个shape和全部反向结果都保留在CSV，没有只展示有利数据。

## 正式端到端验收

256合成样本、batch4、3epochs/192steps、seed42、dropout0、SGD lr0.001。12轮，三配置6种排列各执行两遍；同一进程重置相同权重、数据和随机数。包含DataLoader、H2D、forward、backward、SGD及最终同步；初始化/预热和D2H结果检查在时钟外。没有逐步额外同步或profiler。报告吞吐为每轮768/wall的均值。

| 配置 | mean(s) | median(s) | SD(s) | 平均 samples/s |
| --- | --- | --- | --- | --- |
| B0 | 0.764135 | 0.760892 | 0.009725 | 1005.20 |
| O1 | 0.732040 | 0.731914 | 0.004197 | 1049.15 |
| torch | 0.313893 | 0.306004 | 0.014644 | 2451.46 |

`PAIRED.csv`保存每轮差值，`LOSS.csv`保留全部step的loss。95% CI基于12个配对差值、df=11，非独立组标准差比较。新旧最终loss均为0.00025566318072378635，Torch为0.0002556628314778209；全部误差见TRAINING_ERRORS。没有将上一轮F1的百分比与本轮相加。

补充（同主窗口，各3对，仅少量验证）：

| 模型 | batch | B0(s) | O1(s) | wall降低 | 胜出 |
| --- | --- | --- | --- | --- | --- |
| donkey | 1 | 2.863387 | 2.698691 | 5.75% | 3/3 |
| donkey | 16 | 0.204299 | 0.198417 | 2.88% | 3/3 |
| tiny | 4 | 0.288473 | 0.275963 | 4.34% | 3/3 |

batch1为768steps，batch16为48steps，均处理768个样本；TinyNet为192steps。这些不混入12轮主验收CI。

## 时间线与因果范围

以下采用主窗口b/c各10step captures均值；stage_a trace单独保留但不用于该均值。

| 配置 | 阶段 | GPU μs/step | CPU range μs/step | CUDA API μs/step | kernels/step |
| --- | --- | --- | --- | --- | --- |
| B0 | forward | 553.62 | 1290.35 | 167.69 | 38.0 |
| B0 | backward | 653.17 | 2779.33 | 336.18 | 72.0 |
| B0 | update | 113.55 | 407.04 | 104.24 | 32.0 |
| B0 | ALL | 1320.34 | — | 662.55 | 142.0 |
| O1 | forward | 557.28 | 1219.08 | 161.66 | 38.0 |
| O1 | backward | 657.64 | 2616.80 | 318.70 | 72.0 |
| O1 | update | 113.41 | 400.88 | 104.68 | 32.0 |
| O1 | ALL | 1328.34 | — | 638.14 | 142.0 |
| torch | forward | 541.27 | 795.41 | 234.89 | 33.0 |
| torch | backward | 564.92 | 941.96 | 188.89 | 38.0 |
| torch | update | 28.78 | 158.21 | 9.64 | 1.0 |
| torch | ALL | 1134.98 | — | 558.06 | 72.0 |

O1与B0的kernel数均142/step、H2D均2次且633,616 bytes/step；GPU累计时间约1.32ms/step，基本相同。72 backward kernels/step未变。两次新旧capture中完整backward CPU range从约2.78ms降至2.62ms；与正式wall收益方向一致。此处profiler会扰动耗时，不能替代正式训练结果。

GPU kernel时间线空隙（span减kernel并集）B0 3415.67μs/step、O1 3163.97μs/step。这里包含H2D/同步/CPU提交等，是**没有本进程kernel覆盖的时间**，不是设备所有引擎的严格空闲率。Torch kernel存在并发，sum与union也分别记录，不能相加。

CUDA API具体名称、次数、耗时以及launch/sync分项在PROFILE_BEFORE_AFTER与LAYER_COMPARISON内；Python profiling与CUDA API elapsed独立报告。没有通过wall−kernel推导纯Python主动执行时间。

## 内存

| batch | 配置 | 峰值活跃 MiB | 分配请求/完整测量 | 结束活跃 MiB |
| --- | --- | --- | --- | --- |
| 1 | B0 | 5.906 | 89106 | 0.964 |
| 1 | O1 | 5.906 | 89106 | 0.964 |
| 1 | torch | 28.928 | 44545 | 20.143 |
| 4 | B0 | 16.163 | 24210 | 0.964 |
| 4 | O1 | 16.163 | 24210 | 0.964 |
| 4 | torch | 32.357 | 10945 | 20.143 |
| 16 | B0 | 58.242 | 6066 | 0.964 |
| 16 | O1 | 58.242 | 6066 | 0.964 |
| 16 | torch | 55.961 | 2737 | 20.143 |

B0/O1各batch分配请求数、峰值活跃显存及结束活跃显存一致，连续3次没有增长。仅Python容器减少，GPU数组分配和生命周期不变。表内GPU活跃显存是各框架allocator统计，不是nvidia-smi整进程占用；Torch含其库workspace。测量覆盖重置参数、训练及结果采集，不能直接把分配次数/192当训练每步次数。单算子分配次数/请求字节独立在LAYER_COMPARISON中。请求字节是累计分配量，不是唯一临时显存峰值。

## 正确性与限制

- 新增28项专项测试全通过；完整套件319 passed、2 skipped（明确为CPU完整DonkeyNet，GPU版本实际运行）。开关开启后既有291项也全部通过，2个相同CPU skip不计passed。
- memcheck：0 errors；racecheck：0 hazards/0 errors/0 warnings。没有新CUDA算法，仍对经过新包装路径的GPU操作做验证。
- B0/O1所有局部输出和主/补充训练loss、梯度、权重逐位一致；与Torch按原门限对齐，训练最大绝对误差 7.08e-08。不修改任何既有门限；历史GEMM严格门限记录原样保留。
- 本次只验证指定硬件/工作负载窗口；局部微秒级差值不全部显著，主训练12轮CI为正。默认仍关闭，用户可显式启用。
- 主实验与补测跨日是执行窗口限制，真实Pool形状单独报告；不假称全程单窗口。

## PPT最值得展示的三条结论

1. **只增11行生产代码，训练wall减少4.20%，12/12胜出，梯度/权重/loss逐位一致。**
2. **142 kernels/step完全不变，GPU累计时间近似不变；消除单输出冗余包装能减少主机提交间隙。**
3. **训练H/P由2.434×缩小到2.332×，但仍有明显差距；局部kernel接近不等于完整框架接近。**

`figures/`含5套PNG+SVG，图注明计时范围、GPU、batch和样本数；实际Pool补测单独标注。本轮完成后停止，没有继续优化其他算子。
