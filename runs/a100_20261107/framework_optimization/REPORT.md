# HyTorch A100 算子比较与最终框架优化

本轮已完成阶段A测量及阶段B实际源码优化。只保留一个有证据的候选F1：默认关闭的不可变shape校验缓存。没有修改CUDA计算kernel，没有启用被拒绝的融合SGD。

## 正式端到端结果

| 配置 | N | mean(s) | median(s) | 样本SD(s) | samples/s |
|---|---|---|---|---|---|
| B0 | 12 | 0.851833 | 0.851070 | 0.004398 | 901.61 |
| F1 | 12 | 0.786649 | 0.786538 | 0.004661 | 976.32 |
| torch | 12 | 0.312231 | 0.307335 | 0.010986 | 2462.45 |

F1 wall减少 **7.65%**，吞吐加速 **1.0829×**，12/12配对胜出。平均节省 **65.18ms**，95%配对t置信区间 **[61.68, 68.69]ms**。这是本轮B0分母，未引用历史窗口。

本轮B0/PyTorch=2.728×，F1/PyTorch=2.519×，仍有明显差距。只实现F1，没有F2或组合叠加收益；“最终组合版”是前几轮已验证配置加本轮F1。

## 算子对比

| 类别 | shape数 | GPU时间比最小–最大 | GPU比中位数 | 完整同步调用比中位数 |
|---|---|---|---|---|
| ConvForward | 7 | 0.43–2.67 | 1.19 | 2.72 |
| dWeight | 7 | 1.16–3.84 | 1.81 | 1.57 |
| dInput | 7 | 0.08–1.20 | 0.66 | 1.38 |
| dBias | 7 | 0.40–1.74 | 0.74 | 0.61 |
| FullBackward | 7 | 0.22–1.58 | 0.91 | 2.68 |
| GEMM | 11 | 0.75–3.61 | 1.86 | 1.86 |
| Activation | 6 | 0.75–1.41 | 0.88 | 1.68 |
| Pooling | 2 | 0.92–0.95 | 0.93 | 2.94 |
| PoolBackward | 2 | 1.98–2.03 | 2.00 | 2.20 |
| Linear | 4 | 1.08–1.54 | 1.45 | 2.19 |
| SGD | 1 | 4.59–4.59 | 4.59 | 3.96 |

具体形状、误差、每次计时、kernel名/grid/block和allocator计数见配套CSV及OPERATOR_REPORT。类别中位数不等于训练贡献。部分kernel接近PyTorch，而fullcall仍落后，支持从框架提交路径优化。Linear接口差异、首层dInput梯度需求差异和通用shape范围在算子报告明确说明。

## 证据链与修改

真实step记录66次Function调用、38次_shape_ok、5次Conv forward/5次dWeight/5次dInput；1140次/30step重复np.prod检查是可安全消除的CPU工作。新增256项纯shape验证缓存，仍逐调用检查当前dtype/device/buffer，未保存GPU指针或stream。完整conv/packed dWeight/gemm调用15次配对支持减少提交成本，再由上述12轮无profiler训练验收。

训练kernel数仍142/step，im2col仍10/step，SGD仍32/step；并未把Python缓存包装为CUDA算法加速。PyTorch72/step且SGD为1个kernel。GPU空闲间隙与CPU phase elapsed差距比GPU计算差距更大，但未声称精确拆解全部Python在CPU上运行的时间。详细时间线见FRAMEWORK_PROFILE。

## 补充模型/批大小

| 模型 | batch | B0(s) | F1(s) | wall减少 | 胜出 |
|---|---|---|---|---|---|
| donkey | 1 | 3.064449 | 2.811470 | 8.26% | 3/3 |
| donkey | 16 | 0.226399 | 0.209441 | 7.49% | 3/3 |
| tiny | 4 | 0.310554 | 0.291752 | 6.05% | 3/3 |

每项仅3对，支持这些配置内的可重复性，不代表真实数据精度或所有网络通用加速。

## 正确性、内存与保留建议

完整291 passed、2个既有CPU skip，GPU实际执行；memcheck/racecheck零错误。F1/B0的训练loss轨迹、参数和梯度逐元素数值一致。GPU活跃峰值、分配数保持不变，无workspace生命周期变化。建议保留F1并显式启用；默认仍关闭。未修改高阶求导、GEMM、Col2im、dWeight数学计算或SGD。

## 限制与异常完整披露

- 首轮脚本误用SGD.step已改为HyTorch.update，失败日志保留；未产生可用计时。
- 初始单项梯度全部requires_grad的诊断、F1固定居中的pilot训练均完整归档，正式结果未择优混用。
- 一次F1_c短trace里所有未修改kernel同比变快，缺少同期时钟数据，不能证明原因，更不能算作kernel优化收益。正式wall验收不使用profiler。
- GPU外部新任务启动晚于正式训练和所有计时trace，最后allocator history诊断存在约6.7秒可能重叠，只用于本进程计数。见raw/GPU_WINDOW.md。结束后本任务GPU进程已退出，不代表GPU1当前空闲。
- 历史严格GEMM失败记录保留；本轮未改变数值门限。allocator口径不包含所有库私有分配；不宣称显存优化收益。

## 实验方法与适用范围

同一物理GPU1 A100-PCIE-40GB；GPU0未使用。Python3.10.12、NumPy2.2.6、CuPy14.2.0、PyTorch2.9.1+cu128、CuPy CUDA runtime12090、cuDNN91002，动态库nvcc12.4 -O3 -std=c++14，未启用fast-math。TF32同时关闭，PyTorch正常cuDNN/autograd，benchmark=False，不使用torch.compile；SGD foreach=None/fused=None，实际trace为一个multi_tensor kernel。

算子计时：固定FP32输入(seed42，普通算子normal std0.1)、预热10次、15次平衡正反序；单独测Python提交、同步完整调用、CUDA event区间。event区间可能含CPU提交空隙，不当作纯kernel时间。纯kernel来自另一次15重复Nsight/CUPTI捕获，每调用kernel时间求和后取中位数。GEMM双方预分配输出；其它完整接口保留真实分配/packing成本。原始samples和波动均保留。SGD各实现先重置相同参数/梯度验证，再做各自调用测量，foreach与默认明确分列。

普通算子跨框架误差诊断为atol2e-5/rtol2e-4，全部零超限；不能用这个诊断替换原始单元测试门限。原测试完全未修改；历史严格GEMM失败记录保留raw/previous_gemm_*。新缓存与B0比较要求逐元素相等，训练另按既有权重1e-4/1e-3、梯度2e-5/2e-4、loss1e-4/1e-3检查全部元素/轨迹。

训练：256合成样本、batch4、3epochs192steps、seed42、相同初始化、SGD0.001、dropout0。两个框架均按step从同一DataLoader取NumPy数据并H2D，未单边预加载。GPU loss保留至完整训练最终同步，计时结束后读取全部轨迹；未把最终同步挪出区间。正式计时无profiler；配置B0/F1/Torch的六种排列各两遍，共12轮，paired t区间df11。先前F1固定居中的10轮完整保留raw/pilot_fixed_middle_order，不用于正式收益；调整理由是位置平衡，不是择优筛选。

阶段A原始数据及SHA冻结于stage_a。后补的15重复kernel采集和allocator诊断从baseline/source加载未经本轮修改的生产代码，旧数据不覆盖。阶段B正式比较B0为本轮代码开关关闭，F1开关开启，其余相同。GPU计时和profile都在本次连续窗口；没有拼接历史0.820923s/0.307929s作为分母。

Nsight CPU phase归因：同一训练进程串行phase区间按CUDA API correlationId关联kernel，包括PyTorch autograd worker线程；仅按主线程ID会漏掉其backward，已正确跨线程匹配时间区间。GPU累计时间与区间并集分别保存；CPU elapsed包含API、等待、线程调度，不代表纯Python on-CPU。不能把这些相加。没有锁频，也没有与每个trace同步的时钟记录。

## 交付导航

PPT_SUMMARY.md为六页汇报；figures/包含6张PNG及对应SVG。原始trace在traces/，stage_a/及FROZEN_SHA.json冻结原阶段A，raw/保留全部命令/失败/诊断/测试日志。REPRODUCE.md说明如何在新目录执行，不覆盖证据。
