# 选择 dInput gather 融合

本轮仅实现 **dInput 的 Tensordot + 布局处理 + Col2im 融合**。不改 dWeight，不引入第二个候选 Kernel。

## 直接历史证据

来源为阶段 4C 的 **D 配置、16 正式 Step**，不是本轮新测量。每个 GPU 事件已由 CUDA correlationId → 提交 API → NVTX 归因。重新计算的 [逐层 CSV](results/history/historical-costs.csv) 与 [汇总](results/history/summary.json) 保留。

| 层 | dInput 核时 μs/step | dWeight 核时 μs/step | dInput Kernel/step | 其中 Col2im μs | gcol 字节（形状推导） |
|---|---:|---:|---:|---:|---:|
| Conv2 | 125.572 | 131.042 | 3 | 94.826 | 6,316,800 |
| Conv3 | 68.090 | 60.140 | 3 | 46.530 | 1,584,000 |
| Conv4 | 33.310 | 38.108 | 3 | 16.352 | 414,720 |
| Conv5 | 31.776 | 28.520 | 3 | 16.416 | 165,888 |
| 合计 | **258.747** | **257.810** | **12** | **174.124** | **8,481,408** |

四层 dInput Copy 共 18.804 μs/step；Tensordot 分类已包含 Copy，不能重复相加。Conv1 dInput 在 4C 已被剪掉。dWeight 还包含 Conv1 的约 84.592 μs，所以五层总 dWeight 更大；这里没有把它遗漏后宣称 dInput 是整个网络最大热点。

## 选择依据和成本

- 两条路径在 Conv2–Conv5 的 GPU 核时相近，但 dInput 有明确的完整 gcol 写入/读取及独立 Col2im；每层旧路径为 Copy + GEMM + Col2im，适合一次 gather 取代。
- 一个线程负责一个 NCHW dInput，输出只写一次；没有线程间归约或全局 atomicAdd。可保持权重/偏置梯度与训练公式不变。
- 该形状集合法乘加约 **85,148,928 FMA/step**。融合不减少必要数学乘加，主要目标是移除中间张量与提交。未使用缓存复用时，两个操作数的逻辑读取量约 **681,191,424 字节/step**；这不是实际 DRAM 流量，缓存会影响真实读取。
- 相邻线程沿 W 轴映射，Stride 1 的 gy 访问连续；Stride 2 有按奇偶分支的活跃线程，活跃地址仍具局部性。权重在相同 c/kh/kw 的线程间可经缓存复用；gy 在不同输入通道线程间重复读取。当前原型不使用 shared memory 分块。
- 代价是每线程循环 OC 和 KH×KW、除法/余数、边界分支及重复读取；不能因为少 Launch 就假定超过 cuBLAS。Conv2 的计算和读取压力最大，必须实测。
- dWeight 需要跨 N/OH/OW 的大归约与数据复用，替代已有 GEMM 要引入另一套归约设计。本轮选择更小、可独立验证的 dInput 改动。

## 可验证目标，尚非实测结果

若每层成功路由，新路径每层提交 1 个 Kernel，四层预期从 12→4，整个 D→F 训练 Step 理论从 114→106。新 runtime 源码仅分配 gx，不创建 gcol、打包输入或布局 Copy。必须用新 Nsight 和分配统计核实这些预测；目前 GPU 无空闲设备，全部 GPU 比较为 NOT_RUN。

历史 258.747 μs 是 GPU Kernel 时长求和，不是可直接从训练墙钟扣除的时间。4C 已表明 GPU 工作减少不一定带来稳定吞吐收益，因此不预报端到端加速比。

![历史证据与形状推导](figures/historical-design-evidence.png)
