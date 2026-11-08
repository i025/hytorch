# 候选与选择

| 候选 | 证据 | 决策 |
|---|---|---|
| 既有ReLU/Sum | 已实现但尚未真实GPU验收 | GPU优先事项，先验收再谈新Kernel |
| Autograd priority heap | 分支图排序/key占CPU profile约37.65%；CNN很小 | 唯一新原型，显式开关；CPU证明顺序/梯度一致，不能预称CNN加速 |
| Linear/GEMM Backward | 源码明确仍dot/cuBLAS | 需GPU形状与Trace证明差距，暂不重写已有成熟BLAS计算 |
| Bias+Activation/Loss融合 | 历史短Kernel链、MSE额外中间量 | 有可能减少launch/流量，但既有Sum/ReLU尚未验收，不连续堆积未验证Kernel |
| Sum描述符/Workspace/布局 | Python通用轴规划、非连续物化、长归约workspace真实存在 | 本轮测完整成本，后续依据数据决定，不新增自动调优/缓存系统 |
| H2D双缓冲 | 输入线程不传GPU，当前未明确采用pinned staging/异步依赖 | 单batch约0.604MiB（图像FP32），理论规模不是带宽实测；仅准备公平I/O对照 |

Heap不是CUDA Kernel优化。宽图CPU有收益，也可能在短链CNN被heap维护开销抵消，
只有主任务完成且时间充足，才在同一个GPU0窗口末尾进行独立heap正确性和I/list–I/heap匹配实验。
不把这项自动并入J；J只包含已通过对应真实数据检查的既有Sum/ReLU与同样CPU预取。
负结果保留，不根据单次计时调参或更改默认。
