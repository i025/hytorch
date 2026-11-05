# 通用算子开发路线图

排序是基于源码覆盖和历史 Trace 的**研究优先级**，不是已测收益排序。本轮只实现 Sum。

| 优先级 | 候选 | 手段和可能减少的工作 | 跨任务复用 | 下一步证据 |
|---|---|---|---|---|
| P0 | 已有 ReLU Backward + 本轮 Sum 验收 | ReLU 掩码+乘法到一个 Kernel；Sum 用 block/warp 和分段归约 | CNN/MLP/Transformer | 首先处理本轮消去负样本，跑数值、非默认流与完整算子/训练；不直接启用默认 |
| P1 | 逐元素链/损失融合 | MSE 差值+平方+归约；Tanh 梯度；bias+激活 epilogue | CNN/MLP/Transformer | 新 Trace 区分主机调度和核时；每个被消除的 FP32 全尺寸中间量可省约4N存储、一次写+后读约8N字节 |
| P1 | LayerNorm/Softmax | 行归约、warp/block、shared；复用归约思想但需要独立数值稳定算法 | Transformer、MLP、部分 CNN | 对照 last-axis/跨行、短/长行与消去；避免 softmax exp 溢出；统计实际中间量和 launch |
| P2 | GEMM epilogue/布局 | 已有 tiled GEMM 的 bias/activation、transpose 消除或 BMM | MLP/Transformer/CNN FC | Forward/Backward 分开；cuBLAS 已很强，不能因代码不是原生就假定值得替换 |
| P2 | Conv Workspace/调度 | 已有 dInput Gather 与 Split-K dWeight 的 workspace、归约、形状成本 | CNN | 当前已有原型，先做通用匹配测量；Im2col 的4·N·Cin·Kh·Kw·Oh·Ow 字节是理论规模，不是实测流量 |
| P2 | Pooling Backward | argmax gather/scatter 与 Col2im 合并，减少 gcol/Copy | 使用 Pool 的 CNN | DonkeyNet/SpatialNet 不含 Pool；先找真实工作负载再决定，不能以其端到端成绩验收 |
| P3 | AdamW/裁剪/跨 Tensor 归约 | multi-tensor 融合、少量 block 归约减少 launch | 多种训练任务 | SGD 已融合，不重复实现；需新优化器工作负载和状态/高阶语义验证 |
| P3 | Attention/Embedding 等 | tiled BMM、注意力融合、索引梯度 | Transformer | 先建立模型正确性/基线，不能直接把现有二维 GEMM 当通用 Attention 实现 |

推理更看重 Forward 的 epilogue/Softmax/Norm；训练还包含广播梯度归约、激活梯度与更新。
SumTo 不是独立数学新算子，复用通用 Sum 比单独写 bias-gradient 固定形状分支更合适。

所有流量/launch 减少均为算法估计。CuPy 可能已有 CUB、融合或视图优化，Python 表达式数不等于
Kernel 数。未来顺序：数值通过 → 单算子完整成本 → 新 Trace → 匹配未插桩端到端，
只有收益超过三次重复的波动才讨论保留和调度；性能退化也保留原始数据。
