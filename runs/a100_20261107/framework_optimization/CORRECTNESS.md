# 正确性与安全验证

- 阶段A生产基线完整回归：272 passed、2 skipped；原有两个CPU DonkeyNet测试显式skip，对应GPU测试实际运行，HYTORCH_REQUIRE_CUDA=1。
- 新专项19 passed；覆盖FP32/FP64、CPU/GPU、非连续/负stride视图、不规则非方形卷积、batch动态变化、重复backward与累积、no_grad/create_graph/二阶梯度、输入原地修改后新forward、SGD hooks、stream、预分配输出、dtype拒绝、输出重叠拒绝、对象释放、cache容量/键类型/异常恢复。
- 修改后完整回归：291 passed、2个相同CPU skip；没有GPU不可用skip，没有降低门限。
- Compute Sanitizer memcheck：19 passed、0 errors；racecheck：19 passed、0 hazards。没有新增CUDA ABI或GPU内存路径，但仍测试了候选调用现有kernel的安全性。
- 15个完整调用路径B0/F1逐元素相等；36次正式训练的F1/B0全部loss轨迹、梯度和权重最大绝对误差0；所有supplement同样0。这里描述数值逐元素一致，未把assert_array_equal当作signed-zero位模式检查。
- PyTorch训练对齐全部通过，正式训练所有误差最大绝对值4.47e-8。算子跨框架诊断全部0超限；原始严格GEMM历史失败另保留，不能宣称已经修复。

原生产kernels.cu与动态库SHA完全不变，旧FP32修复和128CTA dispatch保持。旧融合SGD仍不可用且未启用；所有正式训练使用legacy SGD。

| batch | 模式 | GPU活跃峰值MiB | malloc请求/768samples | clear后bytes |
|---|---|---|---|---|
| 1 | B0 | 5.906 | 89106 | 1011200 |
| 1 | F1 | 5.906 | 89106 | 1011200 |
| 4 | B0 | 16.163 | 24210 | 1011200 |
| 4 | F1 | 16.163 | 24210 | 1011200 |
| 16 | B0 | 58.242 | 6066 | 1011200 |
| 16 | F1 | 58.242 | 6066 | 1011200 |

显存为CuPy MemoryHook活跃池峰值（非整卡峰值），不含Torch或CUDA库私有分配。reserved存在池碎片/缓存差异，全部原始数值保留。每组3轮cleargrads/gc/sync后活跃1011200bytes稳定，没有观测到逐轮增长。
