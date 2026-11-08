# F1：有界 shape 校验结果缓存

唯一生产修改位于hytorch/backends/cuda/runtime.py：backend新增metadata_cache=False；_shape_ok根据开关仅对精确tuple且每个维度精确Python int使用256项LRU；miss调用原校验。np.integer、float、bool、list等走原路径，避免相等键使不同类型绕过检查。失败不缓存。

收益依据：38次/step重复纯shape校验、主要np.prod成本；整条conv/packed dWeight/gemm调用配对与正式训练验证。实现只保存不可变shape值及None成功结果，既不保存数值，也不保存GPU地址、array、stream、device。shape变化自然变成新key，容量满时逐出。shape合法性与设备无关，因此复用校验结果无需按device分key；每次实际buffer/dtype/device/alias检查完全保留。

风险控制：不改np.int64乘积的原判定、不改归约顺序、自动求导、Col2im、GEMM、SGD、workspace生命周期或任何CUDA ABI/kernel。原有shape规则本身没有扩大修复范围。默认关闭。仅对验证已证明纯函数的_shape_ok缓存，不缓存_geometry、dtype或设备结果。

候选排序：1)重复shape验证：38/step、可消除、局部低风险，本轮实现；2)Function/Variable开销：广泛但生命周期和高阶梯度风险大，不改；3)排序：小，收益上限低，不改；4)SGD融合：已有FTZ准入失败，不启用；5)数据H2D：两边2次/step相同且没有冗余D2H，不改变。

没有F2，因此最终版就是F1；不伪造组合消融。CPU tracemalloc在256/5000次唯一shape后约52KB/69.5KB，LRU始终最多256项（包括上下文与Python分配，非GPU内存）。GPU活跃峰值与分配次数不变。
