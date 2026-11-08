# 框架热点与训练时间线

| trace | kernel/step | GPU累计us/step | 活动并集us/step | kernel跨度内间隙us/step |
|---|---|---|---|---|
| train_B0_a | 142.0 | 1312.90 | 1312.90 | 3585.32 |
| train_B0_b | 142.0 | 1329.61 | 1329.61 | 3645.36 |
| train_B0_c | 142.0 | 1328.63 | 1328.63 | 3583.73 |
| train_F1_b | 142.0 | 1328.56 | 1328.56 | 3126.21 |
| train_F1_c | 142.0 | 731.22 | 731.22 | 3919.63 |
| train_torch_a | 72.0 | 1130.83 | 1065.01 | 1026.93 |
| train_torch_b | 72.0 | 1134.85 | 1067.29 | 1021.42 |

阶段A实际训练142 vs72 kernel/step。HyTorch forward/backward/SGD为38/72/32，PyTorch33/38/1。两者均每step两次H2D，共633616bytes，计时捕获内无逐step D2H loss读取。没有找到值得删除的训练同步点；本轮不改变数据传输。

冻结trace中HyTorch GPU活动并集1.313ms/step、PyTorch1.065ms/step；kernel跨度内间隙3.585ms vs1.027ms。跨度差约2.806ms中，约0.248ms来自活动并集差、约2.558ms来自无kernel活动区间差。这是对短trace自身跨度的算术解释（约91%来自间隙），不是把profiling数字强行相加解释无profiler训练差距。间隙还可能包含拷贝、CPU等待、调度，不能全部称为纯Python。

CPU cProfile的30step：_shape_ok1140调用（38/step）、累计约13ms，np.prod约9ms；F1热缓存仍1140次_shape_ok，但不再执行该路径的np.prod。Function.__call__1980次（66/step）、backward30次，未改变；sort630次仅约0.41ms累计，不是首选。排序/对象与自动求导大改风险高，因此没有为了任务强行引入第二优化。CPU_FUNCTIONS.csv提供精确计数与累计/自身耗时。

F1不改变kernel数、数学工作或GPU分配数。B0_b/F1_b trace累计GPU时间约1.330/1.329ms/step；CPU提交间隙下降。F1_c所有未修改kernel几乎同比缩短，且无对应时钟日志，根因未确认（动态时钟为可能解释）。所有trace保留；不能将该GPU时间下降视为缓存修改了算法，也不依据它声称GPU加速。

当前约2.73倍训练差距主要体现为更高提交/调度成本、更多小kernel与SGD launch，以及部分原生GEMM性能差距；只有缓存候选的wall收益通过本轮受控消融建立了因果证据。没有完整证明每种剩余因素对2.73倍的精确贡献。
