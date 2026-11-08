# 正确性证据

| 检查 | 结果 | 原始日志 |
|---|---|---|
| 首次CPU定向 | 30pass/2fail | raw/host_candidates.log |
| CPU修正后定向 | 32pass、17 deselected | raw/host_candidates_retry.log |
| GPU候选专项（含CPU调度测试） | 49pass、无skip | raw/gpu_candidates.log |
| 完整回归，HYTORCH_REQUIRE_CUDA=1 | 402pass、2已有CPUskip | raw/full_tests.log |
| Sanitizer memcheck | 26pass、0 errors | raw/memcheck.log |
| Sanitizer racecheck | 26pass、0 errors、0 warnings | raw/racecheck.log |
| 优化CLI完整DonkeyNet | 3epochs/192steps完成 | raw/cli_optimized.log |
| 同进程训练loss/参数/梯度 | 2538条逐位相同 | NUMERICAL_ALIGNMENT.csv |
| 原版精确源码对照 | 10对三类SHA全部一致 | ORIGINAL_COMPARISON.csv |

首次失败为测试适配错误：来源SGD参数与当前backend枚举不同；以及测试对既有pow高阶标量路径作了错误FP32假设。后者改为检查原路径和融合路径的相同dtype/数值；没有修改生产pow，也没有降低allclose门限。失败日志保留。

移植后的参数/输入/偏置梯度与PyTorch比较沿用来源atol1e-5/rtol1e-4或更严门限，其他已有测试不改。ReLU有限数逐位一致、NaN一致、signed zero单独检查；FP64/CPU/empty/mixed dtype回退；transpose/负stride/广播、非默认stream、重复调用、None梯度、leaf、双层conv、权重高阶导数、非默认stream宽GPU图均覆盖。

初始/usr/bin/compute-sanitizer启动失败为缺失libsanitizer-collection.so，未执行kernel检查；保存在raw/memcheck_initial_environment_failure.log。改用现有完整CUDA Toolkit工具后真实执行通过；不将初次失败当通过或静默删除。

GPU专项后又加入6个宽图GPU heap案例，最终全量回归与两种Sanitizer均包含它们。完整回归中的pruning参数测试默认CPU；GPU专项显式HYTORCH_PRUNING_DEVICE=cuda，不用CPU通过替代GPU通过。两项既有CPUskip不计为GPU测试通过。
