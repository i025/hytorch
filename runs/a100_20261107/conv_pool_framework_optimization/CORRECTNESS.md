# 正确性验收

| 阶段 | 实际结果 | 原始日志 |
|---|---|---|
| 修改前完整host+GPU | 291 passed, 2 skipped | raw/tests.log |
| 新增专项 | 28 passed | raw/special.log |
| 修改后完整host+GPU | 319 passed, 2 skipped | raw/tests_after.log |
| O1启用后的全部既有测试 | 291 passed, 2 skipped | raw/tests_on.log |
| Compute Sanitizer memcheck | 28 passed, 0 errors | raw/memcheck.log |
| Compute Sanitizer racecheck | 28 passed, 0 hazards/errors/warnings | raw/racecheck.log |

两个skip均为test_backward_dtype里CPU完整DonkeyNet，GPU版实际执行；skipped未计为通过。O1专项在每项内分别切换B0/O1；tests_on对原有整套测试全局开启O1，单独排除依赖默认False的专项文件（它已单独全部运行）。

新增测试覆盖CPU/GPU×FP32/FP64，batch1/3/2/1连续变化，NCHW/转置/负/零stride，非方形conv/stride/padding，梯度累积/重复backward/create_graph/二阶，输入在建图后原地修改、no_grad、更新后参数；pool负输入/边界padding/重叠窗口/高阶；非默认stream、弱引用生命周期、原alias/dtype拒绝；自定义Function的ndarray/0d/scalar/None/tuple/list异常返回契约和配置异常恢复。没有改动既有测试门限。

测试bitwise以dtype/shape/tobytes比较，包含符号零等；数学路径相同，误差应为0。局部18轮数据的B0/O1、主训练12×192步的loss/16参数及其梯度、补充训练全部逐位相等。Torch局部和训练所有已有atol/rtol门限均通过；训练最大绝对差7.07805157e-08。完整误差和逐步loss均保留CSV。

原生kernel/ABI未改变；GPU压力与安全测试仍确实在A100执行。之前GEMM严格门限失败记录复制到raw/previous_gemm_*，只是历史证据，不改门限、不把旧失败改记为本轮通过。

默认Config为class级配置，沿用原项目using_config机制；本轮没有扩展为线程局部配置，也没有新增跨线程共享GPU缓存。
