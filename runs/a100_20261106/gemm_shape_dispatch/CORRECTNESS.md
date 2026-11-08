# 正确性与数值门限

## 基线

先恢复并核验保存的旧 dispatch/library，最终 dtype guard源码不改：56项 GPU 测试通过（45原CUDA + 11 dtype/完整DonkeyNet/整数输入），0 skipped；22原主机 + 9 dtype主机通过。

第一次合并测试曾 48 passed / 8 failed：native CUDA库先装载 CUDA12.4 runtime，之后导入PyTorch遇到 cudaGetDriverEntryPointByVersion 未定义符号。它是装载顺序错误，未进入这些测试的数值逻辑。随后测试进程先 import torch，采用一致runtime，56项通过。原失败日志和重试完整argv保留，不隐藏失败。

准备阶段一次编译在收到失败结果前已启动；随后旧源码和旧库恢复，成功基线确认后才应用并重新构建本轮dispatch。以 raw/baseline_retry_commands.json 后的成功构建/测试作为验收证据。

## 新版本

- 102 项实际 GPU 测试通过，无 skipped（原45 + dtype11 + 新dispatch46）。
- 22项原主机 + 9项 dtype 主机通过。
- 42组共享GEMM边界测试记录真实 CUDA Graph kernel；基本/blocked另4项直接检查未被重定向。
- 47组shape×2tiles×4实现=376行性能/数值记录；每组12次配对；全部old/new/forced_shared输出逐位相同。
- 所有实现相对FP64参考的L2误差最大 1.99003092e-06，仅为补充误差描述。
- 7轮、两实现训练最终loss和全部权重逐位相同；参数和梯度实测FP32。FP32生产源码完整SHA不变。

## 原严格门限：仍有失败，未降低标准

NumPy FP32参考、atol=rtol=1e-4 原门限原样记录。下列22行未通过；old/new/forced_shared失败集合相同，不是新dispatch引入。2048/4096时cuBLAS也失败，不能据此删除原门限。

| M,N,K | tile | 实现 | FP64 相对 L2 |
|---|---|---|---|
| 24,75,12152 | 16 | old | 1.99e-06 |
| 24,75,12152 | 16 | new | 1.99e-06 |
| 24,75,12152 | 16 | forced_shared | 1.99e-06 |
| 24,75,12152 | 32 | old | 1.99e-06 |
| 24,75,12152 | 32 | new | 1.99e-06 |
| 24,75,12152 | 32 | forced_shared | 1.99e-06 |
| 2048,2048,2048 | 16 | old | 8.11e-07 |
| 2048,2048,2048 | 16 | new | 8.11e-07 |
| 2048,2048,2048 | 16 | forced_shared | 8.11e-07 |
| 2048,2048,2048 | 16 | cublas | 8.11e-07 |
| 2048,2048,2048 | 32 | old | 8.11e-07 |
| 2048,2048,2048 | 32 | new | 8.11e-07 |
| 2048,2048,2048 | 32 | forced_shared | 8.11e-07 |
| 2048,2048,2048 | 32 | cublas | 8.11e-07 |
| 4096,4096,4096 | 16 | old | 1.15e-06 |
| 4096,4096,4096 | 16 | new | 1.15e-06 |
| 4096,4096,4096 | 16 | forced_shared | 1.15e-06 |
| 4096,4096,4096 | 16 | cublas | 1.15e-06 |
| 4096,4096,4096 | 32 | old | 1.15e-06 |
| 4096,4096,4096 | 32 | new | 1.15e-06 |
| 4096,4096,4096 | 32 | forced_shared | 1.15e-06 |
| 4096,4096,4096 | 32 | cublas | 1.15e-06 |


此前轮次的严格门限记录保留在 raw/previous_*。不同固定seed可改变接近零元素的相对误差门限是否触发，所以不能通过更换seed掩盖失败。本次seed=9000+shape_id事先固定，对四实现完全相同。逐位相同和相对L2指标不能替代原严格gate通过；报告不声称“所有GEMM严格数值检查通过”。

## 实际选择证据

DISPATCH_DECISIONS.csv 对应每种实现/shape/tile的图节点kernel、独立CUPTI kernel名字、实际grid/block/CTA/register/shared；native图选择与独立CUPTI完全匹配。非默认stream、预分配out和连续调用在测试中执行。未仅用Python计算的predicate冒充实际dispatch。
