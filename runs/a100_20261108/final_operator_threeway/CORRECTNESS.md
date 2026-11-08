# 正确性与口径说明

- 当前生产测试：`pytest -q tests`，**319 passed，2 skipped**（raw/current_tests.log）。两项 skip 为 CPU 全量 DonkeyNet 重复项，定义见 tests/test_backward_dtype.py；GPU DonkeyNet 回归确实执行，设置 HYTORCH_REQUIRE_CUDA=1，未将 GPU skip 当作通过。
- 窗口前主机专项：23 passed；12 组独立 CPU 窗口展开/折叠几何检查通过。
- 本轮诊断：804 组输出与独立 CPU FP64 参考比较，失败元素总数 **0**。包括全部 GEMM 到 **4096³ 的完整 FP64 乘法**，并非抽样；两种接口、冻结 Shared/Register 也逐项验证。
- benchmark 诊断使用 `abs(error) <= 5e-5 + 2e-5*abs(reference)`。这是新增诊断，不替换原测试的各自严格门限。GEMM 输入标准差 0.25、FP32，精度结论只覆盖本轮输入，不能推出所有数据分布均通过。
- RelativeL2 分母 `max(norm(reference),1e-30)`；最大逐元素相对误差分母 `max(abs(reference),1e-8)`，没有对近零直接除零。所有原始值见 ACCURACY.csv。
- Im2col 使用独立 NumPy sliding_window_view 的 FP64 排列参考；Conv/dWeight 用 FP64 NumPy BLAS；Col2im 用独立 PyTorch CPU FP64 fold，对非对称 padding 采用显式裁剪；Pool 用零 padding、kh/kw 顺序第一个最大值、独立 fold 梯度。
- Pooling 包含全负输入、零 padding 与相等最大值；四例 Native index 与独立参考完全相同（raw/POOL_INDEX_CHECKS.csv）。不把 PyTorch 默认负无穷 padding 的池化直接当作等价参考。
- 五个真实 Col2im 的 shape/dtype/byte strides 与一次新训练采集逐项 assert 相符；36 个 case 覆盖 training-strided、连续、负 strides。
- 无本轮 UNSUPPORTED case。旧 Register 库对于 M/N/K<64 会实际回退 Shared；DISPATCH_DECISIONS.csv 给实际 kernel，不把回退称为 Register。

| 算子 | Native MaxAbs | Native RelativeL2 | 混合容差失败元素 |
|---|---|---|---|
| GEMM | 0.000133982 | 2.08534e-06 | 0 |
| Im2col | 0 | 0 | 0 |
| Col2im | 9.80217e-07 | 9.24238e-08 | 0 |
| Conv forward | 8.76938e-06 | 5.3811e-07 | 0 |
| dWeight（混合） | 6.75029e-06 | 3.33045e-07 | 0 |
| MaxPool forward | 0 | 0 | 0 |
| MaxPool backward（混合） | 1.82539e-07 | 4.76533e-08 | 0 |


完整 API 差异：NumPy Conv/dWeight 为合理的 NumPy BLAS 评测适配器，避免将生产 CPU 的自研 C GEMM 误标成 NumPy；原生单独 Im2col 无公开 Function，使用保留安全检查的诊断 ABI adapter + 原 Function.__call__ 包装。其余 Function 使用生产类；dWeight 和 Pool backward 最终数据已补测原始构造器。所有 Function 测试保持 enable_backprop=True / O1=True，仅计该算子调用，不执行额外 backward。

历史严格失败没有被“修复”或隐藏：raw/historical_strict_gate/ 原样保存前轮长 K、2048/4096 GEMM 的失败证据，仅作为验收历史，**不参与本轮性能比值**。当前测试与随机输入通过不能覆盖或撤销这些历史失败。

执行中保留的诊断失败：首次无 tests/ 参数的 pytest 误收集多个历史 source/tests，120 个模块同名 collection errors，raw/full_tests.log 完整保留；指定当前 tests/ 后通过。workload.py 首次在写 WORKLOAD.csv 时出现重复关键字（真实采集和 shape/stride assert 已完成）；CPU 修复写表不影响数据。extra.py 首次缺少 threadpoolctl，改为 ctypes 查询已加载 OpenBLAS 的线程数，重跑成功；没有安装软件。首次简化构造器数据保留 raw/formal_firstpass/，最终 dWeight/Pool backward 使用 raw/interface_corrected_*。
