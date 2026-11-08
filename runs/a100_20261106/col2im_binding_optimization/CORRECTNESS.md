# 正确性与安全性

| 验证 | 结果 | 原始记录 |
|---|---|---|
| 全部原主机测试及metadata新增测试 | 24 passed、4 deselected GPU项目 | raw/host.log |
| host dtype | 9 passed、13 deselected GPU项目 | raw/dtype_host.log |
| GPU完整回归 | 204 passed、12 deselected CPU项目、0 skipped | raw/gpu_tests.log |
| Compute Sanitizer memcheck | 103 passed，ERROR SUMMARY 0 | raw/memcheck.log |
| Compute Sanitizer racecheck | 103 passed，0 hazards、0 errors、0 warnings | raw/racecheck.log |
| 新binding首次验证 | 5 passed | raw/binding_first.log |
| 36个单算子case三方 | 最大误差0，逐位一致 | COL2IM_COMPARISON.csv |
| 10轮×3完整训练 | loss/梯度/权重逐位一致，最大误差0 | TRAINING_COMPARISON.csv |

GPU强制HYTORCH_REQUIRE_CUDA=1，不能把unavailable转换为skip。现有test_col2im_cuda用独立CPU scatter与PyTorch Fold进行参考验证；重叠累加保持kh/kw顺序，与CuPy逐位相等。PyTorch及高阶梯度原门限rtol2e-6/atol2e-7未改变。

新增test_col2im_binding检查：144-byte POD整数布局、LRU容量128、重复metadata、shape变化、负/零strides变化、输入输出地址变化、非默认stream、热缓存后alias/dtype/布局/device/非法整数参数仍检查、旧ABI可选符号回退、空params返回错误。

设备错误检查采用mock当前device ID和测试专用错误device ndarray subclass，不碰GPU0其他用户任务；这不是实际多GPU buffer集成测试。所有原正/负/零/非单位stride、非方形kernel、非对称stride/pad、无贡献输出及显式FP64 fallback继续运行。cache value只有tuple与bytes，没有GPU地址、数组、stream或device。

原严格GEMM长K/2048/4096失败记录位于前轮gemm_shape_dispatch与fp32_fix_and_profile，未改变门限或声称解决；本轮仅执行既有GPU回归，未重复其大型诊断基准。原始失败证据另附raw/previous_gemm_paired_comparison.csv、previous_gemm_baseline_summary.json、previous_gemm_correctness.md。

失败记录：raw/bench.log为diagnostic Graph在legacy默认stream捕获失败；修正脚本后raw/bench_corrected.log通过。没有生产正确性失败或Sanitizer错误。检查raw/scope_checks.json确认旧CUDA源码为新文件逐字节前缀、functions_conv与FP32 functions.py未变。
