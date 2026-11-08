# 本轮源码变更

- runtime.py：backend新增dweight=cupy/packed，默认cupy；packed_dweight按当前x/gy新生成列矩阵，FP32/device/NCHW/shape/int32范围检查及fallback；可选ABI hy_sgd_multi；私有SGD描述符/别名检查原型。公开sgd_multi返回False，因为FTZ精度门限拒绝。
- functions_conv.py：仅Conv2DGradW.forward增加packed选择与回退；backward修复为使用输入gys，上游梯度公式同时适用于B0/D/S/DS。
- optimizers.py：仅SGD新增backend=legacy/native，默认legacy。精确SGD类型才尝试批量路径；收集/hooks/再次过滤顺序保留，当前native回退原路径。其他optimizer不改。
- kernels.cu：只在原文件后追加SgdTensor/SgdBatch/hy_sgd_multi/sgd_multi_kernel；原有CUDA源内容逐字节为前缀，GEMM/col2im等未改。原ABI版本与入口不变。
- tests/test_final_optimization.py：35项新增专项；包含真实higher derivative、layout、fallback和SGD语义。SGD prototype最初失败测试副本保留raw/prototype_tests.py，生产fallback最终通过不掩盖原型拒绝。

没有forward/backward数值workspace缓存、requires_grad、version counter、完整训练CUDA Graph、混合精度、TF32、fast-math或FTZ开关。dWeight GEMM依然来自CuPy/cuBLAS，不是新手写GEMM。SGD原型是新手写多张量kernel，但未通过全部数值准入。

git_diff.patch只含本轮变更，baseline/git_diff.patch包含进入任务时已存在的修改。raw/newline_cleanup.json记录CRLF清理的AST一致性；几何范围保护后已重测受影响性能，旧结果封存pre_range_guard。
