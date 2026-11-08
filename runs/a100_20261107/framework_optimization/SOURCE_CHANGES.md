# 源码改动（本轮增量）

1. hytorch/backends/cuda/runtime.py：ContextVar默认与backend增加metadata_cache=False；新增bool参数校验；_shape_ok分派到旧校验或256项LRU。旧检查主体移至_shape_ok_uncached，_shape_ok_cached只保留成功的不可变整数tuple检查结果。没有修改dtype/device/alias校验。
2. tests/test_framework_metadata.py：新增19项专项参数化测试。原有所有测试文件内容保持不变。

开启示例：

```python
with backend(matmul="shared", conv="im2col", pool="native",
             col2im="native", dweight="packed", metadata_cache=True):
    # 原训练代码，SGD(lr=0.001, backend="legacy")
    ...
```

建议保留这项小范围实现，在已验证工作负载显式开启；全局默认仍False。未实现第二候选。无CUDA源码/动态库修改，不需要重新编译；未修改自动求导、优化器、数据加载或workspace数值生命周期。

git_diff.patch为相对于本轮baseline/source的增量，baseline/working_tree.patch另保存此前全部未提交修改，勿混淆。诊断脚本位于diagnostics，性能数据不纳入生产代码。没有Commit、Push或PR。
