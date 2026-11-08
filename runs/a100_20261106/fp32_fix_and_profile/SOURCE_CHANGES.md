# 本轮源码修改与审计

只修改生产文件 `hytorch/functions.py`，新增回归测试 `tests/test_backward_dtype.py`。未修改 CUDA kernel、runtime、全局 `as_array`、梯度累积或 SGD 算法。

## Tanh.backward（functions.py:53）

之前：
```python
y = self.outputs[0]()
gx = gy * (1 - y*y)
```
之后：
```python
y = self.outputs[0]()
xp = cuda.get_array_module(y.data)
one = xp.array(1, dtype=y.dtype)
gx = gy * (one - y*y)
```
`Variable.__rsub__` 将 Python int 变为 CuPy int64 零维数组，参与 FP32 数组运算时提升为 FP64。内部单位常量跟随 tanh 输出 dtype，保持计算图和显式 FP64 行为。

## MeanSquareError.backward（functions.py:478）

之前：
```python
diff = x0 - x1
gx0 = gy * diff * (2. / len(diff))
```
之后：
```python
diff = x0 - x1
xp = cuda.get_array_module(diff.data)
scale = xp.array(2. / len(diff), dtype=xp.result_type(diff.dtype, gy.dtype))
gx0 = gy * diff * scale
```
内部缩放常量匹配 diff 与上游梯度的联合 dtype。FP32/FP32 保持 FP32，FP64/FP64 保持 FP64；显式 FP64 上游梯度不被缩窄。整数 diff 的浮点上游梯度也能保留浮点比例，避免把 2/3 截断为 0。未给所有梯度加后置 cast。

GPU 窗口测试及性能数据采用 `dtype=diff.dtype` 版本。窗口结束后 CPU 边界审查发现整数输入可能截断缩放系数，最终使用 `result_type(diff.dtype, gy.dtype)`。已用 CuPy 的纯 dtype API 证明本次实测所有同质 FP32/FP64 路径的系数 dtype 完全相同，并重新通过 9 项 CPU 回归；没有重新申请 GPU 或冒充最终字节版本重跑 GPU。保存 `raw/functions_gpu_tested.py`、`raw/final_metadata_guard.patch`、`raw/final_scale_dtype_metadata.json`。最终整数边界路径仅经 CPU 验证。

## 回归测试

覆盖 CPU/CUDA 的 FP32、FP64 MSE 与 Tanh 一阶/二阶导数；PyTorch 数值参考；梯度累加；显式 FP64 数组提升；Conv2D/Linear dInput/dWeight/dBias；SGD 中间表达式；完整 DonkeyNet 的 native 和 CuPy 前向、反向、保留计算图 dtype、更新后参数对齐；另有整数输入 MSE CPU 边界测试。

`raw/this_task.patch` 是本轮两文件 patch；`raw/working_tree_full.patch` 含开始前已有未提交修改，不能将其都归为本轮。`raw/before.patch` 和 `before_status.txt` 保存开始状态。已有 README、kernels.cu、test_cuda_device.py、bench_cuda_revisions.py 和 docs 修改均保留。

诊断脚本与输出在 ignored runs 目录。`legacy.py` 从保存的原源码 AST 提取两个旧 backward 方法，仅在实验进程内临时替换；退出恢复，不覆盖工作区。没有 Commit、Push 或 PR。
