# 阶段 4C 源码与梯度需求

基线为阶段 4B 提交 `5476cb65f3b0493e8edc7a495551c9cae25587fb`。本轮生产改动限定在 `hytorch/core.py`、`hytorch/functions_conv.py` 和训练 benchmark 的显式开关；模型、优化器、自研 CUDA Kernel 均保持基线实现。

## 最小接口

```python
x = Variable(batch, requires_grad=False)
loss = F.mean_square_error(model(x), target)
model.cleargrads()
loss.backward()
```

[`Variable`](../../../hytorch/core.py) 增加默认 `requires_grad=True` 的关键字参数。其语义是**该变量是否接收梯度并向其 creator 继续传播**。明确为 False 时，反向调度跳过该变量的梯度累积和上游调度。`Conv2d.backward` 会进一步在计算前检查它，避免产生无用 dInput。

这不是 PyTorch 的完整 requires_grad 推导机制：Function 输出仍按原来方式创建默认 True 的 Variable，不推断“所有输入都不需要梯度”，不改变建图规则、不自动冻结参数、不清除调用前已存在的梯度，也不尝试为所有算子剪枝。需要输入梯度时使用默认值或明确 True；不能根据 `creator is None` 判断叶子变量不需要梯度。`backward()` 的直接调用输出明确关闭梯度时报告错误；叶子输出没有 creator 时正常结束调度。

## 实际计算链与剪枝范围

[`DonkeyNet.forward`](../../../donkeycar_hy/model.py) 为五次 Conv2d+ReLU，随后 Flatten、三次 Linear（前两层 ReLU/Dropout，最终 Tanh）。本实验 dropout=0，MSE Loss，SGD。batch4 的卷积输出形状依次为 `[4,24,31,98]`、`[4,36,14,47]`、`[4,48,5,22]`、`[4,64,3,20]`、`[4,64,1,18]`。

原 [`Conv2d.backward`](../../../hytorch/functions_conv.py) 无条件先执行：

```text
gx = Deconv2d(gy, W)
  → xp.tensordot(W, gy, (0, 1))
  → xp.rollaxis
  → col2im_array
gW = Conv2DGradW(x, gy)
  → im2col_array(x)
  → xp.tensordot(gy, col, ((0,2,3),(0,4,5)))  [GPU]
gb = gy.sum((0,2,3))                         [有 bias 时]
```

本轮仅把 gx 分支变为 `gx=None; if x.requires_grad: ...`。gW、gb 的运算内容、顺序和公式不变。conv1 的输入由 benchmark 在 D 组包装为 False；conv1 输出及后续特征默认 True，因此 conv2–conv5 仍执行需要的 dInput。首层 dWeight 仍使用原始输入值；常量输入不意味着常量权重梯度。MSE target 梯度没有在本轮剪枝。

反向调度在每个输入上先处理 `gx is None or not x.requires_grad`，避免对 None 做梯度相加或把它的 creator 加入队列。其他输入照常独立累积。专门测试使用有 creator 的被跳过输入，确认 None 不会意外触发其上游；并检查多层卷积的中间梯度。

本轮没有新增 CUDA Kernel。Combined 的 Forward 仍走既有 native im2col/GEMM，Conv Backward 仍走原 CuPy Tensordot/Col2im。减少的 Kernel 来自不提交整个无用 gx 分支，而非修改 Col2im 数学或 CUDA 实现。

## 高阶路径发现与局部修正

原 `Conv2DGradW.backward(gys)` 使用 `gW, = self.outputs`，取得的是 weakref，然后把它传给 Deconv2d/Conv2d。原始复现报错 `TypeError: <class 'weakref'> is not supported`，见 [修正前日志](tests/higher-order-before.log)。原始复现脚本保存在 [reproduce_higher_order.py](scripts/reproduce_higher_order.py)。

这里应使用传入的高阶梯度 `gys` 作为二阶反传的权重参数，所以局部改为 `gW = gys`；同时对常量 x 的混合输入梯度使用相同需求判断。没有更改卷积数学公式，修正的是被错误传递的对象。修正后原复现通过，见 [修正后日志](tests/higher-order-after.log)。CPU/GPU 测试还与 PyTorch 比较二阶权重梯度、需要输入梯度时的混合导数，以及普通标量三次函数的二阶导数；不能仅以“不再抛异常”作为高阶正确性证明。

这项旧缺陷修正适用于默认和剪枝路径，但普通 `create_graph=False` 训练不执行 `Conv2DGradW.backward`。五组性能测量使用同一个修正后代码版本，因此没有跨不同高阶实现计算累计收益。

## 默认和优化器兼容

[`SGD`](../../../hytorch/optimizers.py) 未修改。参数和梯度仍为原 Variable/Parameter 对象，过滤、hooks、FreezeParam、普通和 Fused 更新均沿用 4B。剪枝不会把模型参数标记为 False。

[`bench_cuda_training.py`](../../../benchmarks/bench_cuda_training.py) 新增 `--prune-input-grad`，默认关闭。只在选中时把送入模型的 x 包装成 `Variable(..., requires_grad=False)`；target、loss、SGD、样本顺序和预热恢复逻辑不变。保留 `--no-fingerprints`；本轮没有运行已有 benchmark 的可选哈希路径。

## 长期 A–E 基线与 Native 排除

| 版本 | matmul / conv / pool | SGD | 输入需求 |
|---|---|---|---|
| A | cupy / cupy / cupy | 普通 | 默认 True |
| B | shared / im2col / cupy，tile16 | 普通 | 默认 True |
| C | shared / im2col / cupy，tile16 | Fused | 默认 True |
| D | shared / im2col / cupy，tile16 | Fused | 明确 False |
| E | PyTorch CUDA | 原匹配 benchmark SGD | PyTorch 输入默认无梯度 |

A 必须称为“**当前代码复现的无 Native CUDA C 基线**”，不是旧 Commit 的原始测量。A 子进程的 `HYTORCH_CUDA_LIBRARY` 特意指向不存在的文件，任何意外 native 加载都会明确失败；另外核对全部 A 测量的 dispatch_counts 为空，以及 A Trace 不含源码定义的 GEMM/Conv/Pool/Fused SGD Kernel 名称。CuPy 使用自己的 CUDA Kernel、cuBLAS 属于 A 的正常执行路径，不能据此称作调用本项目自研 Kernel。

## Profiling 归因

复用 4A 的 [OperatorRanges](../donkeynet-backward-profiling/scripts/profile_training.py) 和 [analyze_trace.py](../donkeynet-backward-profiling/scripts/analyze_trace.py)，旧文件未改动。它从真实嵌套调用标记每层 dInput、dWeight、dBias，及内部 Tensordot/Im2col/Col2im。由于 D 的 conv1 不再调用 Deconv2d，其 dInput 范围应不存在；不能只检查最终 `x.grad is None`。

GPU 事件使用 correlationId 关联到 CUDA 提交 API，再关联 NVTX 父节点。主机 NVTX 时长、API 提交/同步、GPU Kernel 实际执行和单独的未插桩吞吐分别统计。Copy Kernel 与 CUDA Memcpy 是不同记录；Tensordot、dInput、Col2im 等包含关系可能重叠，不能直接把所有分类相加。源码只确定调用关系，实际消失的数量与时间以本轮 Trace 数据为准，不沿用 4A 的 83.49 μs 当成新的测量。
