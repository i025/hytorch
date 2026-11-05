# 唯一原型：通用 FP32 Sum

## 支持范围与接口

```python
from hytorch.backends.cuda import backend
import hytorch.functions as F
with backend(reduction='native'):
    y = F.sum(x, axis=(0, 2, 3), keepdims=True)
```

默认 reduction='original'。原生支持非空 FP32、rank0–8、None/整数/tuple轴和 keepdims，
通用 shape，不按模型/层号调度；沿用 Runtime 的 INT_MAX 元素数限制。原生 ABI 输入连续；
Python 入口可通过已有 `_arrays` 物化非连续输入。CPU、其他 dtype、空输入、rank>8 保留原路径，
dispatch_stats 提供原因。FP16/BF16、整数累加、任意原始 stride ABI 本轮不实现。

权威源码：[CUDA](../../../hytorch/backends/cuda/reduction.cu)、
[CPU轴规划/ABI](../../../hytorch/backends/cuda/reduction.py)、
[Runtime](../../../hytorch/backends/cuda/runtime.py)。实验目录不复制生产源码，避免两份实现漂移。

## 映射与计算

144字节 SumLayout 由主机按连续 shape/轴构造。保留轴和归约轴各保存 shape 与输入元素 stride。
每个输出对应一个保留轴坐标，每个归约线性坐标经 div/mod 转成输入偏移。
因此不先执行整张量 transpose/packing。非末轴可能产生非合并访问，通用映射整数运算也有成本。

256线程/block、8warp；每线程步长256累加输入，用 `__fadd_rn` 固定FP32加法。
warp shuffle树归约，8个warp结果放入32字节 shared，再由第一warp归约。
所有256线程参与同步，最后的 block barrier 避免 grid-stride 下一任务覆盖 shared。
空轴 tuple 复制并加+0以保留 sum 的 signed-zero 语义。

归约长度R≤4096：每输出一个block，一个launch。R>4096：P=ceil(R/4096)，
先按输出和分段并行写 O×P workspace，再调用**同一个 Kernel**将P个部分和归约。
不使用浮点原子，不新增第二种算子；固定分段顺序不承诺与 NumPy/CuPy/FP64 逐位相同。
只有一个 `__global__ reduce_sum_kernel` 和一个导出 `hy_reduce_sum`。

当前流通过现有 `_call` 传入，未加 device synchronize。Workspace 和输出由 CuPy 内存池分配，
CPU训练/装载检查没有调用这些GPU入口。没有使用 CuPy RawKernel/ReductionKernel 替代原生实现。

## 理论成本，不是 GPU 测量

- 单段：约4N字节输入读、4O字节输出写，1次launch。
- 分段：中间workspace大小4OP，额外读写约8OP字节，2次launch；输入仍约读4N。
- 非连续物化还需约8N字节额外读写；实际 Copy Kernel 数需 Nsight。
- 大O、小R时单block/输出利用率可能很差；跨轴 div/mod、访存不连续、workspace分配及 ctypes
  主机成本均可能使其慢于已有 CuPy。没有在线调优或自动改默认。
- ptxas：SM80，32寄存器/线程、32B shared/block、0 spill、0 stack。
  这是编译资源量，不是实测 Occupancy，也不证明该策略适合所有轴。

## 数值设计与已知限制

[CPU参考](scripts/reference.py) 分为独立FP64求和再FP32输出，以及按256lane/warp/分段顺序
模拟的FP32算术。它不模拟GPU线程调度和内存系统，不能作为GPU正确性证明。
CPU模型与安装的NumPyFP32及独立FP64各比较，沿用 atol=1e-5、rtol=1e-4。

144个 shape/输入/keepdims 组合 ×2参考 =288项比较，284通过、4失败。失败是2个不同的
强消去输入各重复 keepdims：

| 归约长度 | CPU模型 | NumPyFP32 | FP64→FP32 | 超差参考 |
|---:|---:|---:|---:|---|
| 65537 | -0.0006866455078125 | -0.0006872415542602539 | -0.0007131537422537804 | FP64（误差2.65082344e-5） |
| 1048577 | 0.006744384765625 | 0.006874799728393555 | 0.006749750580638647 | NumPy（误差1.30414963e-4） |

后一输入模型相对FP64的误差5.36581501e-6，比NumPy的1.25049148e-4小；前一输入两种FP32
都与FP64有约2.6e-5误差。因此与NumPy接近不能证明绝对精度足够，偏离NumPy也不能单独证明
索引错。已完整保存失败输入。**并未通过全部数值检查，不扩大支持精度声明或放宽容差。**
GPU脚本包含这两份原样输入；如失败，窗口停止，不产生通过后的性能结论。
没有为“测绿”改成FP64、补偿和或第二个Kernel策略；下一轮先判断真实GPU与CPU模型是否一致。

CUDA shuffle/shared同步依据 [CUDA12.2 Programming Guide](https://docs.nvidia.com/cuda/archive/12.2.0/cuda-c-programming-guide/index.html)。
安装的 NumPy1.24 实际语义由 CPU 测试确认，未用最新文档版本替代本地实测。
