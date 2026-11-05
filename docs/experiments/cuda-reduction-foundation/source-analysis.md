# 源码覆盖与归约语义

审查基线 `4530176dfdbd320bbd73c18293fabd391def86e4`。以下为源码事实，
不是本轮 GPU 实测。`CuPy` 表示通过 CuPy 提交，可能调用其生成 Kernel、cuBLAS
或仓库已有 ElementwiseKernel；不能一概称为第三方计算，也不能算本项目的 ctypes 原生库。

## 已有原生 CUDA C/C++ 路径

| 算子 | CUDA 源码/入口 | Python 接入与限制 |
|---|---|---|
| GEMM basic/blocked/shared | `kernels.cu::hy_gemm` | `runtime.gemm`，FP32，tile16/32；MatMul/Linear **Forward** 可选 |
| Conv2d Forward | `kernels.cu::hy_conv/hy_im2col/hy_conv_pack` | `runtime.conv2d`，direct 或 im2col+GEMM+pack；不是融合卷积全流程 |
| MaxPool Forward/argmax | `kernels.cu::hy_pool` | `runtime.max_pool2d`；不包含完整 Pooling Backward |
| Conv2d dInput | `conv_dinput.cu::hy_conv_dinput`、`conv_dinput_opt.cu::hy_conv_dinput_opt` | `Conv2d.backward` 的显式 fused/fused_opt；auto 为既有 shape 规则 |
| Conv2d dWeight | `conv_dweight.cu::hy_conv_dweight` | Split-K tiles+reduce 两个 Kernel，显式启用；已有 Workspace |
| Multi-Tensor SGD | `fused_sgd.cu::hy_multi_tensor_sgd` | `SGD(fused=True)`；不是 Adam/AdamW |
| ReLU Backward | `relu_backward.cu::hy_relu_backward` | 既有可选原型；此前 GPU 验收仍 NOT_RUN，本轮未修改 |
| **本轮：Sum** | `reduction.cu::hy_reduce_sum` | 一个通用 Kernel 重用为 partial/final；默认关闭，GPU NOT_RUN |

上述文件位于 `hytorch/backends/cuda/`。已有高阶 Conv/ReLU 求导保留可微原路径。
首层 dInput 剪枝是 `requires_grad`/自动求导调度优化，**不是新增 CUDA Kernel**。

## 仍经 NumPy/CuPy 的路径

| 路径 | 真实调用位置 | 可复用性/注意事项 |
|---|---|---|
| Linear/MatMul Backward | `functions.py::Linear.backward/MatMul.backward` 显式 `MatMul(native=False)` | 走 dot/cuBLAS，不因 Forward 开启 native 就成为原生 GEMM 梯度 |
| BMM 前向/反向 | `functions.py::BatchMatMul` -> `xp.matmul` | Transformer QK、AV；与已有二维 GEMM 不等价，不应直接替换 |
| Conv 原始 dInput | `functions_conv.py::Deconv2d.forward` -> tensordot/rollaxis/col2im | Layout view 是否物化由后续算子决定；原生 Gather 已存在，不重做 |
| Conv 原始 dWeight | `Conv2DGradW.forward` -> im2col/tensordot | 完整 col 大小 N·Cin·Kh·Kw·Oh·Ow；已有 Split-K 方案，不重复实现 |
| Im2col/Col2im 原路径 | `_im2col_gpu/_col2im_gpu`，CuPy ElementwiseKernel | 仓库自写 GPU 代码，但不属于独立 CUDA C/C++ 动态库 |
| Pooling Backward | `Pooling2DGrad.forward` -> zeros/scatter/swapaxes/col2im | 存在大 gcol；有改进可能，需先证明目标网络实际使用 |
| ReLU/Tanh/Sigmoid 等 Forward | `functions.py` 对应 Function | 原数组运算；Tanh/Sigmoid 梯度也是组合逐元素运算 |
| Sum、Mean、SumTo | 见下文 | 本轮唯一原语覆盖 CNN/MLP/Transformer 的广播梯度、bias、全局池化 |
| Softmax/logsumexp/交叉熵 | `functions.py::Softmax/SoftmaxCrossEntropy`、`utils.logsumexp` | 前向直接 ndarray.max/exp/sum；**没有自动接入本轮 Sum**。Softmax backward 的 Variable.sum 接入 |
| LayerNorm | `layers.py::LayerNorm.forward` | 两次 F.mean、中心化、平方、幂、affine；F.mean 接入 Sum，不等于已融合 LayerNorm |
| BatchNorm | `functions.py::BatchNorm` 与 `functions_conv.py::batch_norm` 两套实现 | 原数组 mean/var 仍走原实现；部分 Variable.sum 梯度接入；不能声称全部归约已替换 |
| Adam/AdamW/梯度裁剪 | `optimizers.py` | 数组逐元素与 norm/sum；已有 SGD 融合不覆盖这些优化器 |
| reshape/transpose/concat/getitem/梯度累加 | `functions.py`、`core.py::Variable.backward` | view 本身不必启动 Kernel；后续 contiguous、copy 或广播梯度才可能产生 GPU 工作 |

## Sum / Mean / SumTo 的现有语义

1. `Sum.forward` 保存原 shape，支持 None、整数、负轴、轴 tuple、keepdims。
   backward 通过 `reshape_sum_backward` 与 `broadcast_to` 恢复输入梯度。
2. `F.mean` 原本就是 `sum(x, axis, keepdims) * (output_size / x.size)`；本轮不改缩放公式。
   Mean 不是第二个新 CUDA 算子，空输入仍沿用原行为。
3. `utils.sum_to` 将输入多出的前导轴和目标为1的轴相加，keepdims=True，再 squeeze 前导轴。
   `F.SumTo.backward` 使用 broadcast_to；广播 Add/Mul/Linear bias 的梯度可复用。
4. `MeanSquareError.forward` 原公式为 `sum((x0-x1)**2)/len(diff)`，分母是 Batch 长度，
   **不是所有元素数**。只将该 sum 接入同一原语，差值、平方、缩放仍是原操作。
5. 默认仍直接调用 ndarray.sum，CPU/非FP32 显式 native 请求有可见计数后走原路径。
   8维以上、空输入保留原路径并计数。非连续 FP32 通过已有 `_arrays` 明确物化，完整算子必须计入 Copy。
6. rank0 的整数 axis=0/-1 与空 tuple 的语义按安装的 NumPy1.24 验证；重复/非法轴报错。
   PyTorch `dim=()` 意味全归约，与 NumPy/CuPy `axis=()` 不同，验收脚本已单独处理。
7. Sum backward 本身不调用不透明的原生反向；已有 broadcast/reshape 可继续构建高阶图。
   CPU 一阶/二阶测试已执行，GPU 高阶测试仅准备。

## 模型实际使用

- DonkeyNet：5 Conv + 2 隐层 Linear + 1 输出 Linear，7 ReLU、末尾 Tanh；无 Pool/Norm。
  bias 梯度、广播梯度和 MSE Sum 是归约入口，不能假定独立 Sum 本身占主要耗时。
- SpatialNet：3 Conv/ReLU + 空间 F.sum·scale + Linear/Tanh；覆盖另一种 CNN 的全局归约。
- Transformer：`transformer/attention.py` 经 LinearND、拆头 transpose/reshape、BMM、缩放、mask、
  Softmax、BMM、合头。LayerNorm 使用 F.mean，注意力 Softmax 前向仍使用原数组归约。
- `core.py` 的 Function/Variable 调度主要在 Python；不能用 GPU Kernel 时间估计 Python 独占耗时。

## 历史数据仅作选题依据

[上一轮瓶颈分析](../autonomous-optimization-6h/bottleneck-analysis.md) 引用4H的 I Trace：
Backward 核时1071.923µs，对应主机 NVTX 4224.490µs；7个 ReLU 梯度共14次 launch、
GPU60.118µs、主机范围772.328µs；原生 dInput574.902µs、dWeight301.138µs。
这些口径不同，不能相减得到 Python 独占时间，更不能作为本轮加速数字。
ReLU 原型已经存在，因此本轮不重复开发它。Sum 的价值首先是通用归约基础设施，
是否快于 CuPy/CUB 必须留给匹配 GPU 实验。
