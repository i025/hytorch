# Multi-Tensor SGD 源码与内存生命周期分析

本文描述阶段 4B 实际源码、CPU 模型形状查询和离线 CUDA 编译结果。GPU 数值、Launch 数量与性能结论采用本目录主报告及原始实验数据，不能由源码中只有一次 Launch 推断实际加速。

## 1. 原 SGD 与本轮改动边界

[`Layer.params()`](../../../hytorch/layers.py#L36) 递归遍历模型的 `_params` 集合，返回 `Parameter`。集合遍历不承诺跨进程固定顺序；本轮没有改变参数注册或遍历规则。训练脚本的 `parameters()` 另按名称排序，用于初始化、快照和跨框架数值比较。

[`Optimizer.update()`](../../../hytorch/optimizers.py#L19) 先收集非空梯度参数，依加入顺序调用全部 hooks，随后再次过滤梯度。第二次过滤保留 `FreezeParam` 在 hook 内将 `grad` 设为 `None` 的既有语义。原 `SGD.update_one()` 执行 `param.data -= lr * param.grad.data`：对 GPU FP32 数组而言，乘法产生临时结果，再由减法写回参数。阶段 4A 补测实际观察到 DonkeyNet 每步 16 次 Multiply 和 16 次 Subtract，而非仅凭 Python 表达式推断次数。历史证据见 [4A 补测报告](../donkeynet-backward-profiling/supplement-01/README.md)。

新接口 `SGD(lr=..., fused=True)` 只改变显式启用时的 SGD 更新阶段：

1. `fused=False` 直接调用原 `Optimizer.update()`，原默认行为保留。
2. `fused=True` 仍执行相同的首次过滤、全部 hooks 和第二次过滤；没有活跃梯度时不提交 Kernel。
3. 全部参数为 CPU 数组时仍调用原 `update_one()`；包括原有 CPU dtype/布局行为。
4. 全部为 CUDA 参数时调用 `runtime.multi_tensor_sgd()`；混合 CPU/CUDA 参数显式报错，不静默切换后端。

MomentumSGD、Adam、AdamW 等其他优化器未修改。Forward、Backward、计算图、模型结构和 `kernels.cu` 中的原有 GEMM/Conv/Pooling 均未修改。Hook 本身可能调整梯度，例如 `WeightDecay`；“Fused SGD 不修改梯度”指更新 Kernel，而不是禁止已有 hook 的正常行为。

## 2. 参数与梯度的归属

`Parameter.data` 与 `Parameter.grad.data` 是当前 CUDA 设备上的 CuPy ndarray；参数对象及梯度 Variable 持有相应设备内存。`Layer.to_gpu()` 通过 `Variable.to_gpu()` 转换参数数据。训练时 `model.cleargrads()` 将旧梯度对象置空，随后 [`Variable.backward()`](../../../hytorch/core.py#L126) 设置新梯度或生成新的相加结果。因此不能缓存上一步的梯度地址；即使内存池偶尔复用相同地址，也不构成地址稳定性的保证。

`_sgd_metadata(parameters, gradients)` 每次读取当前数组的 `.data.ptr`，而非持久化梯度地址；缺失梯度直接过滤。它检查：

- 参数和梯度均为 CuPy FP32、C-contiguous、同一当前设备，同一对具有相同 shape。
- 每个活跃张量有 `1..INT_MAX` 个元素；本轮支持最多 64 对活跃张量。
- 参数存储不与另一参数或任何梯度存储重叠，以免并行写入竞争或修改只读梯度；不同梯度之间可以共享只读存储。

不满足限制时报错，不复制为连续数组、不偷偷拆成其他实现。CPU 保留原路径，因此这些限制只针对显式 CUDA Fused 分支。没有梯度的参数、全部冻结参数和空列表都不触发 CUDA Launch。

## 3. DonkeyNet 的 16 个张量

下面来自 CPU 实际执行 `DonkeyNet(dropout_rate=0.)` 和训练脚本 `initialize(model, (3,66,200), 42)`，未使用 GPU、未计算 fingerprint。机器可读结果为 [parameter-shapes.json](results/parameter-shapes.json)。表格按名称排序，只用于展示；Kernel 描述符按本次优化器遍历顺序配对。

| 参数 | HyTorch shape | FP32 元素数 | 字节数 | 256 线程块数 |
|---|---|---:|---:|---:|
| conv1/W | (24, 3, 5, 5) | 1800 | 7200 | 8 |
| conv1/b | (24,) | 24 | 96 | 1 |
| conv2/W | (36, 24, 5, 5) | 21600 | 86400 | 85 |
| conv2/b | (36,) | 36 | 144 | 1 |
| conv3/W | (48, 36, 5, 5) | 43200 | 172800 | 169 |
| conv3/b | (48,) | 48 | 192 | 1 |
| conv4/W | (64, 48, 3, 3) | 27648 | 110592 | 108 |
| conv4/b | (64,) | 64 | 256 | 1 |
| conv5/W | (64, 64, 3, 3) | 36864 | 147456 | 144 |
| conv5/b | (64,) | 64 | 256 | 1 |
| fc1/W | (1152, 100) | 115200 | 460800 | 450 |
| fc1/b | (100,) | 100 | 400 | 1 |
| fc2/W | (100, 50) | 5000 | 20000 | 20 |
| fc2/b | (50,) | 50 | 200 | 1 |
| fc3/W | (50, 1) | 50 | 200 | 1 |
| fc3/b | (1,) | 1 | 4 | 1 |
| **合计** | **16 个张量** | **251749** | **1006996** | **993** |

权重布局遵循原 HyTorch：Conv 为 OIHW，FC 为 `[in, out]`；PyTorch FC 权重 `[out, in]` 在训练对照中按既有代码转置映射。Fused 更新仅按连续元素索引，完全不改变这些布局。一次 DonkeyNet 更新的参数读、梯度读、参数写合计逻辑数据量为 `251749 × 4 × 3 = 3020988` 字节；这是代码的数据访问量，不是实测显存总线流量。

CPU 形状查询可复现为：

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python - <<'PY'
from benchmarks.bench_cuda_training import initialize
from donkeycar_hy.model import DonkeyNet
state = initialize(DonkeyNet(dropout_rate=0.), (3, 66, 200), 42)
for name, value in state.items():
    print(name, value.shape, value.size, value.nbytes, (value.size + 255) // 256)
print('total', len(state), sum(a.size for a in state.values()))
PY
```

## 4. CUDA Kernel、当前 Stream 与元数据

[`fused_sgd.cu`](../../../hytorch/backends/cuda/fused_sgd.cu) 独立编译进原 CUDA 动态库。每个描述符包含参数地址、梯度地址、元素数量和累计线程块末端；每块 256 个线程。Host 对每个张量分配 `ceil(size/256)` 个块，Kernel 根据当前 block 的累计范围确定对应张量。线程只更新属于该张量的有效元素，尾部不足一个块的元素使用边界判断。

DonkeyNet 在源码配置下为一个包含 993 个块的 Launch；每个块不跨张量。64 张量上限以及每张量 `INT_MAX` 元素限制下，总块数最多 536870912，落在使用的有符号 int 范围内。实际 Launch 次数需要 Nsight Trace 验证。

调用链为：

```text
SGD.update(fused=True)
  → runtime.multi_tensor_sgd(parameters, gradients, lr)
    → _sgd_metadata(): 验证并创建当前地址/长度的 host ctypes 数组
    → _submit_sgd(): 按需绑定新 ABI 符号
      → _call(): cp.cuda.get_current_stream().ptr
        → hy_multi_tensor_sgd(..., stream)
          → multi_tensor_sgd_kernel<<<blocks, 256, 0, stream>>>(metadata)
```

ABI 把 host 数组中的地址和长度复制到一个纯数据结构 `SgdMetadata`，随后按值作为 Kernel 参数传入。64 项每项 24 字节，加 count 和 FP32 learning rate 各 4 字节，实际参数大小 **1544 字节**；即使只更新 16 个张量，本实现仍传固定大小结构。静态断言将结构限制在 4096 字节以内。CUDA 12.2.2 文档指出全局函数参数通过常量内存传入，并说明 host launch stub 先按值复制，再把参数复制到设备参数存储。因此 host ctypes 数组及 C++ 栈上的描述符可以在提交函数返回后释放，不需要额外 pinned buffer、Event 回收队列或常驻设备描述符。[CUDA 参数传递与参数处理](https://docs.nvidia.com/cuda/archive/12.2.2/cuda-c-programming-guide/index.html#global-function-argument-processing)

参数与梯度的设备存储仍由调用者保持有效；跨 Stream 的生产、消费和释放须遵守正常 CUDA 顺序。这一接口把更新提交到 CuPy 当前 Stream，不自行同步，不将操作转移到默认 Stream。测试中的非默认 Stream 检查覆盖“梯度生产 → 更新 → 参数消费”的顺序。

这里**没有显式 `cudaMemcpy*` API、设备 malloc/free、参数打包缓冲区或同步调用**，但存在 Kernel 参数的隐式传输，不能把它称为零成本。Nsight 中若没有独立 Memcpy 事件，只能报告没有显式拷贝事件；参数传输及 C++ 描述符填充并入提交路径，无法从这个 Trace 单独拆出纯传输时间。

实验专用脚本在 `_sgd_metadata`、`_submit_sgd` 和 `_call` 外设置 NVTX 范围，以分别记录 Python 验证/准备、native 提交和 ctypes 路径的主机时间。这些嵌套范围不能相加。CUDA API Launch 时间是主机提交时间；通过 correlationId 关联到的 GPU Kernel 时间是设备执行时间；完整 `Optimizer.update()` 同步墙钟还包含等待，两者也不能混算。

## 5. FP32 舍入与数值标准

Kernel 执行 `__fsub_rn(parameter, __fmul_rn(lr, gradient))`：乘法按 FP32 最近偶数舍入，减法再按相同方式舍入。它融合 Launch 和访存组织，刻意保留原 CuPy 两个算子的两次舍入，不把算术改为一次 FMA。NVIDIA CUDA 12.2.2 Math API 明确规定这两个 intrinsic 不会合并成 multiply-add。[CUDA FP32 intrinsic 文档](https://docs.nvidia.com/cuda/archive/12.2.2/cuda-math-api/group__CUDA__MATH__INTRINSIC__SINGLE.html)

本轮没有调整既有 PyTorch 对齐容差。PyTorch 的内部更新顺序仍可能有正常 FP32 舍入差异；跨框架是否通过由实测误差判定。相同 HyTorch Forward/Backward 加原 SGD/Fused SGD 的逐元素相等结果也应由真实测试给出，不能仅凭上述表达式宣布全部输入逐位一致。

## 6. 编译与兼容性

[`scripts/build_cuda.py`](../../../scripts/build_cuda.py) 在既有 `kernels.cu` 外增加 `fused_sgd.cu`；原 GEMM/Conv/Pooling 源文件保持原样。新 `--output DIRECTORY` 可将动态库、build.log 和 build.json 保存在独立实验目录，避免覆盖已有记录；未指定时保留原默认目录。原 CUDA ABI 版本和既有符号签名不变。新符号只在显式 Fused 调用时绑定，所以旧库仍可用于原路径；缺少新符号时明确要求重新编译，不自动退回原 SGD。

SM80 离线编译检查保存在服务器 `runs/fused_sgd_cuda_ptx_01/`。命令使用 Toolkit 12.2.2 的 nvcc，仅生成 PTX 和 cubin，没有执行 GPU 工作。实际 PTX 包含 1544 字节参数区，并直接使用动态索引 `ld.param`；没有 `.local` 声明。ptxas 实测为 12 个寄存器、0 字节 stack、0 字节 spill store/load，constant bank 0 为 1896 字节、bank 2 为 8 字节；常量区总量包含编译器启动参数等内容，不等同于用户描述符大小。

CUDA 文档说明，对 Kernel 参数取地址可能产生逐线程 local copy，`__grid_constant__` 可以避免这种复制；这里先检查实际产物，确认当前代码没有该问题，因此没有增加架构相关注解。[CUDA `__grid_constant__` 说明](https://docs.nvidia.com/cuda/archive/12.2.2/cuda-c-programming-guide/index.html#grid-constant)

## 7. 源码审查结论与边界

独立只读检查覆盖块范围映射、尾元素保护、地址生命周期、缺失梯度过滤、当前 Stream、混合设备和重叠存储错误路径、默认 SGD 与 hooks 顺序；没有发现需要在 GPU 验证前阻止执行的实现错误。最大 64 对、连续 CUDA FP32、同设备、互不重叠参数是本轮显式支持边界，不是对任意优化器或任意 Tensor 的通用实现。

原 SGD 每个张量顺序提交；Fused 允许不同张量并发写入，因此禁止参数之间别名是必要的，而不是静默改变别名模型的更新顺序。不同学习率参数组、稀疏梯度、momentum、weight decay 融合、梯度缩放和多 GPU 组均不在本轮 CUDA Kernel 功能内。已有 WeightDecay hook 仍可先执行，但其自身 Kernel 和开销不计入“SGD 核心一个 Launch”的承诺。
