# 本轮源码修改

## 生产代码：hytorch/backends/cuda/kernels.cu:98

只修改 hy_gemm 的选择条件，保留 ABI、全部参数检查、stream 参数、variant0/1 路径、gemm_shared/register 数学实现以及 FP32 累加语义。

```cpp
const int64_t grid_m = ((int64_t)m + 63) / 64;
const int64_t grid_n = ((int64_t)n + 63) / 64;
const int64_t register_blocks = grid_m * grid_n;
constexpr int64_t kMinRegisterBlocks = 128;
const bool use_register = variant == 2 && m >= 64 && n >= 64 && k >= 64 &&
                          register_blocks >= kMinRegisterBlocks;
```

原先仅 m/n/k>=64 即用 register；现在还需 register 候选 CTA>=128。否则 variant2 沿原有 else 使用 shared。计算采用 int64，未增加设备查询、同步、数据拷贝或新的 C ABI。

128 是用户指定待验证初值；报告已列出 120/127 CTA 的反例，本轮没有擅自调参或改 kernel。构建继续 sm_80、-O3、-std=c++14、-lineinfo、shared cudart，无 TF32/fast-math。

通过把新增条件还原后与 raw/kernels_before.cu 整文件比较，确认除此之外无任何变化；hytorch/functions.py 与 runtime.py SHA 完全不变。前一轮 FP32 修复和整数 dtype guard 保留。

## 测试

- tests/test_backward_dtype.py：将现有整数输入 MSE 测试参数化到 CPU/CUDA，完成上轮未验证路径。
- tests/test_gemm_dispatch.py：21个明确形状 × tile16/32=42项实际选择与数值测试；另4项 variant0/1 不重定向测试。CUDA Graph debug DOT 是运行时实际 kernel 节点，断言预期名字，并执行图后核对 FP64 参考。包含预分配输出、非默认 stream、连续3次调用、CTA127/128/129、维度63/64/65、极不对称、K尾块和不规则矩形。大型1024/2048/4096方阵另外在性能基准逐位/FP64参考及图节点验证。

## 实验脚本

仅放在 ignored runs/a100_20261106/gemm_shape_dispatch/diagnostics/，没有生产性能 hook。保存修改前/后的 .so 用于同进程 rt._library handle切换。forced_shared 使用此前保存的原始 shared 库；三个库用同一输入和输出预分配方式。

CuPy Python cuBLAS包装层拒绝 graph capture，因此测量脚本使用 CuPy 创建的同一handle、同一已加载 cuBLAS 动态库的标准 cublasSgemm_v2 ABI，并设置默认FP32 math。行主序 A@B 通过列主序 C^T=B^T@A^T实现。没有改CUDA数学或降低精度。失败的初次尝试保留在 raw/gemm_paired.log。

## diff与既有工作

raw/dispatch_only.patch 是这次生产修改。git_diff.patch 是包含既有未提交修改和新增测试的完整diff：其中原 register kernel、README、其他旧测试修改不是本轮新增。raw/before.patch、before_status.txt 和 before_hashes.json 保存开始状态。

没有 Commit、Push 或 PR。
