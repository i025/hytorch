# 实际调用路径与计时边界

## Conv2D

`core.Variable.backward` → 读取weak output.grad、按generation排序 → `Conv2d.backward` →

- dInput：`Deconv2d.__call__ → forward → CuPy tensordot(W,gy) → rollaxis → functions_conv.col2im_array → runtime.col2im → dtype/current-device/strides/shape/alias检查 → bounded immutable _col2im_metadata/POD → _call → fresh current stream → hy_col2im_fast → col2im_strided`。
- dWeight：`Conv2DGradW.__call__ → runtime.packed_dweight → dtype/device/geometry检查（F1 shape cache开启）→ contiguous input → fresh col → hy_im2col → gy.transpose/reshape packing → CuPy/cuBLAS GEMM → reshape`。
- dBias：`Variable.sum → Sum.__call__ → CuPy sum`。
- 包装各梯度Variable → 原有gx+old_grad累加 → 按原序调度creator → 按retain_grad清理输出梯度。

生产路径没有直接调用诊断接口。所有检查、分配、packing、GEMM、launch及stream查询都与B0相同。新增快路径只在Function.forward返回之后改变临时Python容器。

## Pooling

`functions.pooling → functions_conv.Pooling.__init__ → Function.__call__ → Pooling.forward → selected('pool',x) → runtime.max_pool2d → _arrays → _geometry → empty(y,float32) + empty(indexes,int64) → _ptr / _call / ctypes → current stream → hy_pool → pool_kernel → 保存indexes → output Variable`。

Backward：`Pooling2DGrad` 保存输入shape/dtype及indexes → zeros(gcol) → arange/索引偏移 → scatter gy → reshape/swapaxes → col2im。高阶使用Pooling2DWithIndexes，不改变max winner状态。

原生池化padding为零。局部对比输入用严格正值，确保Torch(-inf padding)选中相同winner；专项测试另外检查负值/零padding/边界和重复反向的新旧一致性。TinyNet输入在池化前经ReLU。

## 五层测量

| 层级 | 本轮实现与范围 |
|---|---|
| L1 | Nsight CUDA/NVTX/CUPTI；只加总实际kernel execution，15重复；另列kernel并集和数量 |
| L2 | conv direct：预先验证静态metadata，直接调用已有im2col/col2im C ABI，仍包括fresh临时数组和packing；pool direct另外预分配输出。只在diagnostics脚本中暴露。Torch conv direct使用aten.convolution_backward同样请求x/W/b |
| L2 checked | 原有底层runtime/Function.forward，仍执行安全检查，不创建Variable |
| L3 | conv.backward内构造并执行Deconv2d/GradW/Sum三个Function及包装结果，no_grad语义同普通backward；pool新建完整Pooling Function并保存索引。没有同一抽象边界的Torch项留空，不虚构逐层对应 |
| L4 | HyTorch真实Variable.backward遍历+梯度赋值/累加；Torch autograd.grad保留图，同样求x/W/b。重复使用图避免把forward算入backward，清梯度在计时内 |
| L5 | 无profiler完整训练，从DataLoader开始，包含H2D、forward/backward/SGD和最终sync；预热/初始化/结果D2H不在计时内 |

L3 dInput/dWeight/dBias独立Function诊断允许创建图，L3 backward_Functions则模拟默认backward的no_grad状态，两者不是可相加的组件。正式训练没有额外逐层同步。局部CPU提交与同步完成分别采样，两条独立样本不能相减。低开销component微计时为15批×1000调用，cProfile为另一次运行。

工作负载证据：WORKLOAD_COMPLETE.json（DonkeyNet逐Function shape/dtype/strides及GEMM）；WORKLOAD_TINY_POOL.json（TinyNet真实Pooling为4通道）。CSV中的tiny_pool是历史3通道通用探针，真实项为tiny_actual_pool。
