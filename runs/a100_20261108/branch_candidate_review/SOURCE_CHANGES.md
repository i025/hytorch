# 源码变更与来源

来源：experiment/gpu0-autonomous-7h@d58c581；目标基线4317bf5。采用选择性移植，未进行整分支合并。

| 文件 | 修改 |
|---|---|
| hytorch/backends/cuda/relu_backward.cu | 来自来源同名kernel，额外加入非法指针/size检查；FP32乘法/current stream一致 |
| hytorch/backends/cuda/kernels.cu | 只在末尾include新kernel，既有内容逐字保留 |
| hytorch/backends/cuda/runtime.py | 新增可选relu_backward模式、ctypes ABI与wrapper；复用现有安全检查，默认original |
| hytorch/functions.py | ReLU.backward在显式fused+FP32且非create_graph时调用新kernel；其他路径保留 |
| hytorch/core.py | 移植显式requires_grad、None梯度/leaf处理、可选heap；增加bool参数校验；保留O1 |
| hytorch/functions_conv.py | 只有显式不需要输入梯度时跳过Conv2d与Conv2DGradW的dInput；保留packed dWeight |
| benchmarks/bench_cuda_training.py | 三个CLI开关、predict包装及配置记录；默认不变，monitor/helper共用 |
| tests/test_backward_schedule.py | 来源的调度顺序/高阶/异常测试 |
| tests/test_conv_dinput_pruning.py | 来源的输入梯度/参数/高阶/Torch对齐测试；SGD接口适配本分支 |
| tests/test_relu_fusion.py | 集成测试：dtype/fallback、layout、stream、特殊值、边界、GPU heap高阶 |

新增参数为keyword-only，默认requires_grad=True；该标志是显式梯度接收/遍历边界，不实现全图自动推断。默认调度list、默认ReLU original、CLI默认计算输入梯度。原有--final-config不偷偷改变含义。

原有kernel内容SHA验证见raw/source_integrity.json。git_diff.patch仅本轮生产/测试代码相对4317bf5的差异。baseline/包含原源码快照；编译库保留服务器，归档和Git上传不含.so。source/保存修改后源码。
