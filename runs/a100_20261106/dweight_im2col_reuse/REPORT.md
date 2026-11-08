# 阶段4D-1：前向im2col复用的数值准入实验

## 执行结果：在生产修改前停止

按用户第二阶段停止条件，本轮完成源码核对、独立数学/布局/精度实验和原始基线验证后停止。没有增加reuse开关，没有修改生产或测试源码，也没有修改CUDA动态库、GEMM、Col2im、dInput或dWeight原路径。已有4C-2缓存/POD与FP32/GEMM128CTA修改完整保留。

**停止依据是保守的旧实现数值兼容性准入，而非数学公式错误或已证明新实现更不准确。** 将现有DonkeyNet梯度比较数值门限atol=2e-5、rtol=2e-4用于B对A的替换一致性检查，90组中12组超限（12个元素，4个shape/seed组合在3种输入layout下重复）。新B对FP64在这一门限下全部通过；旧A对FP64有9组失败。因此不能把新旧不一致直接解释为新实现精度下降。若审查认为应以高精度参考而不是旧结果作为唯一准入依据，需要明确这一验收解释后再实施，不能在本轮自行放宽或替换验收含义。

原有测试门限本来用于其指定样本与PyTorch比较，并不是仓库声明的全形状误差定理。本轮将同样数值用于unit-normal通用数据的新旧兼容性诊断，原测试本身仍全部通过；没有声称原DonkeyNet单元测试失败。小型dtype更严格门限仅作辅助展示，不用于单独宣判算法错误。

## 当前源码核对

- functions_conv.py Conv2d.forward选择native_conv，runtime.conv2d的im2col路径先 `_arrays` 打包非连续输入，再生成(T,D)连续矩阵、hy_im2col、forward gemm、hy_conv_pack。返回仍只有y，col未传给Conv2d实例。
- Conv2d.backward无条件调用Conv2DGradW(self)(x,gy)，GPU Conv2DGradW.forward重新im2col_array(to_matrix=False)，再tensordot(gy,col,((0,2,3),(0,4,5)))。与指定设计前提一致。
- hy_im2col令row=i/(IC*KH*KW)、q=i%(IC*KH*KW)；row解码n/oh/ow，q解码ic/kh/kw，确实是T=(n,oh,ow)、D=(ic,kh,kw)。
- 核对Config.enable_backprop/create_graph、Variable.backward、Conv2DGradW.backward及现有dtype/高阶梯度测试。当前Variable.data直接暴露可变ndarray，没有通用原地修改version counter，未来复用需要明确保护/失效机制；仅保存指针不能证明内容未变。本轮未实现或验证新的生命周期处理。

## 独立实验

A = 当前GPU im2col_array(...,to_matrix=False) + 原CuPy tensordot。
B = 现有hy_im2col产生真实forward(T,D)列矩阵，G=gy.transpose(1,0,2,3).reshape(OC,T)，B=(G@X_col).reshape(OC,IC,KH,KW)。使用CuPy/cuBLAS矩阵乘法；没有改native GEMM。

数据：5个DonkeyNet真实形状+5个通用形状（非方形kernel、不规则尾块、重叠、padding、stride>1、holes、不同batch/channels）；每个3种输入布局（连续/负stride/步长2slice）×3个seed(42,123,501)，共90组。x、gy均固定seed的N(0,1)转FP32，覆盖通用数值范围，不代表192步训练中每层的实际梯度尺度。每个输入layout表达相同逻辑值，3种layout重复并非3个独立随机样本。统一非默认CUDA stream，NVIDIA_TF32_OVERRIDE=0，没有fast-math。

90/90原native列矩阵与原im2col六维tensor重排结果逐位相等，证明布局公式与非连续输入packing正确。dWeight只有54/90组逐位相等；其余36组改变FP32归约结果。逐项误差判据为abs(B−A) <= atol+rtol*abs(A)。相对误差定义abs(diff)/max(abs(reference),1e-30)，近零位置会放大，因此必须结合绝对误差与混合门限。

| 形状 | 最大 |B−A| | 最大相对误差 B/A | A 对 FP64 最大绝对误差 | B 对 FP64 最大绝对误差 | B/A 超过 Donkey 门限的组数 |
|---|---|---|---|---|---|
| conv1 | 0.00015258789 | 0.00054574639 | 0.00012612692 | 0.00014158245 | 3/9 |
| conv2 | 6.4849854e-05 | 0.0075414781 | 5.5964098e-05 | 4.2116038e-05 | 9/9 |
| conv3 | 1.5258789e-05 | 0.35294118 | 1.3462883e-05 | 1.3120089e-05 | 0/9 |
| conv4 | 0 | 0 | 8.7419767e-06 | 8.7419767e-06 | 0/9 |
| conv5 | 0 | 0 | 5.4970078e-06 | 5.4970078e-06 | 0/9 |
| irregular | 0 | 0 | 7.6933026e-06 | 7.6933026e-06 | 0/9 |
| overlap | 6.8664551e-05 | 0.011673152 | 6.0261013e-05 | 1.6433611e-05 | 0/9 |
| point | 0 | 0 | 7.5675129e-06 | 7.5675129e-06 | 0/9 |
| holes | 0 | 0 | 1.3364117e-06 | 1.3364117e-06 | 0/9 |
| tail | 0 | 0 | 1.0897233e-05 | 1.0897233e-05 | 0/9 |


| 门限来源 | atol / rtol | B vs A 失败组数 | A vs FP64 失败组数 | B vs FP64 失败组数 |
|---|---|---|---|---|
| test_cuda_host.py Conv backward | 1e-5 / 1e-4 | 21 | 24 | 12 |
| test_backward_dtype.py DonkeyNet | 2e-5 / 2e-4 | 12 | 9 | 0 |
| 小型 Conv+Linear dtype 测试（辅助） | 2e-6 / 2e-5 | 36 | 45 | 42 |


“B对FP64通过Donkey门限”不意味着B每个元素都更准确：例如conv1最大绝对参考误差B=1.4158e-4、A=1.2613e-4，最大误差位置与混合门限失败位置不同。报告完整保留两边参考误差，不选择性展示。

## 可复现失败坐标

conv1 / seed42 / contiguous，dW[20,1,0,2]：

- A=-0.11882781982421875
- B=-0.11876296997070312
- 独立CPU long-double参考=-0.11878202656945513
- |B−A|=6.4849853515625e-5，原Donkey数值门限此处约4.3765564e-5，超限1.4818倍。
- A参考误差4.5793255e-5，B参考误差1.9056599e-5。该例新结果更接近参考，但不满足按旧结果判定的兼容性检查。

conv2 / seed123 / contiguous，dW[14,3,1,2]：A=-0.008676528930664062，B=-0.008701324462890625，CPU参考=-0.008687375804047267。两者对FP64均在Donkey门限内，但B−A约2.47955e-5超过旧结果兼容门限，说明两种FP32舍入不能简单视为同一结果。

完整失败点见raw/failure_examples.json。CPU独立im2col + np.longdouble归约复核7个记录（含重复comparison），与GPU FP64参考最大差约1.04e-13，排除了复用同一GPU布局导致的参考错误。不是由TF32或降精度导致；实际cuBLAS内部归约树差异是合理解释，本轮没有采集其底层算法ID，不将此推断表述为已观测kernel实现细节。

## 验证结果

主机24+9=33 passed；原GPU204 passed、12 CPU项目deselected、0 skipped。日志在raw/baseline_*.log，强制HYTORCH_REQUIRE_CUDA=1。memcheck和racecheck分别执行完整90组独立实验，错误0、race hazards0。数值兼容性失败作为数据输出，不是内存安全错误；Sanitizer退出码0也不代表数值门限通过。

这些是未修改基线与独立算术实验的验证，不能当作已实现reuse的高阶梯度、重复backward、no_grad、输入原地修改或跨step生命周期验证。后者没有候选实现，未执行。

## 性能与显存：未执行项明确留空

因数值准入未通过，未实现生产reuse，未运行10轮端到端或新旧Nsight性能比较，也未宣称消除backward im2col。DWEIGHT_BENCHMARK.csv包含90组数值准入数据，所有GPU/CPU时延字段留空；TRAINING_COMPARISON.csv、PROFILE_BEFORE_AFTER.csv以NOT_RUN和原因记录，不能当作零耗时或零加速。

MEMORY_COMPARISON.csv的单case col_bytes来自真实分配矩阵的nbytes；理论若五层都保留列矩阵，batch4 payload合计12,127,008 bytes（11.565 MiB），batch同比线性增长。batch1/4/8/16行均明确标为公式推算，不是训练峰值显存实测。未测额外存活时间、allocator峰值或跨step增长，不据此判断内存代价是否可接受。

本轮停止点无需运行未验证候选的192步性能；也没有引入保持旧累加布局的第二方案、implicit GEMM、梯度裁剪或其他CUDA kernel。前轮任何0.83s训练时间不作为本轮性能结果复用。

## 资源与交付

使用物理GPU1 A100-PCIE-40GB，先确认0MiB/0%且无计算进程。GPU0其他任务未触碰。同一阶段使用窗口内完成数学实验与基线验证，最终GPU1无计算任务。源码和动态库SHA与baseline完全一致。Git仍保留进入本轮时的未提交修改，没有Commit、Push或PR。

baseline保存原始源码、已有git diff、状态和动态库；source保存复现依赖；git_diff.patch是空文件，表示本阶段没有生产或测试源码变更。诊断脚本、原始90组数据、失败坐标、独立参考、验证日志、环境和命令均打包；动态库留服务器并给出SHA，不包含大型数据集/权重。完成后停止，交给ChatGPT审查数值验收含义。
