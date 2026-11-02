# 阶段 4G：通用 Conv2d dWeight Split-K Implicit GEMM

2026-11-02，基线 `adaaa3cc4489e3ff86e7960e3290af381e61f8c8`，分支 `experiment/conv-dweight-fusion`。本轮只实现一个 dWeight 算法；没有开始4H，没有改变默认后端、既有dInput、Fused SGD或自动求导公式。

**GPU验收完成。** CPU回归76项、新增GPU277项、既有GPU回归446项均通过，最终无失败、无跳过。132种形状的算子测量及DonkeyNet/SpatialNet各三次256步训练完成，17份Nsight Systems归因完成。正式结果使用[acceptance-03](results/acceptance-03/summary.json)，全部原始记录见[数据索引](data-index.md)，命令见[commands.md](commands.md)。首次失败窗口和离线分析修正记录完整保留。

本轮同GPU0匹配窗口中，I/F训练吞吐提升 **10.64% / 9.17%**，I/A累计提升 **23.77% / 24.34%**（DonkeyNet / SpatialNet）。纯dWeight设备核时99种形状变快、33种变慢；完整同步墙钟129种变快、3种变慢。值得保留为显式可选后端，尚不能默认替换所有形状。

## 算法与支持范围

原路径为Im2col、布局物化、CuPy tensordot/cuBLAS。新方案将 `M=Cout, P=Cin*Kh*Kw, R=N*OH*OW` 视为矩阵乘法：16×16线程块协作载入shared tile，按卷积几何隐式读取输入，每个加载值参与16个输出。R按128项Split-K生成部分和，再由第二个CUDA Kernel以8个归约lane及共享内存树形合并。两个Kernel属于一个算法，不是两个候选优化。

新计算位于 [`conv_dweight.cu`](../../../hytorch/backends/cuda/conv_dweight.cu)，通过现有动态库、ctypes和当前CUDA Stream调用，未使用CuPy RawKernel、ElementwiseKernel或cuDNN替代。没有完整Im2col，没有全局原子，使用FP32 `__fmaf_rn` / `__fadd_rn`。较短累加链改变舍入顺序，不保证逐位一致。

支持当前设备上的连续FP32 NCHW x/gy、OIHW输出、groups=1、dilation=1、合法正整数kernel/stride和非负padding，沿用INT_MAX元素数限制。本轮覆盖Batch1/4/16，Cin3/16/32/64及边界通道，Cout16/32/64及边界通道，1×1/3×3/5×5及矩形kernel，stride每轴1/2、padding每轴0/1、矩形及非整块空间。不宣称穷尽所有合法参数。形状完整清单见[predeclared-shapes.json](results/predeclared-shapes.json)。生产代码没有模型名、层号或固定网络尺寸分支。

显式启用：`backend(conv_dweight="fused")`；训练CLI为 `--conv-dweight fused`。默认仍为original。普通一阶反向使用新算法；`create_graph=True`保留既有可微路径；不支持的dtype/layout明确报错，不静默复制或fallback。dWeight与首层dInput剪枝独立，dBias和Forward不变。详见[设计](design-decision.md)、[源码分析](source-analysis.md)。

## 环境与GPU生命周期

| 项目 | 实测 |
|---|---|
| 系统 / CPU | Ubuntu20.04.6，2×AMD EPYC7763，256逻辑CPU |
| Python / NumPy / CuPy | 3.8.10 / 1.24.4 / 12.3.0（现有项目独立`.venv`） |
| PyTorch / cuDNN | 2.4.1+cu121 / 9.1.0；TF32关闭 |
| 驱动 / Toolkit / nvcc | 535.247.01 / CUDA12.2 / 12.2.140 |
| CuPy运行时 | 12020（CUDA12.2）；Torch发行包CUDA12.1 |
| GCC / Nsight Systems | 9.4.0 / 2023.2.3 |
| 物理GPU / UUID | A100-PCIE-40GB，GPU0，`GPU-f099fdbd-adf2-6157-3462-4145bc2a8359` |
| CUDA映射 | `CUDA_VISIBLE_DEVICES=GPU-f099fdbd-adf2-6157-3462-4145bc2a8359`，逻辑0，PCI `0000:01:00.0` |

CPU准备、代码、回归、编译、277项测试收集和76项统一任务计划均先完成。选择时GPU0显存0MiB、利用率0%、无计算进程；未发现Slurm/PBS命令或分配环境。监控再次确认空闲后才启动，没有占用其他七张卡。`CUDA_VISIBLE_DEVICES`不被解释为独占租约。

实际有**两个窗口**，不能写成整个阶段只有一次：

| 窗口 | UTC开始 → 结束 | 状态 |
|---|---|---|
| 首次01 | 04:40:36.917819 → guard退出04:40:50.033992 | 276通过、1测试断言失败；释放接口另报错；后续GPU任务NOT_RUN |
| 用户明确批准的一次完整补测02 | 04:43:59.662666 → 04:51:14.983436 | 76/76任务完成，`cudaDeviceReset=0`，统一释放 |

第二窗口北京时间为12:43:59.663–12:51:14.983。两次均为同一GPU0，设备切换0次；完整正确性、算子、48组训练和17份Trace都在第二个连续窗口完成。正常初始化的父进程context跨子任务保留，没有忙循环或显存占满。监控329次无外部进程冲突；整卡采样峰值1786MiB，这是约1.2秒间隔的采样峰值，不是精确进程峰值。结束后只读检查为0MiB/0%（[记录](environment/gpu-release-02.log)）。随后导出SQLite、分析、绘图和Git操作均在`CUDA_VISIBLE_DEVICES=''`下进行。

首次窗口因异常未写出window.json，无法给出精确显式释放时刻；04:40:49.032436监控已无本轮计算进程，由进程退出释放。用户随后明确回复“允许一次 GPU0 完整补测”。完整错误及修复见[window-01-failure.json](results/window-01-failure.json)，不隐去第一次失败。

## 正确性和编译

| 项目 | 实测结果 |
|---|---|
| CPU回归 | 最终76 passed / 0 failed / 0 skipped；此前74、75项的输出保留 |
| CPU算术模型 | 132形状、33,066个抽样比较，最大绝对误差1.3113021851e-5；FP32 FMA/split/reduce算术模型，不是CUDA线程模拟或全量GPU证明 |
| 新GPU测试 | 277 passed / 0 failed / 0 skipped（10.71s） |
| 既有GPU回归 | 446 passed / 0 failed / 0 skipped（17.18s） |
| 新GPU逐值比较 | 3831组比较、0超差、均有限；包含默认/非默认Stream、反复调用、全梯度、剪枝/保留输入梯度、普通/Fused SGD、两模型3步及高阶路径 |
| 其中dWeight | 528组；max abs 1.4305114746e-6，max relative L2 6.4848674736e-7 |
| 其中dInput / dBias | max abs 3.5762786865e-7 / 6.103515625e-5；均通过既有绝对+相对容差；dBias较大绝对误差对应较大参考值，未放宽阈值 |
| 算子Benchmark数值检查 | 2376组对照、0超差，最大绝对误差1.5884637833e-5 |
| 完整训练一致性 | 48组、12,288步、780组Loss/最终权重对照全部通过；max abs 4.8894435167e-8 |

算子容差沿用`atol=1e-5, rtol=1e-4`；训练Loss/更新后权重沿用`atol=1e-6, rtol=1e-5`。相同初始权重、输入逐数组比较，不做哈希或fingerprint；所有训练命令均为`--no-fingerprints`。FP32数值与PyTorch对齐，不要求逐位一致。

独立卷积正确性参考统一使用PyTorch native FP32（关闭cuDNN）。这是沿用4F已保存的c23/v19 cuDNN Forward诊断，避免把已知参考Forward异常归为dWeight错误；独立算子PyTorch计时、完整模型测试和训练仍使用标准cuDNN路径。全部比较对象和容差写入CSV，没有根据本轮胜负换参考。

无GPU阶段用`nvcc --arch=80`构建SM80+compute80 PTX；动态库符号、ABI1加载通过。`conv_dweight_tiles`为32寄存器/线程、2048B shared，`conv_dweight_reduce`为20寄存器/线程、1024B shared；均0 spill、0 stack、0 local。编译报告是静态资源用量，不代表实际occupancy；本轮未运行Nsight Compute，也未重新探测其计数器权限。原始build.json、build.log、资源报告均归档。

## 跨形状算子性能

每种实现独立进程、三次重复，每形状5次预热+30次正式调用，共35,640次普通计时；另有35,640次独立Nsight调用。原路径包含Im2col+tensordot全部工作，新路径包含工作区/输出准备、两个Kernel及调用；PyTorch为`torch.nn.grad.conv2d_weight`。同一seed42 FP32输入在各进程逐数组确认相同。

纯设备时间为API correlationId关联的CUPTI Kernel执行时间之和。CUDA Event跨度包含主机提交间隙，单列保存，不当作纯核时。完整同步墙钟包含Event记录、调用、末尾同步，主机提交和等待也单列。两类计时来自同一物理窗口内的独立进程，不混为一次测量。

| 指标（132形状） | 新路径对原路径 |
|---|---|
| 纯设备Kernel时间 | 99快 / 33慢；中位加速1.4118×，范围0.5563–5.8043× |
| 完整同步墙钟 | 129快 / 3慢；中位加速1.6760×，范围0.9548–2.6808× |
| 三个墙钟退化项 | h08 0.9849×、v19 0.9548×、n01 0.9935× |

胜负按三次进程均值比是否大于1统计，包含接近波动的差异，不等于每项都有统计显著性；全部逐次值和标准差保留。没有把正确性失败或不支持项计入胜出数，本轮132形状均通过数值检查。

以下为真实网络形状的dWeight，单位μs/call；模型/层名仅用于结果解释，没有进入运行时分支。

| 形状 | 原GPU | 新GPU | 原完整墙钟 | 新完整墙钟 | GPU加速 | 墙钟加速 |
|---|---:|---:|---:|---:|---:|---:|
| n00 Donkey Conv1 | 81.611 | 65.931 | 174.109 | 131.971 | 1.238× | 1.319× |
| n01 Donkey Conv2 | 128.106 | 132.428 | 193.195 | 194.459 | **0.967×** | **0.993×** |
| n02 Donkey Conv3 | 58.232 | 51.773 | 169.032 | 117.698 | 1.125× | 1.436× |
| n03 Donkey Conv4 | 36.401 | 24.772 | 161.418 | 92.024 | 1.469× | 1.754× |
| n04 Donkey Conv5 | 27.044 | 24.798 | 153.933 | 92.270 | 1.091× | 1.668× |
| n05 Spatial Conv1 | 115.878 | 19.964 | 236.314 | 88.149 | 5.804× | 2.681× |
| n06 Spatial Conv2 | 30.539 | 20.318 | 161.026 | 88.657 | 1.503× | 1.816× |
| n07 Spatial Conv3 | 29.763 | 20.173 | 160.850 | 88.165 | 1.475× | 1.824× |

[全部132形状及PyTorch](results/acceptance-03/micro/comparisons.csv)、[每次调用](results/acceptance-03/micro/latencies.csv)、[每次CUPTI核时](results/acceptance-03/micro/trace-calls.csv)、[三进程统计](results/acceptance-03/micro/summary.csv)。Conv2变慢仍保留；地址运算、边界检查、shared同步、部分和读写及小tile效率可能限制性能，尚无计数器数据区分这些原因。

## 中间数据与内存

旧col逻辑字节数为`4*R*P`，新部分和为`4*ceil(R/128)*M*P`，另需最终dW输出。129形状的新工作区小于旧col，3形状更大，最大为旧col的2.1333倍，不宣称每形状都节省内存。

独立、非计时内存测量中，CuPy MemoryHook记录一次完整操作的pool请求及存活量，PyTorch记录allocator峰值增量。跨全部形状的最大临时存活量为：原路径18,619,392B，新路径2,048,000B，PyTorch18,874,880B。它们是各形状最大值，不能作为同一形状的逐项节省百分比。逐形状见[memory.csv](results/acceptance-03/micro/memory.csv)，与整个GPU窗口的1786MiB整卡采样峰值是不同口径。

## 匹配训练与长期性能表

A为**当前代码复现的无Native CUDA C基线**，CuPy+普通SGD+未剪枝；不是历史最初commit原始成绩。A指定不存在的Native库，dispatch为空，完整Trace无我们的GEMM、Conv、Pooling、dInput、Fused SGD、dWeight Kernel；CuPy/cuBLAS内部CUDA不算本项目Native。

F为shared GEMM+im2col Forward+pool CuPy+tile16+Fused SGD+首层剪枝+原Gather dInput，未采用4F的fused_opt或Auto；I只在F上增加本轮fused dWeight；E为PyTorch CUDA。两个模型的初始权重、数据顺序相同，FP32、batch4、lr.001、seed42、dropout0、TF32关闭，10步预热后恢复初始状态。每配置三次独立重复、每次256步（1024samples），轮换顺序。

正式吞吐使用未插桩训练epoch墙钟，包含完整step、数据/传输及日志等开销；`steps.jsonl`中的内部step范围较窄，另列保存。同步phases与Nsight计时均不参与正式吞吐比。

| 模型/版本 | 吞吐samples/s，均值±样本SD | 256步墙钟s，均值±SD | 完整ms/step | 相对F | 相对A | 对PyTorch比值 |
|---|---:|---:|---:|---:|---:|---:|
| Donkey A | 763.83±8.07 | 1.34071±0.01414 | 5.23714 | — | 1.0000× | 55.88% |
| Donkey F | 854.44±5.90 | 1.19848±0.00829 | 4.68156 | 1.0000× | 1.1186× | 62.51% |
| Donkey I | **945.36±6.50** | **1.08322±0.00745** | **4.23133** | **1.1064×** | **1.2377×** | **69.17%** |
| Donkey E | 1366.80±93.71 | 0.75146±0.04969 | 2.93541 | — | 对照 | 100% |
| Spatial A | 1220.13±2.78 | 0.83926±0.00191 | 3.27836 | — | 1.0000× | 45.64% |
| Spatial F | 1389.67±15.84 | 0.73693±0.00843 | 2.87863 | 1.0000× | 1.1390× | 51.98% |
| Spatial I | **1517.12±5.52** | **0.67497±0.00245** | **2.63659** | **1.0917×** | **1.2434×** | **56.75%** |
| Spatial E | 2673.57±36.24 | 0.38305±0.00515 | 1.49631 | — | 对照 | 100% |

| 本窗口比较 | Donkey吞吐变化 / 墙钟加速 / 耗时下降 | Spatial吞吐变化 / 墙钟加速 / 耗时下降 |
|---|---|---|
| F/A（已有优化合计） | +11.86% / 1.1187× / 10.61% | +13.90% / 1.1389× / 12.19% |
| I/F（本轮增量） | **+10.64% / 1.1064× / 9.62%** | **+9.17% / 1.0918× / 8.41%** |
| I/A（累计） | **+23.77% / 1.2377× / 19.21%** | **+24.34% / 1.2434× / 19.58%** |
| I/E | 69.17%吞吐；完整step为E的1.4415倍 | 56.75%吞吐；完整step为E的1.7621倍 |

吞吐比采用三次samples/s均值之比，墙钟加速采用三次墙钟均值之比，二者因取平均次序略不同。所有比值来自本次同卡同窗口，未乘接历史百分比。B/C/D/H未在4G重测，不能拆分本表F/A的历史优化贡献；历史表保留在各阶段报告。

三次I吞吐均高于三次F，支持本窗口的稳定收益；每次训练不足1.4秒，且Donkey PyTorch样本SD较大，不能外推为长任务、其他GPU/模型/数据的确定增益。这里只是合成训练性能与数值一致性，不是真实自动驾驶精度。逐次训练、Loss、权重误差与进展表见[training](results/acceptance-03/training)。

## Backward与Nsight证据

独立同步phases测量（ms/step，均值±SD）：

| 模型/版本 | Forward | Backward | Update |
|---|---:|---:|---:|
| Donkey F | 1.4983±0.0266 | 2.7108±0.0746 | .1141±.0009 |
| Donkey I | 1.4902±0.0104 | **2.3418±0.0277** | .1101±.0003 |
| Spatial F | .9623±.0367 | 1.6504±.0652 | .0783±.0028 |
| Spatial I | .9392±.0070 | **1.3084±.0090** | .0751±.0005 |

Backward同步墙钟下降13.61% / 20.72%；这组有阶段同步开销，不代替未插桩吞吐。

以下是每份Trace16个正式step的统计，单位μs/step，保留各Trace预热/初始化在outside_phase中，不纳入这里：

| 指标 | Donkey F → I | Spatial F → I |
|---|---:|---:|
| dWeight Kernel数 | **24 → 10**（−58.33%） | **14 → 6**（−57.14%） |
| dWeight Im2col / Copy数 | **5/10 → 0/0** | **3/6 → 0/0** |
| dWeight cuBLAS GEMM / 新tiles+reduce | 5/0 → 0/10 | 3/0 → 0/6 |
| dWeight全部GPU核时 | 335.289 → 300.773（−10.29%） | 170.428 → 62.260（−63.47%） |
| dWeight主机NVTX累计范围 | 1320.759 → 616.172 | 822.384 → 402.695 |
| 整个Backward Kernel数 | 66 → 52 | 41 → 33 |
| 整个Backward GPU核时 | 1114.251 → 1077.675（−3.28%） | 325.795 → 222.364（−31.75%） |
| 整个Backward主机范围 | 4864.615 → 4497.330（−7.55%） | 3053.895 → 2560.333（−16.16%） |
| Backward launch API累计时间 | 483.173 → 418.070 | 307.273 → 241.739 |
| Backward自己Kernel之间的间隙 | 3652.531 → 3306.983 | 2632.888 → 2244.378 |

所有17份Trace的Kernel API关联未匹配数为0。I的每个dWeight父节点严格包含1个tiles+1个reduce，无Im2col/Copy/cuBLAS；Donkey80次、Spatial48次，与16step×层数相符。原路径每层4–5个Kernel，除GEMM外还可能有矩阵后处理，不一律当成5。

归因使用API correlationId及提交时的NVTX；PyTorch工作线程按提交时唯一主线程phase窗口归因，并显式标记。不是按异步GPU时间重叠猜父节点。间隙定义为该Backward所提交Kernel最早开始至最晚结束的跨度减区间并集，不等于整卡空闲或SM利用率。API时间不是Kernel执行时间；NVTX包含插桩开销，不与正常训练墙钟直接相减。GPU与主机结果共同支持减少提交、布局与间隙的解释，无法将10.64%吞吐收益精确分摊给各项。

PyTorch存在Kernel区间重叠：Donkey E每步Kernel时间之和1555.098μs，区间并集1316.805μs；Spatial E分别235.427/230.307μs。两者分开保存，不强行令其相等，也不把Kernel时间之和解释为物理设备占用率。[最终记录检查](results/publication-review.json)确认八份训练Trace中按关联API提交时间分配的Backward Kernel均无遗漏或重复。

## 失败、修复及限制

1. 首次GPU窗口geometry测试把裁剪后非连续gy传入，先触发正确的layout错误，却断言geometry错误。修复仅在测试中将错误几何张量物化为连续布局。276通过/1失败记录保留；最终277通过。生产Kernel和容差未变。
2. 控制器释放误用CuPy12.3不存在的`runtime.deviceReset`。改为ctypes绑定已配置Toolkit的`cudaDeviceReset`，CPU解析符号测试通过，最终真实返回0。首次只能依据退出和监控说明资源释放，不捏造精确显式时间。
3. 离线首版Backward窗口表只按NVTX本地父ID取Kernel，漏算PyTorch工作线程；原始phase-summary的correlation/phase归因是正确的，吞吐/数值结果未受影响。最终acceptance-03改为关联API提交时的唯一phase窗口，PyTorch每步66/24个Backward Kernel与phase统计一致。acceptance-01中E的窗口核时/间隙作废，旧文件保留。首次重汇总因日志拒绝覆盖退出（postprocess-02.log），修正日志目的地后在新目录完成，未重跑GPU。
4. 33种形状纯核时退化、3种墙钟退化，部分和工作区也有3形状变大。保持显式可选，不增加调度规则或新Kernel。
5. 没有实际occupancy/DRAM性能计数器；不能将地址运算、带宽或寄存器成本的推测写成已证实的退化根因。仅使用nvcc资源报告及Nsight Systems。

本轮没有修复CUDA数学Bug、没有放宽数值标准，没有改动4A–4F历史数据。下一步应先人工审查跨形状退化、PyTorch差距和较长训练稳定性；本轮不执行新的优化阶段。

## 图与完整记录

以下均由真实CSV/CUPTI生成并标注为数据可视化，**不是真实终端截图**：

1. [dWeight原路径/新路径/PyTorch延迟](figures/final-03/01-dweight-latencies.png)
2. [132形状加速和退化分布](figures/final-03/02-cross-shape-speedups.png)
3. [Im2col、Copy与Launch减少](figures/final-03/03-kernel-copy-launch-counts.png)
4. [两模型A/F/I/E吞吐及三次波动](figures/final-03/04-matched-throughput.png)
5. [本轮增量、累计收益及PyTorch差距](figures/final-03/05-incremental-and-cumulative.png)
6. [相对PyTorch吞吐水平](figures/final-03/06-pytorch-gap.png)
7. [Backward CPU/GPU时间线](figures/final-03/07-backward-timeline.png)
8. [独立同步阶段计时](figures/final-03/08-synchronized-phases.png)
9. [逐步Loss](figures/final-03/09-losses.png)
10. [旧col与新部分和工作区](figures/final-03/10-workspaces.png)

完整服务器原始根目录：`/home/xymo/projects/hytorch/runs/conv_dweight_window_02/`（规范路径前缀为`/srv/lab-disks/disk2/users/xymo/projects/hytorch/`）。17份`.nsys-rep`、SQLite、逐值NPZ和动态库留服务器，Git仅上传文本、CSV/JSON/XML、压缩原始日志及PNG。[file-index.csv](raw-records/file-index.csv)逐文件列绝对路径、字节数和Git记录，没有哈希、密码、模型权重或二进制构建产物。
