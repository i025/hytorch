# 阶段 4F：通用 dInput Gather 四路累加实验

基线 `e2d60c26cdb28facd4a41dd3b443c3148e397b66`；分支 `experiment/conv-dinput-kernel-optimization`。实验日期：北京时间2026-11-02，原始日志使用UTC 2026-11-01。只实现一个新CUDA C/C++ Kernel，显式 `fused_opt` 启用，默认和4E Auto规则不变。

**正确性通过；尚未证实相对旧 Gather 的稳定端到端训练收益。** 完整dInput墙钟有41/124形状加速、83/124退化。第二个匹配窗口中，H/F吞吐变化为DonkeyNet −0.51%、SpatialNet +0.27%，均小于独立进程波动。H/A累计+12.90%/+12.10%包括已有Native、SGD、剪枝和旧Gather的贡献，不能归为本轮Kernel的独立收益。

纯设备Nsight已补齐三次独立重复：98形状更快、26形状更慢；26份完整Trace解析 **PASSED**，未关联Kernel为0。未插桩算子及训练测试也均完成三次重复。GPU被其他任务重新占用时停止本次采集；先前GPU2的22份部分Trace作为独立辅助数据保留，不混入最终GPU1配对统计。

## 文件与状态

| 项目 | 状态及实测范围 |
|---|---|
| 主机回归 | 67 passed / 0 failed / 0 skipped |
| CPU源函数体抽样 | 124形状，32,076条逐位置比较通过；不代表GPU执行 |
| SM80编译、动态库与ABI | PASSED，新旧Kernel独立导出 |
| GPU回归 | 最终446 passed / 0 failed / 0 skipped |
| 未插桩跨形状测量 | PASSED；历史100+全新24形状；每方法5预热、30记录、3进程 |
| 匹配训练窗口01、02 | 各60进程、15,360个正式Step；全部数值对齐通过；不合并两个窗口的比值 |
| Nsight归因 | PASSED；GPU1同窗口26份完整导出，纯Kernel三次重复，未关联Kernel为0 |
| NCU实际Occupancy/访存计数器 | NOT_RUN；未更改权限或调用sudo |
| 新方案稳定训练收益 | NOT_ESTABLISHED |

阅读顺序：[设计依据](design-decision.md)、[源码与正确性问题分析](source-analysis.md)、[全部命令](commands.md)、[数据索引](data-index.md)。全部124个几何和逐形状正负结果见[合并表](results/report-01/all-shapes.csv)；摘要标记不替代原始数据。

## 唯一改动及通用性

旧Kernel每线程一个NCHW dInput元素，Cout沿一条FP32 FMA链顺序累加。新 `conv_dinput_opt.cu` 保留线程映射、Kh/Kw循环、访存方向及256线程block，把Cout分成四条独立FP32 FMA链，随后用FP32树形加法合并，尾通道用原FMA处理。目的是增加指令并行度；没有新增第二种优化方案。

全部几何来自N/Cin/Cout/H/W/OH/OW/Kh/Kw/stride/pad参数，生产代码没有模型名、层号或固定网络尺寸判断。支持连续FP32 NCHW/OIHW、groups=1、dilation=1，每轴stride 1/2、padding 0/1，以及合法矩形kernel、非4倍数Cout和边界输出；沿用既有尺寸检查。未实现FP16/BF16或其他布局。

旧 `conv_dinput.cu`、4E选择器、SGD、GEMM、Pooling、Forward、dWeight、dBias数学实现均未修改。`create_graph=True`仍走原可微路径；`requires_grad=False`首层剪枝先于Native调用。沿用ctypes动态库及当前CUDA Stream，不使用RawKernel代替新Native实现。

编译器资源：[原始build.log/build.json](raw-records/initial-01/conv_dinput_opt_build_01/)、[资源CSV](results/kernel-resources.csv)。旧/新分别40/48寄存器，stack、local、spill、shared均0字节。寄存器增加20%，可能影响驻留；**这不是实测Occupancy，也没有NCU证据把退化归因于某个硬件瓶颈**。

显式使用：

```python
from hytorch.backends.cuda import backend
with backend(matmul="shared", conv="im2col", conv_backward="fused_opt"):
    loss.backward()
```

训练CLI为 `--conv-backward fused_opt`。不支持的显式输入按既有契约报错，不新增静默fallback。`original`、`fused`、`auto`均保留，Auto不选择本轮新Kernel。

## 环境、设备与测量边界

Ubuntu20.04.6，双AMD EPYC7763、256逻辑CPU；A100-PCIE-40GB。驱动535.247.01；CUDA Toolkit/nvcc12.2.140，CuPy运行时12.2，PyTorch构建CUDA12.1/cuDNN9.1.0；GCC9.4；独立项目虚拟环境Python3.8.10、NumPy1.24.4、CuPy12.3.0、PyTorch2.4.1+cu121。详见[完整系统和包记录](environment/system-info.log)。未安装新框架或更改共享环境。

最终正确性、未插桩微基准和两个正式训练窗口都使用物理 **GPU2**，UUID `GPU-158426a4-6684-7863-8cbe-ed6595f97683`，以UUID设置 `CUDA_VISIBLE_DEVICES`，进程内逻辑设备0。最终26份Nsight使用另一空闲窗口的物理 **GPU1**，UUID `GPU-c6683db1-d2aa-97b9-c1b4-e497bd7ff0c5`，逻辑设备0，UTC17:46:44–17:51:48。纯Kernel比值在GPU1内部配对，训练比值在GPU2内部配对，不跨卡混算。GPU2的部分Trace及GPU0/6中断尝试另留档。

启动要求无计算客户端、0%采样利用率、显存≤4MiB；持续监控外部进程，进入后只停止本轮进程。0%利用率但有客户端不算空闲。既有guard对Nsight退出进程出现归属判断边界，4F脚本增加父进程归属和Linux退出状态判断；原始误判/真实外部进入均保存，不改4E脚本。没有独占GPU锁、固定频率或CPU隔离；完整监控见[13个窗口记录](results/hardware-02/windows.json)及[294个子命令采样区间](results/hardware-02/commands.csv)。最终Nsight窗口222个采样无外部客户端标记，整卡采样峰值524MiB；训练窗口峰值569MiB。它们不是精确进程峰值，零采样的子命令不插值。

成功采集的最外层执行会话返回143、未生成日志包装metadata，原因未确定；保留[独立记录](logs/wrapper-exit-06.json)，不补造外层成功状态。内部guard返回0、52个采集/导出子命令全部成功回写完整argv和输出，26份SQLite解析及数值断言均通过；完整性依据这些内部记录，不把外层退出码写成0。

三个计时口径独立保存：

1. **纯设备Kernel**：Nsight CUPTI activity duration，correlationId→提交API→父NVTX。预分配gx，3进程×30条/形状/新旧方法，全部已补齐。
2. **完整dInput**：已在GPU上的gy/W开始，Python检查、分配、ctypes提交、尾事件同步都计入墙钟；H2D和参考计算在外。CUDA Event记录的是stream跨度，含主机提交间隙，不能称纯Kernel时间。另有预分配/预绑定ABI范围，仍保留这个区别。
3. **训练**：正式未插桩吞吐使用完整训练epoch墙钟，排除进程启动和预热，包括数据准备、H2D、训练、Loss记录。Forward/Backward/Update在单独phases进程同步计时，不能加总替代throughput成绩。Nsight另开进程，不与普通吞吐混用。

## 正确性、原始失败与修复

原有卷积容差 `atol=1e-5, rtol=1e-4` 保持不变；SGD及网络测试沿用各自既有容差。首次真实GPU运行 **52 failed / 394 passed**：

- 48项是旧测试helper只接受标量stride/pad，本轮独立集合引入tuple导致测试代码TypeError。只在新测试中加入项目既有pair处理，不改生产计算。
- 4项是c23/v19在两个Stream上原Forward与默认cuDNN参考不一致，尚未执行到新dInput。独立CPU FP64诊断：c23 CuPy误差2.6154e-7、PyTorch原生FP32误差2.9309e-7、cuDNN误差0.00508193；v19分别3.0325e-7、3.4860e-7、0.00712423。TF32已关闭，cuDNN内部原因本轮未确定。

本轮124形状全部统一使用PyTorch原生CUDA FP32作为正确性参考，未按形状选取有利参考，未调整容差。计时PyTorch及训练E仍为正常cuDNN，保留实际输出与原生参考差异。第二次GPU回归因外部进程进入GPU0中止；第三次固定GPU2，**446/446通过，无跳过**。失败输入、输出、XML、stdout/stderr及FP64诊断都在[原始文件索引](raw-records/initial-01/file-index.csv)。生产Kernel未因这些测试问题而修改。

覆盖历史100+独立24形状、default/nondefault Stream、FP32尾通道、矩形空间/kernel、每轴stride/pad、多次反向、dWeight/dBias、首层保留/剪枝、两个CNN三步训练、普通/Fused SGD及高阶求导。124形状中的新鲜集合在实现和计时前冻结，未用于调优四路方案。

GPU正确性CSV中dInput最大绝对误差5.9604645e-7、相对L2最大3.0114928e-7；dWeight最大绝对误差1.3753772e-5；dBias最大6.1035156e-5，相对L2最大8.7530203e-8。后两者在原相对容差内，且并未修改其实现。全部断言通过，而非仅根据最大绝对值判断。

两个256step训练窗口各有1248项Loss/最终权重比较，最大绝对误差均4.1909516e-8；相同输入及初始权重用实际数组数值比较确认，不计算哈希。H的第1重复Loss：DonkeyNet 0.0007207735→0.0001886905，SpatialNet 0.0475530513→0.0037775042；各步完整数值保留，不只首尾。合成数据用于执行与性能诊断，不是真实驾驶训练结果。

## 跨形状性能：全体结果

未插桩36个独立进程，累计66,960条正式单次耗时；每形状/方法/范围5预热、30测量、3独立重复。完整操作四方法，预分配范围比较两种Native。原始Event、主机提交、尾同步、墙钟、内存和数值误差见[微基准数据](results/micro-01/)及[逐进程原记录](raw-records/initial-01/conv_dinput_opt_micro_01/results/)。

下面纯Kernel为GPU1三次Nsight重复，完整墙钟为GPU2三次未插桩重复；**同一行不把不同窗口数值相减推导主机成本**。

| 集合 | 纯Kernel更快/更慢 | 纯Kernel旧/新中位数 | 完整墙钟更快/更慢 | 完整墙钟旧/新中位数 |
|---|---:|---:|---:|---:|
| 历史100 | 79 / 21 | 1.170× | 28 / 72 | 0.971× |
| 独立24 | 19 / 5 | 1.114× | 13 / 11 | 1.003× |
| 全部124 | 98 / 26 | 1.159× | 41 / 83 | 0.980× |

按均值大小计数，不把接近1的形状全部认定为统计显著。83形状满足三个纯Kernel进程均值范围分离且旧/新≥1.05，作为描述性稳定性条件，不替代统计检验。`process-summary.csv`记录每进程30次的均值/标准差/Median/P95；`summary.csv`汇总三个进程均值，其P95不是把90条调用合并的P95。全部原始调用保留；新方案完整墙钟相对CuPy为98形状更快、26更慢，不等于相对旧Gather同样有收益。

| 例子 | NCHW / OIHW；stride,pad | 旧/新纯Kernel μs | 旧/新完整墙钟 μs | 墙钟旧/新 |
|---|---|---:|---:|---:|
| c00 | 1,3,7,9 / 16,3,1,1；1,0 | 8.369 / 7.212 | 67.294 / 71.504 | 0.941× |
| c12 | 4,32,25,27 / 64,32,3,3；1,1 | 187.653 / 150.820 | 237.797 / 209.649 | 1.134× |
| c15 | 16,64,17,19 / 64,64,3,3；2,0 | 253.117 / 385.453 | 297.521 / 450.100 | 0.661× |
| c31 | 4,32,33,31 / 64,32,5,5；2,1 | 302.351 / 441.272 | 342.503 / 505.892 | 0.677× |
| o11 | 16,64,7,19 / 64,64,5,5；2,1 | 264.868 / 399.541 | 313.046 / 468.326 | 0.668× |

独立集合中的o11也明显退化，不能用历史集合中的成功形状证明通用加速。新旧均为一次Kernel、一个gx分配，无gcol/额外布局Copy；本轮没有进一步减少Launch或中间张量。372条完整操作内存记录/方法均为1次malloc请求、0临时峰值字节（输出以外）；预分配范围均为0请求。实测内存请求和完整操作Kernel类别/次数分别见[memory.csv](results/micro-01/memory.csv)和[micro-kernels.csv](results/profiles-01/micro-kernels.csv)。

## 匹配训练和长期性能进展

两个模型均固定Batch4、FP32、SGD lr0.001、seed42、Dropout0、TF32关闭；DonkeyNet输入[4,3,66,200]，独立SpatialNet输入[4,3,16,16]。相同初始权重、输入和数据顺序；10step预热后恢复权重，每配置3独立进程×256step、轮换顺序。A是**当前代码复现的无Native CUDA C基线**，并非重跑历史最初Commit。A的动态库路径故意不存在、dispatch为空，Nsight也未见项目Native Kernel；CuPy内部cuBLAS不算本项目Native。

- A：CuPy+普通SGD，无剪枝。
- D：shared GEMM / im2col Forward / pool cupy / tile16 + Fused SGD + 首层剪枝。
- F：D+旧Gather；H：D+新四路Gather；E：PyTorch CUDA/cuDNN。

第一窗口波动较大（例如DonkeyNet D进程间CV23.97%）。预先记录[完整重复协议](results/repeat-training-protocol.json)，保持配置不变，第二窗口不并行运行CPU数据分析，再完整重复五组、两个模型、两种计时模式。没有只挑表现好的单组重跑，也不把两窗口最优值拼在一起；CPU共享调度对首轮波动的因果影响未独立验证。

**第二完整窗口作为下表口径**，GPU2、UTC17:11:02–17:14:27。samples/s为三进程算术均值±样本标准差；墙钟为训练完整epoch除256的均值。原始三次训练秒数全部保留在[training-raw.csv](results/training-02/training-raw.csv)，完整Step和Loss在同目录。

| 模型 | 版本/内容 | samples/s ± SD | 完整 ms/step | 相对前一配置吞吐变化 | 相对A吞吐 | 达PyTorch吞吐 |
|---|---|---:|---:|---:|---:|---:|
| DonkeyNet | A CuPy/普通SGD | 645.501 ± 10.808 | 6.197893 | — | 1.0000× | 42.73% |
| DonkeyNet | D Combined/融合更新/剪枝 | 682.436 ± 24.972 | 5.866518 | +5.72% | 1.0572× | 45.18% |
| DonkeyNet | F 加旧Gather | 732.455 ± 39.809 | 5.472158 | +7.33% | 1.1347× | 48.49% |
| DonkeyNet | H 加新Gather | 728.750 ± 19.226 | 5.491433 | **−0.51%** | **1.1290×** | **48.24%** |
| DonkeyNet | E PyTorch | 1510.647 ± 29.138 | 2.648536 | — | 对照 | 100% |
| SpatialNet | A CuPy/普通SGD | 1048.084 ± 18.100 | 3.817251 | — | 1.0000× | 40.52% |
| SpatialNet | D Combined/融合更新/剪枝 | 1135.382 ± 7.472 | 3.523146 | +8.33% | 1.0833× | 43.89% |
| SpatialNet | F 加旧Gather | 1171.803 ± 7.446 | 3.413634 | +3.21% | 1.1180× | 45.30% |
| SpatialNet | H 加新Gather | 1174.919 ± 13.099 | 3.404774 | **+0.27%** | **1.1210×** | **45.42%** |
| SpatialNet | E PyTorch | 2586.617 ± 176.202 | 1.551377 | — | 对照 | 100% |

本轮未单测B/C，因此D/A包含多个历史优化，不能把D/A分拆成B/A、C/B等未经本轮实测的增量。[全部比值及墙钟口径](results/training-02/speedups.csv)：

| 指标 | DonkeyNet | SpatialNet |
|---|---:|---:|
| H/F 吞吐变化 | −0.5058% | +0.2658% |
| H/D 吞吐变化 | +6.7866% | +3.4822% |
| H/A 累计吞吐变化 | +12.8969% | +12.1016% |
| H/E 吞吐比 | 48.2409% | 45.4230% |
| F墙钟/H墙钟（新相对旧耗时加速比） | 0.99649× | 1.00260× |
| H相对F墙钟变化 | 增加0.3522% | 下降0.2595% |
| A墙钟/H墙钟（累计耗时加速比） | 1.12865× | 1.12115× |
| H相对A墙钟下降 | 11.3984% | 10.8056% |

吞吐提升、墙钟下降和加速比独立计算；均值的倒数不等于倒数的均值。E约为H吞吐的2.073×/2.202×，本轮仍有明显差距。不能跨历史窗口连乘百分比。

**首个较嘈杂窗口完整保留**：

| 模型 | A samples/s ±SD | D | F | H | E | H/F吞吐变化 |
|---|---:|---:|---:|---:|---:|---:|
| DonkeyNet | 635.073±117.364 | 706.268±169.269 | 792.567±102.760 | 780.736±105.981 | 1417.980±34.433 | −1.49% |
| SpatialNet | 1053.590±138.707 | 1118.700±130.885 | 1020.214±64.490 | 1018.266±121.259 | 2411.150±477.763 | −0.19% |

见[首轮所有数值](results/training-01/)，不删除首轮负结果。两窗口均不支持“新Kernel使训练稳定加速”的结论。

单独phases进程的均值（ms/step）如下，插入阶段同步，不能与上表训练墙钟混算：

| 模型 | F 前向/反向/更新 | H 前向/反向/更新 | E 前向/反向/更新 |
|---|---:|---:|---:|
| DonkeyNet | 1.718849 / 3.101156 / 0.131235 | 1.701055 / 3.067546 / 0.129037 | 0.672465 / 1.645667 / 0.172359 |
| SpatialNet | 1.126309 / 1.926042 / 0.091765 | 1.102292 / 1.872338 / 0.088921 | 0.426610 / 0.774680 / 0.158613 |

原始各步及3次进程波动见[training-summary.csv](results/training-02/training-summary.csv)。阶段均值中的小幅缩短不替代未插桩吞吐结论。

## Nsight证据和训练没有稳定加速的原因

26份完整Trace使用CUDA API correlationId归因，未关联Kernel为0；不按异步时间重叠猜测父算子。训练A/F/H各两个模型16step，微基准完整路径3条/形状，纯Kernel三次30条/形状。没有对每个训练小算子插入同步。完整训练Trace来自GPU1独立插桩窗口，以下仅作归因，不计算GPU2的正式训练加速。

| 模型/层（仅报告标识） | F旧Gather μs/call | H新Gather μs/call | 旧/新 |
|---|---:|---:|---:|
| DonkeyNet conv2 | 266.370 | 393.098 | 0.678× |
| DonkeyNet conv3 | 164.239 | 240.697 | 0.682× |
| DonkeyNet conv4 | 78.013 | 66.692 | 1.170× |
| DonkeyNet conv5 | 66.538 | 60.112 | 1.107× |
| SpatialNet conv2 | 40.636 | 35.756 | 1.136× |
| SpatialNet conv3 | 16.260 | 12.334 | 1.318× |

层名仅用于结果归因，生产Kernel不读取它们。DonkeyNet前两处必要dInput明显变慢，四处dInput合计575.159→760.600 μs/step（+32.24%）；SpatialNet合计56.896→48.090 μs/step（−15.48%）。因此“多数形状更快”不代表这两个模型的热点组合都获益。

| Nsight训练范围 | DonkeyNet F / H | SpatialNet F / H |
|---|---:|---:|
| Backward Kernel次数/step | 66 / 66 | 41 / 41 |
| Backward累计Kernel μs/step | 1106.542 / 1292.330 | 327.132 / 318.853 |
| Backward主机NVTX μs/step | 4946.017 / 4960.254 | 3016.168 / 2959.970 |
| Launch API主机 μs/step | 523.583 / 504.633 | 299.906 / 315.788 |
| 自己Kernel跨度内的间隙 μs/step | 3734.266 / 3566.066 | 2591.199 / 2543.285 |
| Backward CuPy Copy次数/step | 10 / 10 | 6 / 6 |
| Backward Col2im次数/step | 0 / 0 | 0 / 0 |

第一层剪枝均保留，F/H Gather调用分别每16step为64/64（DonkeyNet）和32/32（SpatialNet）。本轮只改变已有Kernel内部累加，**不减少F→H的Launch、Copy或中间分配**。Copy仍来自未修改的其他路径。各类GPU Kernel、父节点、主机API及MemoryHook请求分别见[profiles原始汇总](results/profiles-01/)和[每step归一表](results/report-01/)。

实测支持：小形状dInput主机提交约数十μs，常明显高于设备Kernel；例如c00旧/新提交47.631/52.922 μs，纯Kernel为8.369/7.212 μs，但完整墙钟67.294/71.504 μs。不能把跨窗口差值当精确Python耗时，却足以说明“设备更快”不自动等于“完整操作更快”。四路包装、输入检查和提交范围未消失。

Nsight中Backward主机范围和自己的设备Kernel间隙仍明显大于Kernel累计时间，SpatialNet新方案每步仅省约8.8μs dInput设备时间，不能据此保证毫秒级训练显著加速。间隙是本次归因Kernel之间的区间，不是SM空闲率或GPU利用率；API主机时长也不是设备执行时间。这里没有CPU采样，不能把全部间隙定量归给Python。

本次throughput Trace的Backward NVTX范围内没有同步API或Memcpy记录，不代表整步训练没有同步/传输：Loss读取、H2D、尾同步及初始化/预热分别位于其他阶段或outside_phase。CUDA设备Memcpy与CuPy布局Copy Kernel也分别统计，不混用次数。

寄存器增加、编译指令调度和访存复用是退化的候选原因，未获NCU实际计数器证明；不把假设写成根因定论。已确认的是形状相关设备退化、未减少主机调用/Launch，以及在共享服务器上的端到端波动。

## 结论与边界

本轮优化Cout依赖链，数值正确、几何通用，但收益依赖形状，完整操作与训练没有普遍稳定收益。建议仅保留显式可选研究后端用于复现和对照，**不提升为默认，不交给Auto**。没有证据支持全面替换旧Gather。

尚未解决：cuDNN两形状Forward参考差异的内部原因、实际Occupancy/缓存/指令延迟归因、共享CPU调度噪声及最外层日志包装143的原因。纯Kernel已补齐三重复；GPU2原先两重复的辅助结果在[profiles-partial-01](results/profiles-partial-01/)独立保留，不冒充同一GPU1窗口。未来方向需人工审查决定，本轮不新增第二个Kernel或进入下一阶段。

未做哈希/指纹校验，所有训练均带 `--no-fingerprints`；不更改历史实验、系统配置、GPU权限或其他用户任务。不合并main。提交和推送本分支后停止。

## 实测图像

全部由CSV、Nsight导出或编译资源报告生成，明确标为数据可视化，不是终端截图。图像及逐文件来源见[图像索引](data-index.md#结果图)。
