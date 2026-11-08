# 阶段4C：原生CUDA Col2im实现与A100验证

## 结论

指定算法已实现，默认仍为CuPy。原生kernel在真实五层布局的无profiler单算子事件测量中加速1.68–3.32×，但**7轮完整训练全部比CuPy慢**，均值0.831259→0.836731 s，退化约0.658%。不能宣称端到端加速，不默认启用native，不自行引入第二套算法。

200项GPU回归、32项host、memcheck/racecheck全部通过；同一pair最终loss、全部参数梯度和权重逐位相同。前轮FP32修复、GEMM128 CTA、前向和dWeight实现均保留。

## 修改前热点确认（不是旧FP64的215μs）

先读最新shape-aware dispatch的train_new trace：col2im **232.6826 μs/step**，占1556.4135 μs GPU kernel总计的14.95%，占backward880.7051 μs的26.42%，每步5次、每层1次。其占dInput340.5230 μs的68.33%；同期dWeight340.8753、dBias57.2996、激活反向73.7861、Linear反向39.7824 μs。故仍属主要热点，满足继续实现的前提。

五层FP32输入的布局如下，KH/KW即shape第3/4轴；元素strides由已实测byte strides除以4，均非连续：

| 层 | 之前最新col2im μs/step | col逻辑shape | 元素strides | SH,SW | PH,PW |
|---|---|---|---|---|---|
| conv1 | 59.6739 | [4, 3, 5, 5, 31, 98] | [3038, 303800, 60760, 12152, 98, 1] | [2, 2] | [0, 0] |
| conv2 | 94.3107 | [4, 24, 5, 5, 14, 47] | [658, 65800, 13160, 2632, 47, 1] | [2, 2] | [0, 0] |
| conv3 | 46.1442 | [4, 36, 5, 5, 5, 22] | [110, 11000, 2200, 440, 22, 1] | [2, 2] | [0, 0] |
| conv4 | 16.2625 | [4, 48, 3, 3, 3, 20] | [60, 2160, 720, 240, 20, 1] | [1, 1] | [0, 0] |
| conv5 | 16.2913 | [4, 64, 3, 3, 1, 18] | [18, 648, 216, 72, 18, 1] | [1, 1] | [0, 0] |


图像输出H/W分别conv1=66/200、conv2=31/98、conv3=14/47、conv4=5/22、conv5=3/20。最新trace的backward序列与此前FP32详细metadata逐项匹配名字/grid/block，共1470 kernels；标签来自匹配，duration取最新trace。raw/baseline_hotspots.json和baseline_backward_categories.json保存原始来源与校验方法。

## 实现与范围

新增col2im_strided/hy_col2im；runtime新增backend(col2im="cupy"|"native")和strided binding；functions_conv只在col2im_array加选择入口。默认cupy，显式FP64和不支持metadata走原CuPy，无静默降精度。

每个NCHW输出独立计算；先推导kh/kw合法上下界，再跳到stride同余类起点并按stride递增。oh/ow起点只算一次，内层逐次减1；stride1省略模和除法。按原kh/kw升序FP32累加，无atomic。传入6个真实signed int64元素strides，不pack输入；输出连续。输出、offset、步进防溢出由binding/宽索引保护。

原CUDA文件完整字节前缀不变，证明GEMM及其他既有kernel未修改。原CuPy _col2im_gpu、dWeight/前向/autograd类AST不变，FP32 functions.py SHA不变。完整代码、适用范围和精度行为见SOURCE_CHANGES.md。

## 正确性与Sanitizer

| 项目 | 实际结果 |
|---|---|
| 完整GPU回归 | 200 passed，无skipped |
| 主机回归 | 32 passed |
| memcheck | 67 passed，0 errors |
| racecheck | 67 passed，0 hazards/errors/warnings |
| 原CuPy与独立CPU scatter | 新增形状测试逐位相等 |
| PyTorch Fold及高阶梯度 | FP32/FP64、matrix/6D输入对齐 |
| 36个microbenchmark case | 输入不变，结果逐位相等 |
| 7轮训练 | loss、最终全部参数梯度和权重逐位相等 |

测试覆盖零/负/非单位strides、非方形核、非对称stride/pad、无贡献输出、非默认stream、连续调用及预分配out。完整日志和统计在CORRECTNESS.md及raw/。

## 单算子：同进程配对

固定seed501+case_id；相同col、输出预分配，10warmup，15次正反交替。CUDA event包围32次同算子CUDA Graph回放后除以32，排除H2D/D2H和逐算子Python提交间隙。原CuPy的ElementwiseKernel表达式由原函数AST逐字读取，仅诊断调用复用预分配out；每case再与原生产函数验证逐位一致。

| 层，真实非连续布局 | CuPy event μs | Native event μs | 加速 | CuPy直接CUPTI μs | Native直接CUPTI μs |
|---|---|---|---|---|---|
| conv1 | 50.880 | 15.616 | 3.258× | 59.413 | 18.880 |
| conv2 | 70.304 | 22.016 | 3.193× | 94.379 | 29.259 |
| conv3 | 32.320 | 9.728 | 3.322× | 45.792 | 14.592 |
| conv4 | 9.824 | 5.856 | 1.678× | 14.859 | 9.152 |
| conv5 | 7.392 | 3.104 | 2.381× | 14.517 | 6.560 |


同一几何还测连续和负stride布局；另有重叠、不重叠、stride3有空洞、非对称pad、不规则尺寸、不同N/C等共36case，CSV共72行。独立直接调用CUPTI各3次用于确认kernel时间/配置；不是15轮事件统计的替代。Graph热缓存事件与直接调用/profile/训练不是相同缓存与提交条件，绝不跨列计算加速。

实际native kernel为col2im_strided，block256、38 registers/thread、0 shared bytes；旧CuPy col2im为block128、32 registers/thread、0 shared。每case实际grid见CSV observed_signature。没有测occupancy、带宽或stall计数器。

### 完整披露退化case

| case | layout | CuPy μs | Native μs | Native延迟增加 |
|---|---|---|---|---|
| holes | training | 2.624 | 3.168 | 20.73% |
| holes | contiguous | 2.496 | 3.168 | 26.92% |
| holes | negative | 2.560 | 3.168 | 23.75% |
| tiny | training | 2.272 | 2.336 | 2.82% |
| tiny | contiguous | 2.272 | 2.336 | 2.82% |


共31/36 case中位延迟改善，5/36退化。holes的KH=KW=1、stride3，多数输出无贡献，native较旧版本慢约20.7–26.9%；tiny几何两个布局约2.8%退化。宽索引/一般化边界计算和launch粒度可能在轻量shape上占优劣，但这是源码支持的假设，未用硬件计数器确认。本轮不增加shape特例或第二套优化。

## 端到端：唯一变量col2im backend

保持相同FP32修复、同一编译库、GEMM128 CTA和所有其他backends：shared/im2col/native。固定256合成样本、batch4、3epochs/192steps、seed42、SGD .001、dropout0；每模式10warmup，7轮交替，每轮恢复相同权重。

wall包括CPU DataLoader、pageable H2D、forward/backward/SGD和最终同步；无逐步日志、loss read或sync，参数/梯度快照在计时外。两组phase NVTX相同。

| 实现 | wall s，mean ± sample SD | samples/s | 最终loss |
|---|---|---|---|
| cupy | 0.831259 ± 0.003858 | 923.92 | 0.000255663209828 |
| native | 0.836731 ± 0.003305 | 917.87 | 0.000255663209828 |


| 配对轮次 | CuPy wall s | Native wall s | CuPy−Native ms（负为退化） |
|---|---|---|---|
| 0 | 0.838753 | 0.842969 | -4.216 |
| 1 | 0.833651 | 0.839393 | -5.742 |
| 2 | 0.830230 | 0.835010 | -4.780 |
| 3 | 0.828713 | 0.834436 | -5.723 |
| 4 | 0.828393 | 0.836115 | -7.721 |
| 5 | 0.827845 | 0.835546 | -7.701 |
| 6 | 0.831229 | 0.833647 | -2.418 |


均值差CuPy−Native为-5.472 ms/192steps，配对差样本SD=1.895 ms，描述性t区间（df6，95%）为[-7.224,-3.719] ms；7对均为负。不是只比较两组独立SD。该区间只描述当前窗口，不能保证其他设备/负载下同样比例。

14次最终权重hash一致，loss=0.0002556632098276168；梯度最大绝对差0，权重最大绝对差0。未运行真实数据集准确率实验。

## 同窗口训练profiling（μs/step）

| backend | GPU kernel总计 | col2im | backward GPU | forward GPU | SGD GPU | backward CPU elapsed | launch API CPU elapsed |
|---|---|---|---|---|---|---|---|
| cupy | 1456.193 | 217.604 | 822.751 | 520.066 | 113.376 | 2949.548 | 599.861 |
| native | 1410.897 | 85.847 | 734.722 | 557.704 | 118.471 | 3056.474 | 628.768 |


| 层 | 本轮CuPy训练col2im μs | 本轮Native训练col2im μs |
|---|---|---|
| conv1 | 55.763 | 20.099 |
| conv2 | 88.515 | 31.590 |
| conv3 | 43.120 | 15.894 |
| conv4 | 15.226 | 10.906 |
| conv5 | 14.979 | 7.357 |


本轮CuPy trace测得col2im **217.6035 μs/step**，native为85.8467 μs，约减少60.55%；这是本窗口新FP32实测，不是历史FP64的215 μs。之前最新232.6826与本轮217.6035属于不同采样窗口，不用于直接计算本轮加速。

真实训练总kernel数均147/step，col2im均5/step；H2D/memcpy均20次/10steps、6,336,160 bytes，未增加拷贝。final stream synchronize均1次/trace，没有新增逐步同步。

### 为什么算子快但训练更慢

已测事实：backward主机NVTX elapsed从2949.548增至3056.474 μs/step；launch API elapsed从599.861增至628.768 μs/step。CuPy的5个driver launch被5个native runtime launch替代，总kernel数不变。GPU kernel+memcpy活动时间并集占trace窗口从28.91%降至27.12%，说明GPU计算缩短没有直接压缩完整提交窗口。

源码可见新增runtime会检查geometry/strides/设备/out别名并构造18个int64 ctypes参数；这些Python和binding成本与异步提交空隙可能抵消GPU节省。未开启Python/CPU栈采样，因此不能把全部CPU elapsed差定量归因于某一个检查。NVTX CPU elapsed包含子调用/可能等待，不是纯Python主动执行时间；API和GPU时间重叠，不可相加，也不能把profile差值直接换算成未profile wall差。

此外，未改代码的前向GPU也从520.066变为557.704 μs，dWeight从317.774变为340.740 μs，约7%的同向采样差异表明这两份单次trace有条件漂移（频率/缓存/采样影响等未分别测定）。因此不能把总GPU变化45.296 μs全归因于col2im算法，更不能忽略七轮未profile的实际退化。微基准15轮交替是kernel效果更稳健的证据，训练七轮配对是端到端验收依据。

## 环境、完整性与交付

物理GPU1 A100-PCIE-40GB，GPU0 VLLM未触碰。使用前确认GPU1空闲，同一集中窗口完成全部GPU工作后无本轮进程/显存占用。未锁频、未改权限、未终止他人任务。nvcc12.4.99 sm_80、-O3/-lineinfo/-std=c++14，无fast-math；运行环境/每条命令/UTC起止时间在raw/window_commands.json，用户时间按北京时间换算。

原baseline库、完整必要baseline源码和working diff保存。二进制trace/库留服务器，上传包仅含报告、CSV、文本原始日志、源码和脚本；具体路径/大小/SHA见EXTERNAL_ARTIFACTS.json。原GEMM严格门限的历史失败仍保留，没有调整或宣称解决。

结论：实现与独立算子加速成立，但本轮端到端验收无收益且轻微退化，默认CuPy保持。后续是否研究binding/调度成本或其他算子，由ChatGPT审阅决定，本轮停止。
