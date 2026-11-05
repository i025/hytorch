# CPU 数据流水线专项：生命周期修复与性能负结果

本轮 **CPU工作完成**。分支 `experiment/cpu-training-pipeline`，基线
`9909e546d9314e342663d75c4a10d1f46b4b7351`。没有查询GPU、创建CUDA设备上下文、
运行GPU计算、编译CUDA、修改Kernel/Runtime/优化器或安装依赖。

实际交付是现有异步 DataLoader 的 **Worker 生命周期修复**、显式文件句柄选项、完整CPU测量及统一
后续对照入口。已有异步加载的吞吐优势不归功于本轮；归一化候选没有稳定收益，
不集成正式预处理。CPU结果不能作为GPU端到端训练加速。

[源码分析](source-analysis.md) · [命令](commands.md) · [完整数据索引](data-index.md) ·
[机器汇总](results/summary.json) · [原始90次性能](results/raw-01/loader-grid-01/runs.csv)

## 验证与范围

| 项目 | 实测结果 |
|---|---|
| 最终完整CPU回归 | **124 passed / 0 failed / 0 skipped** |
| 首轮完整CPU回归 | 122 passed / 1 failed；历史符号解析测试缺少CUDA_PATH，记录保留 |
| 同步/异步、重复/不足Batch/顺序/重置 | 合成回归通过，包含满队列关闭、错误传播、析构及旧Worker join |
| 真实图像逐元素比较 | 最终4版本×10配置×1024张，共40960次图像呈现，图像/标签全部相等 |
| 固定划分 | 原6330训练、1608验证、98排除，四版本逐索引一致 |
| 后续训练数据策略 | A/I/P/R/J/E/E-prefetch，2个Epoch各20样本，14组通过 |
| 正式CPU性能 | 初始90次+句柄原型30次+集成复测18次，138独立进程；顺序随机化，全保留 |
| CPU分阶段/cProfile/队列诊断 | 完成，独立于正式吞吐 |
| GPU正确性、Nsight、训练吞吐 | **NOT_RUN**，仅准备脚本 |

CPU回归第一次失败不是数值/生命周期失败：既有测试仅加载libcudart并解析
`cudaDeviceReset`符号，未调用函数或设备。为第二次命令设置已有
`CUDA_PATH=/usr/local/cuda-12.2`后123项通过；新增文件句柄测试后最终124项通过，没有改测试或系统环境。
真实图像校验脚本首次因每Batch重复读取NPZ参考而中断，修正为一次性载入后通过；
首次中断日志及原因保留，不记为PASS。项目导入提示未安装scipy，SVHN不在本轮范围，
没有为此安装依赖或静默切换数据集。

## 环境与公平条件

Ubuntu20.04.6，双EPYC7763；使用既有`.venv`。具体Python/NumPy/Pillow及包清单见
[环境](environment/system.txt)。CPU计算命令限定逻辑CPU0–3，nice=10，
OpenBLAS/OMP/MKL均1线程，最多4个加载Worker，单次顺序运行一个正式测量。
`CUDA_VISIBLE_DEVICES=''`。这是共享服务器受限CPU实验，不是整机独占基准。

真实数据位于 `/home/xymo/datasets/udacity/data/driving_log.csv`。直接复用4H的
`runs/udacity_real_prepared_01/protocol.json`，固定相同1024样本、顺序、Batch4及
RGB/crop/resize/FP32/CHW预处理。无增强、不改划分或标签。

初始三个版本：existing=基线加载器；repaired=仅本轮生命周期修复；
inplace=修复加载器+实验性原地归一化。正式计时逐批消费、不保留整套图像，
包含线程启动与耗尽。数据读取反复命中同一文件集，未清理OS缓存，不是冷盘I/O测试。
原始记录中的hot-cache标签表示这种重复文件协议，不代表逐次测量了页缓存命中率。

## CPU准备主要耗时

独立1024图分阶段诊断的平均耗时（显式read_bytes后PIL解码）：

| 阶段 | ms/图 |
|---|---:|
| 文件读取 | 0.04350 |
| JPEG解码 | **0.38939** |
| RGB转换 | 0.01617 |
| Crop | 0.01357 |
| Resize | **0.20287** |
| FP32归一化与连续CHW | 0.06394 |
| 诊断总墙钟 | 0.73318 |

JPEG解码约53.1%，Resize约27.7%，两者合计约80.8%（相对上述诊断总墙钟）。
正式懒加载路径另有完整cProfile；不把显式read_bytes的分项当作训练时间分摊。
这是CPU已发现的热点；本轮没有更换图像库或预处理算法来改变数值。

## 已有异步能力与本轮改动分开

以下仅prefetch=2，images/s均值±样本标准差，n=3；全部深度及原始值见CSV。

| Worker | existing基线 | repaired生命周期修复 | inplace候选（未集成） |
|---:|---:|---:|---:|
| 0 | 1292.36±29.56 | 1268.04±55.68 | 1265.36±86.06 |
| 1 | 1137.63±54.48 | 1194.58±73.13 | 1159.60±77.82 |
| 2 | 2076.13±133.47 | 1951.64±68.09 | 1977.02±127.17 |
| 4 | 2462.18±233.36 | 2347.98±278.04 | 2576.57±102.90 |

修复版本2线程相对自身同步约+53.9%，这是**原有异步功能**，不能作为本轮新增收益。
生命周期修复的2线程均值相对existing为−6.00%，4线程−4.64%，1线程+5.01%；
完整结果并不证明本轮提高了正常完整Epoch吞吐，应保留这些负值和波动。

归一化候选相对repaired：同步−0.21%、2线程/prefetch2为+1.30%、4线程/prefetch2
为+9.74%，但4线程/prefetch4为−9.24%。10种配置中仅3种平均更快，7种更慢，
没有一种三次重复均更快。**不采纳为正式预处理优化，不根据最好的单格声称加速。**

### 队列、线程与内存

修复版prefetch=2：平均`next()`同步3.156ms、1线程3.351ms、2线程2.044ms、
4线程1.710ms；`next()`包含等待及调用开销，并非纯Queue操作。
独立插桩中，2线程结果get累计484.57ms/256 Batch（约1.893ms/Batch），
put累计2.87ms；说明快速消费时主要等待样本生产，而不是put锁/满队列。
Worker样本处理、组装与Queue耗时会重叠，不能全部相加。

2/4线程启动均值约1.40/1.95ms，正常耗尽清理约0.046/0.047ms。
1024图场景启动只占很小部分，没有证据需要引入常驻线程池。
进程CPU占比同步约99.9%、2线程180.7%、4线程239.4%；RSS高水位约226.8/230.7/
236.2MiB。CPU100%是一核；RSS包含解释器/模块/元数据，不是Batch净内存。

prefetch=1/2/4：2线程分别1965.73±120.08、1951.64±68.09、2027.26±187.09
images/s，差异与波动接近。增大深度没有一致收益。保留2线程、prefetch2作为
下一阶段控制并发的对照起点，而不是宣称它是所有消费者/模型的最优配置。

## 本轮新增CPU优化：显式文件句柄

正式路径的cProfile发现每1024图有1024次`realpath`，累计0.089223秒（嵌套成本），
涉及10240次lstat。给Pillow传入已打开的文件句柄可绕过这次重复路径解析。
解码、裁剪、Resize、FP32归一化、连续CHW及标签完全不变。

新增 **默认关闭** 的 `UdacityDataset(csv_path, open_stream=True)`，仍调用同一个
`preprocess_udacity`，通过文件上下文可靠关闭句柄。没有重写解码器或缓存图像。
独立cProfile中`realpath`调用消失；原始profile及逐图数据均保留。

为避免与初始90次跨时段拼接，另做匹配对照，每组均使用修复版DataLoader。
以下为真正集成可选API后的3次重复，prefetch=2，均值±样本SD：

| Worker | 原路径 images/s | open_stream images/s | 吞吐提升 | 平均墙钟加速比 | 更快重复 |
|---:|---:|---:|---:|---:|---:|
| 0 | 1206.73±61.20 | 1263.67±5.58 | +4.72% | 1.0489× | 2/3 |
| 2 | 1932.21±102.70 | 2189.78±82.25 | **+13.33%** | **1.1343×** | **3/3** |
| 4 | 2411.01±77.86 | 2797.45±429.83 | +16.03% | 1.1416× | 2/3 |

先前5次独立重复的原型实验也全部保留：同步+6.95%（4/5）、2线程+9.09%（5/5）、
4线程+27.88%（5/5）。两批各自计算，不混合成单一成绩。集成后4线程有一次−2.09%，
标准差大，不能声称4线程或所有条件稳定改善；2线程在两个实验中均每次更快，
支持保留这个简单选项。它是本轮 **CPU读取入口优化**，不是已有异步预取收益，
也不能被称为CUDA Kernel或GPU训练加速。

[逐次句柄测量](results/raw-02/stream-integrated-01/runs.csv)、
[完整汇总与误差条](results/stream-summary.csv)、[比较JSON](results/stream-comparisons.json)、
[新增图](figures/08-explicit-file-open.png)。正式预处理数学和默认dataset行为都不变。

## 生命周期修复的确切收益

满结果队列、2个Worker的同一合成复现：旧实现等待10.001204秒，仍有2个活线程，
但已清空线程引用；修复后约0.000108秒，0活线程。
这是取消正确性与退出延迟的改善，不是正常训练加速。
新增停止Event、可退出结果提交、真实join及`close()`/上下文管理。自然耗尽、
重新开始、Worker异常、析构和提前退出均有回归。

提前`break`应使用`with DataLoader(...) as loader`或显式`close()`；不能假定Python
会自动关闭仍被持有的迭代器。当前用户样本读取必须返回后才能join，线程无法被
强行取消。顺序pending不受队列容量约束仍是已知限制，见[源码分析](source-analysis.md)。

## 下一次GPU验收优先级

统一入口[future_training.py](scripts/future_training.py)已完成CPU dry-run，
[126项计划](results/future-gpu-02/plan.json)尚未运行。包含DonkeyNet/SpatialNet、
各三重复256步、10步预热、compute/io/phases分开，同一窗口轮换A/I/P/R/J/E/E-prefetch。

1. 先验收前一分支Native ReLU及同状态数值；本轮没有GPU正确性结果。
2. 优先比较I/P与E/E-prefetch的真实I/O训练，确认现有预取对两个框架的收益，
   包含CPU争用、H2D和完整线程生命周期，再分析R/I及J/I。
3. 计算J/A累计、J/E及J/E-prefetch差距，使用同一物理卡同一配置的实测；
   检查数据预取是否遮住CPU处理，而非直接把CPU images/s转成训练加速。

| 未来指标 | 本轮状态 |
|---|---|
| P/I与E-prefetch/E（已有CPU预取） | NOT_RUN |
| R/I（Native ReLU）与J/I | NOT_RUN |
| J/A累计、J/E、J/E-prefetch | NOT_RUN |
| GPU Kernel时间、显存、Nsight | NOT_RUN |

未来所有版本统一显式`open_stream=True`，公平地共享这项CPU优化；其开关不能只给HyTorch使用。
模型初始化、真实Batch、顺序、优化器和计时口径均共享；没有用PyTorch专有加载器
为其更换数据协议。普通吞吐仍使用所有路径一致的既定边界同步口径，并非新引入
异步GPU提交优化。GPU计划需要下一轮单独授权及正确性前置，本轮不自动运行。

## 图与原始数据

均由实测CSV/JSON生成，非截图：

1. [吞吐与三版本](figures/01-loader-throughput.png)
2. [预取深度](figures/02-prefetch-depth.png)
3. [预处理阶段](figures/03-preprocess-stages.png)
4. [本轮增量与负结果](figures/04-incremental-results.png)
5. [Queue诊断](figures/05-queue-diagnostics.png)
6. [满队列关闭复现](figures/06-worker-shutdown.png)
7. [CPU与RSS](figures/07-cpu-memory.png)
8. [文件句柄匹配对照](figures/08-explicit-file-open.png)

[完整索引](data-index.md)列出逐次CSV/JSON、逐Batch耗时、逐图阶段、Queue调用、
环境、日志、失败与复测及服务器二进制路径。原图、历史参考NPZ和pstats留服务器。
本轮生产修改仅涉及DataLoader及UdacityDataset可选打开方式，新增测试及实验脚本/报告。完成提交推送后停止，等待审查。
