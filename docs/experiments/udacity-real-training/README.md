# 阶段4H：GPU1共享环境下的真实Udacity训练

2026-11-04，基线`a5a50159f6217f10eee8028b31503a444212b1ac`，分支`experiment/real-udacity-training`。本轮没有新增CUDA Kernel、修改卷积/自动求导/优化器或接入Unity。

**实验执行完成，长期验收未全部通过。** 用户确认首次原进程正常退出并批准同GPU1重测后，第二连续窗口完成全部47项任务。CPU82项回归、真实图像CPU408组及GPU816组短程数值比较通过；36组256步性能测量全部满足既有数值容差。但完整三epoch训练的长轨迹严格比对有119组失败，验证指标也未优于初始模型/零预测基线。负结果与完整原始数据保留，没有放宽容差、重选样本或修改Kernel使结果变绿。

所有性能数字标记 **SHARED_GPU**，不是独占A100最终基准。正式含I/O吞吐I/F平均+2.99%、I/A累计+16.00%、I/E为77.76%；增量与波动接近，不能宣称稳定收益。驻留输入的I/F平均+5.55%，但A/I波动很大，同样不能作确定加速结论。

[数据索引](data-index.md)、[完整命令](commands.md)、[数据报告](dataset-report.md)、[源码与计时口径](source-analysis.md)、[机器摘要](results/acceptance-01/summary.json)。

## 验收状态

| 项目 | 状态 | 实测 |
|---|---|---|
| 数据检查/固定划分/预处理 | PASS | 8036条、24108张RGB图，缺失0 |
| CPU回归 | PASS | 82 passed / 0 failed / 0 skipped；此前80项日志保留 |
| 真实图像CPU三步对照 | PASS | 408组，最大绝对误差5.0291419029e-8 |
| 真实图像GPU三步对照 | PASS | A/F/I/E共816组，最大绝对误差3.1664967537e-8 |
| 四版本真实训练烟测 | PASS | 8步，Loss有限、参数更新；I确实调度Split-K dWeight |
| 36组256步匹配测量数值 | PASS | compute/io/phases各四版本×三重复，参数和Loss通过原容差 |
| 完整真实训练与验证执行 | PASS | 四版本各3epoch、4749步、18990张训练样本呈现；每epoch完整1608张验证 |
| 完整训练严格数值一致性 | **FAIL** | 全部1530组比较中119组失败，全部集中在完整训练任务；最大绝对差1.6611814499e-4 |
| 泛化改善 | **未证实/本轮指标未达标** | 第三轮验证MSE/MAE/RMSE仍差于初始模型和零预测基线 |
| Nsight F/I路径验证 | PASS | 两份16步Trace、0未匹配Kernel/API，Im2col/Copy移除、新Kernel实际运行 |
| 稳定端到端增量收益 | 未证实 | I/O平均+2.99%，但波动重叠；驻留输入CV约14% |
| 原任务应用延迟与自身SM活动 | BLOCKED / NOT_AVAILABLE | 未提供应用延迟指标；原任务pmon SM字段均不可用，不能承诺零影响 |
| 独占GPU验收/Unity/新Kernel | NOT_RUN | 不属于本轮执行范围 |

## 环境和实际GPU生命周期

Ubuntu20.04.6，2×EPYC7763/256逻辑CPU；现有项目独立`.venv`，Python3.8.10、NumPy1.24.4、CuPy12.3.0、PyTorch2.4.1+cu121、cuDNN9.1、Toolkit/CuPy Runtime12.2、nvcc12.2.140、驱动535.247.01、Nsight Systems2023.2.3。完整环境见[environment](environment)。复用4G SM80动态库，CPU ABI/符号检查通过；没有修改/重编新Kernel或做哈希验证。

全阶段固定物理 **GPU1**，UUID `GPU-c6683db1-d2aa-97b9-c1b4-e497bd7ff0c5`，`CUDA_VISIBLE_DEVICES`为此UUID，内部逻辑0。实际有两个窗口，不能称整个阶段只有一次：

| 窗口 | 北京时间2026-11-04 | 结果 |
|---|---|---|
| 首次01 | 15:20:46.754–15:20:59.202 | 正确性通过，原PID2975119退出；监控终止本轮smoke-A，后续未运行 |
| 用户明确允许的完整补测02 | **15:27:53.830–15:34:29.858** | **47/47完成**，同GPU1连续完成全部正式任务，无设备切换 |

首次启动时两个原进程占24990/7473MiB，整卡32504MiB。监控发现PID2975119退出时最高利用率9%、最高显存33110MiB；没有证据据此判定异常或因果关系。用户随后明确确认“原任务正常结束，允许同 GPU1 重新完整补测”。首次记录不删不改，见[中断记录](results/window-01-interruption.json)和[用户确认记录](results/retry-authorization.json)。

第二窗口启动时原PID2216423占24990MiB，整卡25001MiB，空闲15959MiB。正常估算本轮约2GiB以内，采用batch4；没有申请全部显存，CPU控制器从未创建CUDA context，没有忙循环或显存占位。子任务顺序执行，未并行训练/benchmark，未操作原任务。

第二窗口结束后所有本轮子进程退出，监控线程join，GPU1仅原PID2216423仍在，整卡恢复25001MiB。**没有执行cudaDeviceReset或GPU Reset，没有修改Power Limit/Clock、MIG/MPS。** GPU结束后SQLite导出、分析、绘图、Git均在`CUDA_VISIBLE_DEVICES=''`下进行，未再次启动GPU。

## 数据与预处理

使用实际`/home/xymo/datasets/udacity/data/driving_log.csv`及IMG；8036记录、center/left/right共24108张RGB320×160，CSV引用缺失0，center唯一8036张。steering范围−0.9426954至1，零转向54.27%。仅训练center，直接使用CSV标签，没有套用DonkeyCar文件名标签或虚构左右修正。

统一Pillow RGB→裁顶部70/底部25→bilinear resize200×66→float32像素/127.5−1→连续CHW。训练、验证、单图接口共用`preprocess_udacity`。无增强、无颜色fallback，模型结构不变。

按时间末20%留出验证，边界purge10秒：训练6330、验证1608、排除98，实际边界间隔10.037秒。每epoch使用seed42+epoch的固定排列；所有后端相同。吞吐实验固定seed42从训练集合取1024图，预处理缓存与CSV行号留服务器。这个划分减少邻帧泄漏，但同一轨道/采集片段可能重复场景，不代表跨轨道、天气或闭环驾驶泛化。

## 四版本协议与计时

A为当前代码复现的CuPy+普通SGD、未剪枝、无本项目Native基线；性能子进程指定不存在的Native库，实际dispatch为空。CuPy/cuBLAS内部CUDA不算本项目Native。F为shared GEMM+im2col Forward+pool CuPy+tile16+Fused SGD+首层剪枝+原Gather；I只在F上加4G Split-K dWeight；E为匹配PyTorch CUDA/cuDNN。没有使用fused_opt或Auto。

全部FP32、batch4、SGD lr=.001、seed42、dropout0、TF32关闭，相同初始化/数据/顺序，10步预热后恢复初始参数。所有训练`--no-fingerprints`；初始化逐参数数组比较，输入协议逐索引比较，不做SHA/MD5/checksum。

- compute：1024张真实预处理图像一次传入GPU，正式256step，排除磁盘/预处理/H2D；墙钟仍包含Python、同步、Loss日志，不是纯Kernel计时。
- io：同1024图像顺序，每步真实读JPEG、预处理、H2D、训练；操作系统缓存不清理，不是冷盘实验。
- phases：独立同步阶段计时，不参与正式compute/io吞吐比。
- train：全6330图×3epoch，末批2张保留，独立同步F/B/U以记录epoch诊断；验证全1608张，不能把其吞吐混为未插桩基准。
- trace：F/I各16正式step，单独Nsight；预热/初始化在outside_phase，不能混入正式16step统计。

## 长期真实性能表（SHARED_GPU）

均为同卡、同一第二窗口三次独立256step的samples/s均值±样本标准差。四版本顺序按重复轮换AFIE→FIEA→IEAF。

| 版本 | 驻留输入samples/s | 含I/O完整samples/s | 含I/O相对F | 含I/O相对A | 含I/O PyTorch比值 |
|---|---:|---:|---:|---:|---:|
| A 无Native | 627.13±91.39 | 431.45±8.74 | — | 1.0000× | 67.03% |
| F 旧优化 | 759.92±14.68 | 485.95±15.03 | 1.0000× | 1.1263× | 75.50% |
| I 新dWeight | **802.11±114.32** | **500.48±11.16** | **1.0299×** | **1.1600×** | **77.76%** |
| E PyTorch | 1607.90±47.79 | 643.62±3.46 | — | 对照 | 100% |

| 本窗口比较 | 驻留输入吞吐变化 / 墙钟加速 / 耗时下降 | 含I/O吞吐变化 / 墙钟加速 / 耗时下降 |
|---|---|---|
| I/F 增量 | +5.55% / 1.0402× / 3.87% | **+2.99% / 1.0302× / 2.93%** |
| I/A 累计 | +27.90% / 1.2799× / 21.87% | **+16.00% / 1.1599× / 13.79%** |
| I/E | 49.89%吞吐；平均墙钟为E的2.0334倍 | 77.76%吞吐；平均墙钟为E的1.2864倍 |

吞吐比是三次samples/s均值之比，墙钟加速是三次wall均值之比，不能混用或与历史4G百分比相乘。三次真实原值、完整step与标准差保存在[training-raw.csv](results/acceptance-01/training-raw.csv)、[timing-summary.csv](results/acceptance-01/timing-summary.csv)、[speedups.csv](results/acceptance-01/speedups.csv)。

A驻留输入三次为521.79/674.44/685.17，I为670.47/876.50/859.36，CV约14.57%/14.25%。没有丢弃第一次较慢成绩。I/O的I三次506.15/487.62/507.67，F为478.58/476.03/503.25，区间重叠；平均增量不足以证明共享环境中稳定加速。观察到时钟765–1410MHz变化，但没有证据把全部波动归因于共享任务或时钟，不做无计数器支撑的归因。

### 独立同步阶段计时

单位ms/step，三次均值；原始SD同样保留：

| 版本 | 数据读取/预处理 | Transfer | Forward | Backward | Update |
|---|---:|---:|---:|---:|---:|
| A | 2.9810 | .1340 | 1.6705 | 3.4644 | .4282 |
| F | 3.0120 | .1374 | 1.6991 | 3.0582 | .1283 |
| I | 3.0408 | .1403 | 1.7232 | 2.6329 | .1253 |
| E | 3.1464 | .2118 | .6794 | 1.9237 | .1955 |

含I/O普通计时中，每batch读取/预处理均值约3.06/2.98/3.15/3.02ms（A/F/I/E）。数据处理是显著成本，限制计算优化映射为端到端收益；不能把多个独立计时实验直接相减当作精确瓶颈分摊。

## 三轮真实训练与验证

每版本实际4749次更新，初始/最终参数和完整Loss保留。所有Loss有限，参数实际变化。下表非性能三重复，不能用三轮时间当作三个独立重复：

| 版本 | Train MSE epoch1→3 | Val MSE epoch0→3 | 第3轮MAE | 第3轮RMSE |
|---|---|---|---:|---:|
| A | .01627805→.01538791 | .01597852→.01706523 | .08057055 | .13063396 |
| F | .01627802→.01538883 | .01597852→.01706774 | .08058443 | .13064356 |
| I | **.01627801→.01538777** | **.01597852→.01706862** | **.08059391** | **.13064692** |
| E | .01627806→.01538942 | .01597852→.01706626 | .08058219 | .13063789 |

验证Loss在epoch1升至约.01823，epoch2/3回落，但仍高于初始模型。零预测基线MSE/MAE/RMSE为 **.01666691/.06865888/.12910038**，仍优于第三轮四模型；训练均值常数基线为.01808696/.08154359/.13448776。训练标签均值+.01599205、验证−.03640254，分布存在差异；尚未用额外实验量化其影响。训练Loss下降不是已收敛的证明，更不是自动驾驶能力或泛化提升的证明。本协议没有按验证结果调参、延长训练或改变数据分布来追求通过。

每轮F/B/U、GPU池/allocator记录、读取时间、epoch墙钟、全部1608图预测与验证指标均见[epochs.csv](results/acceptance-01/epochs.csv)、[validation-predictions.csv](results/acceptance-01/validation-predictions.csv)。权重及真实图像仅在服务器。

## 长轨迹数值失败与离线诊断

梯度/Forward沿用atol1e-5、rtol1e-4；Loss/更新权重沿用atol1e-6、rtol1e-5。没有修改标准。

816组三步真实GPU比较通过，36组256step固定子集的末态权重/Loss也通过。但完整6330图、三epoch的最终16参数+4749个Loss序列比较，全部非自身A/F/I/E对照出现超差：总1530组中的119组失败，全在train任务。**不把执行完成写成数值PASS。**

- A/E首次Loss超差在第959步，F/I在第1422步；F/E和I/E在第863步。均早于第一轮末批1583步，因此首次偏离不是由首次N=2末批更新触发。
- F与I（此处以I为参考）有1919个Loss超差，最大差8.9220702648e-5。其末态权重41371个元素超差；最大权重差8.8435597718e-5，位于fc2/b[46]，两值.0146077238023/.0146961594000，允许差约1.14696e-6。
- 全部比较的最大绝对差1.6611814499e-4来自完整Loss轨迹。A/E自身也出现长期失配，不能单独归咎于新dWeight。
- 初始化、协议/数据顺序相同，预处理固定无增强，短程局部梯度通过。不同FP32归约、非线性路径与长期迭代敏感性是待验证候选解释，**不是已确认根因**。

没有逐步激活/梯度/参数检查点，现有末态与Loss日志无法确定第一个出错算子，不能宣称发现或修复了CUDA数学Bug，也不能直接断言只是正常舍入。未做无依据的源码修补；未再次启动GPU。首次失败的输入CSV行号、数值、最坏参数坐标及全部误差保存在[drift-01](results/drift-01/summary.json)与[numerical-errors.csv](results/acceptance-01/numerical-errors.csv)。建议人工审查后再决定是否用相同中间状态的一步梯度/更新对照定位，不能直接放宽长期标准。

## Nsight真实路径

两份独立16step Trace，Kernel/API未匹配数0，归因使用correlationId→提交API→NVTX父节点，不用异步时间重叠猜父节点。下列均为插桩数据，不与普通训练吞吐混算：

| 指标（每正式step） | F | I |
|---|---:|---:|
| dWeight Kernel数 | 24 | **10** |
| dWeight Im2col / Copy数 | 5 / 10 | **0 / 0** |
| dWeight cuBLAS GEMM数 | 5 | 0 |
| 新tiles / reduce数 | 0 / 0 | **5 / 5** |
| dWeight累计GPU μs | 338.014 | 301.138 |
| 其中tiles / reduce μs | 0 / 0 | 268.532 / 32.606 |
| dWeight主机范围累计 μs | 1372.875 | 569.340 |
| 整个Backward Kernel数 | 66 | 52 |
| 整个Backward GPU μs | 1110.499 | 1071.923 |
| 整个Backward主机μs | 5209.158 | 4224.490 |
| data主机μs | 3411.276 | 3293.607 |

I每层dWeight均为1tiles+1reduce，80次父节点全部验证。Forward/Backward/Update范围内两版本sync API数均0；StepProfiler原有边界同步在范围外，没有证据显示新dWeight额外插入同步。主机范围缩短与减少提交相符，但不能把主机提交时间等同GPU核时。Timeline只捕获本轮进程，不把间隙称为整卡空闲/SM利用率。

## 原任务与共享资源观察

第二窗口290次采样，原PID2216423在290/290中存在，显存始终24990MiB。整卡最高25721MiB，**本轮进程采样峰值714MiB**，结束后仅原进程；整卡利用率最高64%、温度最高30°C、功耗最高70.39W，SM时钟765–1410MHz。未观察到本轮OOM、原进程消失或持续≥95%负载触发。

原任务pmon SM字段290次均为N/A，应用延迟/吞吐指标未提供，因此不能证实原任务性能不变，更不能承诺零影响。显存/存活只证明可观察资源状态。监控是采样而非精确峰值，有7个较短epoch计时区间没有独立资源采样，明确标记NOT_SAMPLED，而不是当成零活动。逐任务/epoch对应见[per-run-resource-samples.csv](results/extra-01/per-run-resource-samples.csv)，完整原始pmon仍保留。

## 代码清单与后续审查

新增生产数据适配器`donkeycar_hy/udacity_dataset.py`；新增CPU测试`tests/test_udacity_dataset.py`、`tests/test_udacity_experiment.py`；本目录新增数据审计、匹配训练、共享资源控制器、正确性、漂移诊断、Nsight离线分析和绘图脚本。既有CUDA Kernel、runtime、优化器、模型与benchmark均未修改。

阶段执行和数据归档完成，长期数值/泛化验收有明确负结果。下一步仅建议共同审查首次偏离数据、选择中间状态重放诊断方案和验证协议；不自动开始新Kernel优化、4I或Unity集成。所有失败、首次中断及用户批准补测记录保留。

## 全部结果图

均由实测CSV/CUPTI生成，不是终端截图：

1. [四版本真实吞吐](figures/final-01/01-throughput.png)
2. [Train/Validation Loss](figures/final-01/02-train-validation-loss.png)
3. [Steering MAE/RMSE](figures/final-01/03-steering-errors.png)
4. [独立同步阶段耗时](figures/final-01/04-phases.png)
5. [相对无Native累计收益](figures/final-01/05-cumulative-gains.png)
6. [与PyTorch差距](figures/final-01/06-pytorch-gap.png)
7. [GPU1共享资源](figures/final-01/07-shared-resources.png)
8. [真实数据Nsight时间线](figures/final-01/08-nsight-timeline.png)
9. [标签分布](figures/final-01/09-steering-distribution.png)
10. [完整epoch训练/验证时间](figures/final-01/10-epoch-times.png)
11. [长轨迹数值失败](results/extra-01/11-long-run-drift.png)
12. [验证与初始/零预测基线](results/extra-01/12-validation-baselines.png)

另外保留[准备阶段标签图](figures/preparation-01/dataset-steering.png)和[CPU/GPU真实图像短程正确性](figures/preparation-01/real-image-correctness.png)。大型图像数据、NPZ权重、两份nsys-rep和SQLite留服务器，Git上传完整可移植文本/CSV/JSON/XML/PNG及逐文件索引。
