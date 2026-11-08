# HyTorch GPU 资源监控与训练瓶颈分析

本轮交付可复用、默认关闭的 TrainingMonitor、DonkeyNet/TinyNet CLI 集成、通用示例、真实 A100 数据及六组 PNG/SVG。没有修改 CUDA 算法、GEMM 调度、F1/O1 或优化器数学语义，没有提交 Git。GPU 已释放。

## 环境与实验设计

物理 GPU 1：A100-PCIE-40GB，UUID GPU-0bc09ee6-967a-b53d-bf80-38379585cf08；CUDA_VISIBLE_DEVICES=1 对应逻辑 0。驱动 580.95.05，Python 3.10.12、NumPy 2.2.6、CuPy 14.2.0、PyTorch 2.9.1+cu128。GPU 使用前后快照保存在 raw/gpu_before.txt、gpu_after.txt。没有终止其他进程或修改系统配置。

固定 256 个合成样本、batch=4、3 epochs、192 steps、seed=42、SGD=.001。使用 shared GEMM / im2col forward / native pool、col2im / packed dWeight / F1 metadata cache / O1 compact Function / legacy SGD。实验环境 JSON 的 config 是进入上下文之前的默认配置，实际运行配置见 final_config 和 diagnostics/experiment.py 的 config()。TF32 关闭；PyTorch cuDNN deterministic=True、benchmark=False；CPU 数学库线程数 4。

四种模式同进程 12 轮平衡交替，每轮重新恢复初始化、数据顺序、优化器状态，统一 warmup 和训练末同步。禁用组使用同一个训练 helper 的 no-op monitor。无 profiler 的正式训练是验收依据；独立 Nsight 和 allocator hook 诊断不混入正式结果。PyTorch 三组参考试验使用相同模型/数据/初始化，但 cuDNN/cuBLAS 与 HyTorch 自研实现不同。

## 监控开销（12 轮配对）

| 模式 | 完整 monitor 生命周期 wall/s | samples/s | 相对均值开销 | 配对 Δwall 的 95% CI/ms | 仅训练窗口/s |
|---|---:|---:|---:|---:|---:|
| disabled | 0.731336 | 1050.13 | — | — | 0.731329 |
| timing | 0.770375 | 996.92 | +5.34% | [25.15,52.93] | 0.756042 |
| resources | 0.741567 | 1035.64 | +1.40% | [-3.73,24.19] | 0.728912 |
| all | 0.793535 | 967.82 | +8.50% | [34.19,90.20] | 0.767744 |

完整 wall 包含初始化、最后事件读取与 CSV/JSON 导出；训练窗口只含相同训练循环及最终同步。全部监控的训练窗口开销为 **4.98%**，配对差 CI [8.43,64.40] ms/run。资源采样的训练窗口差 CI [-15.66,10.82] ms，不支持“采样加速训练”的结论。区间为 Student t 配对区间（df=11）；共享服务器连续样本可能有关联，不是跨机器总体保证。没有删除异常值。原始逐对数据见 raw/OVERHEAD_PAIRS.csv，完整测量见 RUNS.csv。

PyTorch disabled 三次训练窗口均值 0.429981s，all 为 0.440362s；HyTorch disabled 为 0.731329s。此处是框架参考，不是本轮算子加速，也不能将两组不同轮次视为直接配对。

## GPU/CPU 资源

短训练只有约 4–5 个采样点，因此另做 30 epochs / 1920 steps、约 7.62s 的趋势窗口。共 39 点，其中 38 点采集开始时间位于训练内；CPU 首次 percent 为 null，37 个有效点。周期 0.2s，约跨 50 steps，不能逐 step 定位 GPU 执行阶段。

| 指标 | 实测 | 范围/来源 |
|---|---:|---|
| GPU utilization | 平均 38.82%，24–41% | NVML 整卡、驱动自身统计窗口 |
| 显存控制器 utilization | 0% | 本次驱动有效读数，不是显存容量百分比 |
| 整卡已用显存采样峰值 | 987.44 MiB | NVML，非瞬时真实峰值 |
| 本进程 GPU 显存采样峰值 | 460 MiB | NVML PID accounting |
| CuPy active 采样峰值 | 14.23 MiB | 趋势窗口默认 memory pool |
| CuPy reserved 采样峰值 | 26.18 MiB | 默认 memory pool |
| 温度 | 28°C | 短窗口恒定，不能推断长时间热稳定性 |
| 功耗 | 平均 43.94W，39.13–44.71W | NVML 整卡 |
| SM clock | 765–810MHz | 未锁频；memory clock 1215MHz |
| 系统 CPU | 平均 4.51%，1.8–39.4% | 全系统多核归一化 |
| 本进程 CPU | 平均 105.02%，94.9–275.8% | 100%=一个逻辑核 |
| RSS 采样峰值 | 936.18 MiB | psutil 本进程；10 个线程 |

趋势采样调度延迟均值 1.424ms、最大 48.373ms；采集耗时均值 1.689ms。保留实际时间与偏差，不声称严格等间隔。采样观察到的计算进程只有本实验 PID，但整卡指标仍不能当作本进程 GPU 利用率。整卡与 PID 显存差额不能未经进一步分析归因到某一类分配。

单独的 192-step allocator hook 功能试验：CuPy active 跟踪峰值 **16.16 MiB**，同期采样峰值只有 **7.29 MiB**。跟踪峰值最后一次达到时的 CPU 标签是 forward、step191。它是主线程默认池分配瞬间的峰值，不是整个 GPU 的精确峰值或 GPU kernel 阶段定位；上一 step 梯度存活可能影响 forward 内存，但这里只是解释假设。hook 不用于正式开销测量。

## 时间分解

以下是 12 轮 all 模式的 2304 steps 均值；CUDA Event 是同一 stream 时间区间，包含提交间隙，不是纯计算时间。

| 阶段 | CPU 主机/ms | CUDA Event/ms |
|---|---:|---:|
| DataLoader next | 0.09057 | 不适用 |
| 数据准备/metadata | 0.00455 | 不适用 |
| H2D | 0.15954 | 0.15210 |
| Forward | 0.95764 | 0.95261 |
| Loss | 0.13223 | 0.12609 |
| Backward | 2.23271 | 2.23163 |
| Optimizer | 0.33561 | 0.32920 |

Host iteration 平均 3.98951ms，最大 75.3744ms（保留异常点）；step event 区间平均 3.97802ms。异步 throughput 的单 step 完成延迟没有被独立测得，step_wall_s 留空。禁用监控的完整训练窗口除以 192 为 3.80901ms/step，是吞吐等效平均值。另做逐 step 末同步的 latency diagnostic，完成 wall 平均 **4.10095ms/step**，不能替代正式吞吐结果。

## 瓶颈证据与边界

独立的 monitor-disabled Nsight Systems 20-step 时间线：每 step 142 个 kernel，其中 Backward 72 个；Forward 34、Loss 4、Optimizer 32。纯 kernel 时间分别为 615.37、21.38、809.08、182.36 μs/step。Backward 同时占最大主机时间和最大纯 kernel 时间，是首要分析对象。

该 trace 的活动跨度 95.45ms，kernel/memcpy/memset 并集 33.36ms，约 65.05% 区间没有这些已记录活动。它表示 **profiler 下观察到的时间线间隙**，不是 SM occupancy，也不是无 profiler 训练的精确空闲率。Profiler 有额外 CPU 开销，约 4.77ms/step 的该窗口不能替换正式约 3.81ms/step。主机约一个核、NVML 利用率约 39%、大量小 kernel 与时间线间隙共同支持 CPU 提交/调度与 GPU 等待是重要限制的假设；不能把所有间隙归因于 Python。

合成数据 next/prepare 较小，当前证据不支持磁盘加载为主要瓶颈；这不适用于真实图片解码。当前显存、温度没有显示容量或热压力。后续可研究 backward 包装/launch 数量与提交间隙，但本轮没有实施进一步优化。

## 正确性、交付与限制

完整回归 **347 passed, 2 skipped**；两个 skip 是已有 CPU DonkeyNet 重复路径，GPU 测试实际执行，新增 GPU 测试 6 项通过。1884 条数值比较无门限失败，监控开关之间逐位一致；PyTorch 参考最大绝对误差 4.47035e-8、最大 relative-L2 1.15414e-5（小梯度相对量敏感）。详见 CORRECTNESS.md。

既有 ResourceMonitor/StepProfiler API 保留；CLI 默认不开新监控。旧 CLI 曾隐式启动资源采样，本轮按默认关闭要求改为明确选择；原有训练数学流程保留。默认旧 StepProfiler 的 phase 同步行为未被本轮更改；正式对比均用新的统一 helper，不与旧 phase profiler 混比。

监控需要额外事件和文件导出，全部开启不是“零开销”。建议正常吞吐测试关闭、资源趋势单独开启、详细阶段计时按诊断需要开启。跨 stream 并发不支持完整归因；长训练需在安全边界 flush_events 控制事件存活量。无新增 CUDA kernel 或异步 GPU buffer，不额外运行无关 Sanitizer。报告、设计、指标字典、源码 diff、原始数据、6 PNG+6 SVG 和 PPT_SUMMARY 一并交付；大权重与二进制 trace 不打包，原位置与 SHA 见 EXTERNAL_ARTIFACTS.json。
