from pathlib import Path
import json,csv,hashlib,difflib,shutil,subprocess
R=Path('/home/xieminhui_pubuse/hytorch'); O=R/'runs/a100_20261108/training_monitor'
def put(n,s): (O/n).write_text(s.strip()+'\n')
put('REPORT.md','''# HyTorch GPU 资源监控与训练瓶颈分析

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
''')
put('DESIGN.md','''# TrainingMonitor 设计与 API

仓库现有 hytorch/profiling.py 是模块文件，因此新增 hytorch/training_monitor.py，并由 profiling.py 导出 TrainingMonitor，避免破坏已有 import。

```python
from hytorch.profiling import TrainingMonitor
m = TrainingMonitor(enabled=True, gpu_index=0, sample_interval=.2,
                    output_dir="runs/my_monitor", cuda_backend="cupy")
with m:
    it = iter(loader)
    for i in range(len(loader)):
        with m.step(i, epoch=0):
            with m.phase("data_load", gpu=False):
                batch = next(it)
            with m.phase("forward"):
                prediction = model(batch)
            # 用 phase 分别包裹准备、传输、loss、backward、update。
print(m.run_dir, m.summary())
```

可直接运行 examples/training_monitor.py；通用 monitored_epochs 位于 hytorch/monitor_loop.py，显式包裹 next(iterator)，通过 transfer/predict/loss_fn/backward/update/retain_loss 回调接入任意模型。retain_loss 应返回 detached 数组，避免保留计算图。支持自定义 prepare_batch；此次合成输入准备为 identity 加 metadata，不冒充图片解码。

构造默认 enabled=False；禁用时不导入 NVML/psutil/GPU 库、不建采样线程、不输出文件。timing/resources 可分别开关；CPU 使用 cuda_backend=None、gpu_index=None；Torch 使用 cuda_backend='torch'。每次 start 创建唯一 run 子目录，异常退出也 stop/export；同一实例可再次运行。step/phase 由创建运行的线程操作，拒绝嵌套、重复 start 和错误生命周期。提供 start/stop、step、phase、set_step_metadata、flush_events、summary、export。

GPU 设备用当前 runtime 的 UUID/PCI 对应 NVML handle，记录逻辑编号、物理编号、CUDA_VISIBLE_DEVICES；不根据掩码字符串猜编号。纯资源模式可显式 UUID/PCI，避免与 runtime selector 冲突。后台线程只选择其自身逻辑设备（非零时），不修改主线程设备/stream。NVML 或 psutil 不可用时继续训练、记录错误和 null。

采样线程 priming psutil，第一条 percent=null。单次读取 NVML、psutil、allocator 计数，不创建 nvidia-smi 子进程、不主动复制 GPU 数据或同步。样本保存实际时钟、计划时钟、延迟和采集耗时；CPU phase 标签只是采样开始时的 host 标签。停止时通知、join 并关闭 NVML。

phase 在当前 CUDA stream 上记录起止 events，退出 phase 不同步；stop/flush_events 在明确完成边界同步相关 stream 后批量读事件。phase 首尾 stream 改变时该 event 区间标为 unsupported；不能捕获所有跨流依赖或并发 GPU 工作，阶段间区间可能重叠，禁止直接求和当总计算量。throughput 保存 host_iteration_s，CUDA step_wall_s=null；latency 在 step 末同步并记录完成延迟，仅用于诊断。

事件及明细保留量为 O(steps×phases)，长训练应在安全 epoch 边界 flush_events；明细仍保存在内存，极长任务需要后续分块导出设计。allocator_peaks 默认关闭。CuPy hook 跟踪主线程默认 pool 分配瞬间，记录 latest-equal peak 的 CPU 标签；其他线程/custom allocator 不在该“精确”范围。Torch opt-in 会重置目标设备峰值计数，影响共享这些计数的同进程代码，应仅诊断使用。

CLI --monitor 启用新循环；--monitor-components disabled/timing/resources/all；--monitor-mode throughput/latency；--monitor-interval；--monitor-allocator-peaks；--output。--final-config 选取已接受配置，不改 kernel。没有 --monitor 时保留已有 StepProfiler 训练分支，资源采样默认关闭。--nvtx 可独立配合 helper 用于外部 trace。stop 导出耗时计入完整生命周期开销，但不计入 training_window_s。
''')
put('CORRECTNESS.md','''# 正确性与测试证据

- raw/full_tests_final.log：347 passed, 2 skipped in 5.98s。GPU 运行设置 HYTORCH_REQUIRE_CUDA=1；skips 为既有 CPU DonkeyNet 重复路径，不算 GPU 通过。
- 新增 tests/test_training_monitor.py：22 个 CPU 测试；tests/test_training_monitor_cuda.py：6 个 GPU 测试。覆盖禁用、时钟/生命周期、嵌套误用、异常清理、空运行、独立输出、缺依赖、mapping/sentinel、采样首点、非零设备采样线程、event 延迟读取/跨流、非默认 stream、重复训练、allocator、Torch 和数值对齐。
- raw/NUMERICAL_ALIGNMENT.csv：1884 条检查，failed_elements 全为 0。监控开关、latency、allocator、TinyNet 比较逐位相等；PyTorch vs HyTorch 允许原有门限，最大 abs=4.470348358154297e-8，最大 relative-L2=1.1541391638554686e-5，不能称跨框架逐位相等。
- 原门限不变：weight/loss atol=1e-4,rtol=1e-3；gradient atol=2e-5,rtol=2e-4。同时记录 bitwise、max_abs 和 relative_l2。
- CPU CLI 旧循环与新 monitor 循环最终权重相同；实际 CUDA DonkeyNet 完整训练、Torch CLI smoke 均成功，见 raw/cli_* 日志。数据加载显式 next，优化器顺序不变。
- 初次 GPU 新测试 5 pass/1 fail：CuPy 先初始化后加载 Torch 导致混合 CUDA runtime 的 cudaGetDriverEntryPointByVersion 符号错误；测试与混合框架实验先导入 Torch，CLI 也在 Torch GPU 路径先导入再初始化设备。重跑 6 pass，原失败日志保留。
- 初次 CPU CLI 因 helper 初建于错误目录 import 失败，移动至 hytorch/monitor_loop.py 后通过，保留原日志。
- 初次聚合脚本 mode 字段重复导致 TypeError，修复为 monitor_components/monitor_mode 后完整重跑成功。旧 partial raw/runs 目录保留但不用于正式统计；RUNS.csv 明确列出纳入的 run_dir。
- 未修改底层 buffer 生命周期或 kernel，无新增 CUDA 内存处理；本轮未运行 Compute Sanitizer，不将此前结果冒充本轮验证。
- raw/source_integrity.json 证明本轮 CUDA source/.so、core、functions、conv、优化器相对快照未变。所有生产变化见 stage-only git_diff.patch。
''')
put('PPT_SUMMARY.md','''# 课程汇报建议（六页）

1. **为什么监控**：GPU 算子快不等于完整训练快；区分主机提交、GPU 区间、纯 kernel 与硬件状态。目标：为 HyTorch 增加可复用、可关闭、可验证的监控能力。
2. **系统设计**：TrainingMonitor + 显式 DataLoader next + 当前 stream events + 后台 NVML/psutil。UUID/PCI 映射，独立 CSV/JSON，异常清理，缺失指标 null。默认关闭，不改变算法。
3. **硬件趋势**：用 figures/01_gpu_utilization_memory、02_gpu_temperature_power、03_cpu_utilization（均有 PNG/SVG）。GPU 均值 38.8%，28°C、43.94W；CPU 进程约 1.05 核。38 个训练内采样点/7.62s，不逐 step 归因。整卡采样峰值 987.44MiB 与 CuPy hook 活跃峰值16.16MiB 是不同范围。
4. **训练瓶颈**：用 figures/04_step_time_distribution、05_phase_breakdown。Backward 主机 2.23ms/step，独立 Nsight 72/142 kernels、809μs纯 GPU/step。主机提交间隙值得研究；65% trace 间隙不是 SM occupancy。吞吐等效3.809ms/step，诊断完成延迟4.101ms，不能混用。
5. **监控成本**：用 figures/06_monitor_overhead。12轮配对：完整生命周期 timing+5.34%、resources+1.40%（CI跨0）、all+8.50%；all仅训练窗口+4.98%。所有异常值保留，正式结果无 profiler。347测试通过、2已有CPUskip，1884数值比较无门限失败。
6. **结论与后续方向**：日常趋势仅开 resources，定位阶段再开 timing，吞吐验收关闭。下一步可研究 backward 的主机包装和 launch 合并、分块事件导出、更长资源采样；本轮未实施新算子优化。PyTorch FP32 参考0.430s vs HyTorch0.731s，说明仍有框架性能差距，不是本轮优化收益。

所有图片标明 A100、batch4、固定配置、采样间隔与测量方法。不要把 Event 区间称为纯 kernel 时间，或把采样最大值称为全卡精确峰值。
''')
# Stage-specific source snapshot and diff.
paths=['hytorch/profiling.py','hytorch/training_monitor.py','hytorch/monitor_loop.py','benchmarks/bench_cuda_training.py','requirements-cuda.txt','examples/training_monitor.py','tests/test_training_monitor.py','tests/test_training_monitor_cuda.py']
diff=[]
for p in paths:
 src=R/p; dst=O/'source'/p;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
 old=O/'baseline'/p
 diff.extend(difflib.unified_diff(old.read_text().splitlines(True) if old.exists() else [],src.read_text().splitlines(True),fromfile='a/'+p if old.exists() else '/dev/null',tofile='b/'+p))
put('git_diff.patch',''.join(diff))
checks=[]
for x in json.loads((O/'raw/baseline_hashes.json').read_text()):
 p=R/x['path']; h=hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
 checks.append(dict(path=x['path'],baseline_sha256=x['sha256'],current_sha256=h,unchanged=h==x['sha256']))
(O/'raw/source_integrity.json').write_text(json.dumps(checks,indent=2))
changed=[x['path'] for x in checks if not x['unchanged']]
print('Baseline changed:',changed)
put('SOURCE_CHANGES.md','''# 本阶段源码变化

本阶段基线是开始工作时的脏工作区，不是 Git HEAD。baseline/ 保存原源码与动态库；raw/git_diff_before.patch 是既有改动。git_diff.patch 只表示本阶段前后差异，不应将此前算法优化归入本轮。source/ 为当前修改文件快照。

| 文件 | 改动 |
|---|---|
| hytorch/training_monitor.py | 新增可选 NVML/psutil 采样、runtime 身份映射、事件计时、生命周期、导出、allocator诊断 |
| hytorch/monitor_loop.py | 新增通用显式 next 训练 helper、阶段边界和回调接口 |
| hytorch/profiling.py | 仅重新导出 TrainingMonitor，保留原工具 |
| benchmarks/bench_cuda_training.py | 可选 monitor 分支、CLI、最终已接受配置选择、Torch CUDA import顺序、NVTX helper |
| requirements-cuda.txt | 增加可选 NVML Python binding 依赖说明 |
| examples/training_monitor.py | 通用 CPU/CUDA 示例 |
| tests/test_training_monitor.py | 22 项 CPU 测试 |
| tests/test_training_monitor_cuda.py | 6 项 GPU 测试 |

本轮未改 GEMM/Col2im/dWeight/Pooling kernel、dispatch、core/自动求导或 SGD 数学。基线 SHA 与现 SHA 见 raw/source_integrity.json；变化的既有文件：'''+', '.join(changed)+'''.

诊断与复现脚本位于 diagnostics/；所有原始失败日志保留。没有 Commit、Push、PR。基线 .so 与 Nsight 二进制不进入压缩包，SHA/服务器路径另列。
''')
