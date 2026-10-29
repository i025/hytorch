# 修复 FP64 类型提升后的历史 Nsight Systems 复核

本页只分析已经存在的历史记录，不运行 GPU。输入是 [原始采集命令](../a100-gpu-window/profiling/nsys-monitor-after-fix/launch.json) 对应的 `/home/xymo/projects/hytorch/runs/nsys_dtype_fixed_01.sqlite`（1,376,256 字节）与 `.nsys-rep`（562,858 字节）；历史文件没有修改。SQLite 内全部 CUDA API、Kernel、Memcpy、Synchronization 与 NVTX 行已导出到 [historical-nsys-fixed-01](results/historical-nsys-fixed-01/)，保留原始纳秒时间戳、SQLite rowid 和 correlationId。公开 CSV 中的主机进程/线程编号替换为稳定标签；原始编号仍保留在服务器 SQLite，以及首次未归一化导出的 `runs/stage4a_historical_analysis_initial_01/`。

这次历史配置为物理 GPU 2、逻辑 GPU 0，DonkeyNet、64 个合成样本、Batch 4、FP32、SGD、Seed 42、Dropout 0、Warmup 2；Combined 参数实际是 `matmul=shared, conv=im2col, pool=cupy, tile=16`。训练模式是 `throughput` 并启用 `--nvtx --no-fingerprints`。这是含插桩的 16 Step 采集，不能与正式非插桩计时混合。

## 统计范围与精确复现

| 指标 | 本次从 SQLite 重新计算 | 范围 |
|---|---:|---|
| 全 Trace Kernel 数 | 2,664 | 含 2 次 Warmup 与 16 次正式 Step |
| 训练 NVTX 包络内 Kernel 数 | 2,368 | GPU 区间与第一个训练 NVTX 开始至最后一个结束的包络相交 |
| 由训练 NVTX 内 API 提交的 Kernel 数 | 2,368 | 通过 correlationId 关联提交 API，再按同一线程的 NVTX 归因 |
| 训练 NVTX 包络 | 161.090596 ms | 不是所有 NVTX 长度之和，也不是整个进程时间 |
| 包络内 Kernel + Memcpy 区间并集 | 23.611117 ms | 相交区间合并且裁剪到包络 |
| 区间覆盖率 | 14.6570424% | 23.611117 / 161.090596，**不是 GPU 或 SM 利用率** |
| 训练提交 Kernel 时长之和 | 22.977883 ms | 只含 Kernel，区别于并集与 Memcpy |
| 全 Trace Kernel 时长之和 | 25.847437 ms | 含 Warmup |

历史报告中的 2,664、2,368、14.657% 因而可以精确复现。每个 Step 都有 148 次 Kernel：Forward 39、Backward 77、Update 32；两个 Warmup 共 296 次。没有找不到提交 API 的 Kernel。未出现包含 `float64`、`f64` 或 `d884` 的 Kernel 名称；名称检查只是辅助证据，严格 FP32 状态还依赖先前类型审计和修复记录。

归因不以 GPU 开始时间落在哪个 NVTX 中为准，而使用 `Kernel.correlationId → CUDA API.correlationId → API 所在线程的 NVTX`。14 个 Backward Kernel 和 15 个 Update Kernel 的执行区间未完全处于提交时的主机 NVTX 内，说明异步执行确实会越过主机边界。

## Phase 与 Backward 细目

[完整 Phase 表](results/historical-nsys-fixed-01/phase-summary.csv) 的时间口径如下；GPU 列是归属于该 Phase 的 Kernel 执行时间之和，CPU 列是主机 NVTX 墙钟时间，两者可以重叠，不能相加。

| Phase | 主机 NVTX 累计 ms | CUDA API 累计 ms | 其中 Launch API ms | GPU Kernel 累计 ms | Kernel 数 |
|---|---:|---:|---:|---:|---:|
| Data | 2.031036 | 0 | 0 | 0 | 0 |
| Transfer | 2.974599 | 0.690515 | 0 | 0 | 0 |
| Forward | 43.298793 | 4.644542 | 4.644542 | 8.239914 | 624 |
| Backward | 95.977064 | 10.400689 | 9.561872 | 12.940981 | 1,232 |
| Update | 11.025846 | 2.614576 | 2.614576 | 1.796988 | 512 |

Backward 占全部已标注 NVTX 主机时长 155.307338 ms 的 **61.7982%**。Backward 中 866 / 1,232 个 Kernel 小于 10 µs；Update 中全部 512 个 Kernel 小于 10 µs，平均 3.509742 µs。主机 API 提交成本与大量短 Kernel 值得关注，但仅靠这个 Trace 不能把全部剩余时间归因给 Kernel Launch。

Backward GPU Kernel 分类如下。分类表是 Kernel 级统计，历史 Trace 没有具体 Autograd 节点 NVTX，因而不能将全部 cuBLAS、copy 或 reduction 精确分配给某个 Conv/Linear 节点。

| 类别 | Kernel 数 | GPU 累计 ms | 占 Backward Kernel 时长 |
|---|---:|---:|---:|
| cuBLAS GEMM/GEMV/部分和归约 | 320 | 3.635086 | 28.0897% |
| CuPy Col2im | 80 | 3.391406 | 26.2067% |
| CuPy Copy/布局变换 | 240 | 1.964265 | 15.1786% |
| CuPy Im2col | 80 | 1.568225 | 12.1183% |
| CuPy Elementwise | 384 | 1.379434 | 10.6594% |
| CuPy/CUB Reduction | 128 | 1.002565 | 7.7472% |

单个 Kernel 名称中 `col2im` 累计最长：80 次，平均 42.392575 µs，最大 88.384 µs。`cupy_copy__float32_float32` 为 240 次、平均 8.184437 µs；`im2col` 为 80 次、平均 19.602812 µs。仅 Col2im + Im2col 已占 Backward GPU Kernel 时长 38.3250%，但只占整个插桩训练包络 3.0788%；这两个不同分母不能混用。

自研 `gemm_shared`、`im2col_kernel`、`conv_pack` 只归属 Forward，没有出现在 Backward。完整名称与所有次数保存在 [kernel-summary.csv](results/historical-nsys-fixed-01/kernel-summary.csv) 和 [kernel-events.csv](results/historical-nsys-fixed-01/kernel-events.csv)。

## 拷贝、同步、分配与主机间隔

- 训练 Transfer 提交了 32 次 H2D，共 10,137,856 字节，GPU Memcpy 累计 0.596226 ms。包络内另有 15 次 D2H，共 60 字节、0.038752 ms，发生在 Phase NVTX 外；与每 Step 获取标量 Loss 的源码路径一致。最后一个 Step 的 Loss 读取在最后 Update NVTX 之后，故不包含在此包络。
- 包络内 Phase 之外有 30 次 `cudaStreamSynchronize`，API 时长之和 0.173473 ms；15 次 `cudaMemcpy` API 为 0.404813 ms。API 耗时是主机调用时间，不是 Memcpy GPU 执行时间；同步 API 时长也不能与 GPU 工作简单相加。
- Forward/Backward/Update NVTX 内没有 `cudaMalloc`、`cudaFree` 或同步 API。全 Trace 的显存分配、HostAlloc、Free 在初始化、Warmup 或退出范围内；因此不能把全程 `cudaMalloc` 12.441227 ms 或 `cudaFree` 14.877611 ms 直接说成正式训练每步开销。CuPy 内存池内的 Tensor 分配不一定调用 CUDA 分配 API，需要获准的新采集验证；本轮该项为 NOT_RUN。
- 包络内记录到的 CUDA API 区间并集是 18.944959 ms；其余 142.145637 ms 不在这些 API 内。Kernel + Memcpy 之外的区间合计 137.479479 ms，其中与 CUDA API 相交 17.084919 ms，与 Launch API 相交 15.479322 ms，与同步 API 相交 0.173473 ms。这些区间关系不是互斥的主机/设备耗时分解。
- 第一正式 Step 的 Forward / Backward / Update NVTX 分别为 11.469087 / 30.056986 / 1.596600 ms；其余 15 Step 均值为 2.121980 / 4.394672 / 0.628616 ms。最大无已记录 GPU Kernel/Memcpy 间隔为 17.655379 ms，出现在第一 Step Backward，其中仅 0.304583 ms 落在 CUDA API 内。因此两次 Warmup 不能保证该短 Trace 已排除首步异常。

历史采集使用 `--sample=none --cpuctxsw=none`，缺少 CPU 栈和线程调度数据。无法进一步将上述间隔分解为 Python/CuPy 运算、操作系统调度、Profiler 开销或其他主机等待；也未获得真实 SM Occupancy、带宽或 Warp Stall 计数器。不能声称“85.343% 时间 GPU 利用率为零”，只能说该进程在训练包络中有这些未覆盖的已记录设备活动区间。

## 可复现命令与完整数据

在仓库根目录执行；输出必须是不存在的新目录：

```bash
python3 docs/experiments/donkeynet-backward-profiling/scripts/analyze_history.py \
  --input runs/nsys_dtype_fixed_01.sqlite \
  --output docs/experiments/donkeynet-backward-profiling/results/historical-nsys-fixed-01
```

- [summary.json](results/historical-nsys-fixed-01/summary.json)：统计定义与所有总数。
- [schema.json](results/historical-nsys-fixed-01/schema.json)：实际 SQLite 表结构。
- [nvtx-events.csv](results/historical-nsys-fixed-01/nvtx-events.csv)：全部 80 个训练 Phase 范围。
- [cuda-api-events.csv](results/historical-nsys-fixed-01/cuda-api-events.csv)：完整 API 行、名称、Phase 和 correlationId。
- [kernel-events.csv](results/historical-nsys-fixed-01/kernel-events.csv)：全部 2,664 个 Kernel 及提交关联。
- [memcpy-events.csv](results/historical-nsys-fixed-01/memcpy-events.csv)、[synchronization-events.csv](results/historical-nsys-fixed-01/synchronization-events.csv)：完整事件。
- [phase-summary.csv](results/historical-nsys-fixed-01/phase-summary.csv)、[kernel-summary.csv](results/historical-nsys-fixed-01/kernel-summary.csv)、[api-summary.csv](results/historical-nsys-fixed-01/api-summary.csv)、[memcpy-summary.csv](results/historical-nsys-fixed-01/memcpy-summary.csv)：三种明确统计范围的汇总。
- [gpu-activity-gaps.csv](results/historical-nsys-fixed-01/gpu-activity-gaps.csv)：全部设备活动间隔与 API 交集。

已准备的 Autograd 节点插桩可用于区分 Conv 输入梯度、Conv 权重梯度、Linear 梯度和布局变换，但本轮新 GPU 采集为 **NOT_RUN**，不会自动重采集。历史数据仅支持这些候选方向，不能代替新的匹配实验与节点级归因。

## 后续节点级分析工具（本轮新 GPU 采集为 NOT_RUN）

[analyze_trace.py](scripts/analyze_trace.py) 是供新 Trace 使用的离线解析器。支持 `forward/backward/update/data/transfer` 或 `PHASE/` 前缀的主机阶段标签，其他嵌套 NVTX 作为算子标签；`OP/bwd:Conv2d → OP/fwd:Deconv2d → OP/internal:col2im` 会分别保留独占与包含子节点的统计。PyTorch 原生标签的序列号从聚合名中删除，原标签仍保留。跨线程工作按提交 API 时刻唯一的主机 Phase 区间归属，并单独标记；从不通过异步 GPU 执行时刻决定 Phase。

工具输出 `operator-hotspots.csv` 的 `exclusive` 行将每个 Kernel 归入最深算子一次，`inclusive` 行为祖先包含子调用的统计，不能将不同祖先的 inclusive 行相加。没有节点 NVTX 的历史记录只会得到 `unlabeled:backward`，不会虚构 Conv2d 整体耗时。全部 API、Kernel、Memcpy、NVTX 事件及关联路径分别导出，`phase-summary.csv` 同时提供主机时间、API 时间、Kernel 时间之和与区间并集。

在旧 SQLite 上执行该通用工具，所有 Phase 的 Kernel 数/时间和 Memcpy 数/时间与本页历史专用分析一致，覆盖率和总数也完全一致。额外 CPU 合成 SQLite 测试验证了嵌套父子归因、GPU 越过主机范围、跨线程节点和独占不重复计数；这些合成纳秒时间戳不是 GPU 测量数据。

```bash
python3 docs/experiments/donkeynet-backward-profiling/scripts/analyze_trace.py \
  --input runs/nsys_dtype_fixed_01.sqlite \
  --output runs/stage4a_generic_history_analysis_01 \
  --label historical-fixed-validation
python3 docs/experiments/donkeynet-backward-profiling/scripts/check_trace_analysis.py \
  --output runs/stage4a_trace_attribution_check_01
```

通用工具复核输出保留在上述 `runs/` 目录。CPU 合成检查 01 为 PASSED；随后为解析器补充“相同 correlationId 的多个 API 必须在同一线程且彼此嵌套”的必要一致性检查及 schema 导出，最终 CPU 合成复测另存 `runs/stage4a_trace_attribution_check_02/`，仍为 PASSED，旧结果不覆盖。两次检查均未失败，本轮没有需要修复的 GPU 或数值错误。本轮没有获取新 Conv2d 节点级数据，因此 Conv2d Backward 整体占比仍为 **NOT_MEASURED**；直接可确认的 Im2col + Col2im 时间仅是 GPU Kernel 时间下界。
