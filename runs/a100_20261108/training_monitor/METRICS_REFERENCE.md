# 指标字典

空 CSV 字段对应 JSON null，表示不适用、缺失或不支持，结合 status_json/错误判断；不能改成正常 0。时间列显式后缀 _s（秒）、_ns（纳秒）、_us（微秒）、_ms（毫秒）；容量为 bytes，图中 MiB=2^20 bytes。

| 列/字段 | 来源、范围与解释 |
|---|---|
| timestamp_unix_s / monotonic_ns | 采样/step开始时间；Unix用于跨记录观察，perf_counter单调时钟用于持续时间 |
| elapsed_s / scheduled_monotonic_ns / schedule_delay_s | 相对monitor开始、计划时刻、实际调度延迟；非强制等间隔 |
| sample_interval_s / sample_collection_s | 目标周期/一次采集CPU耗时；NVML自身统计窗口不等于目标周期 |
| gpu_uuid / physical_gpu_index / logical_cuda_index / pci_bus_id / cuda_visible_devices | runtime UUID/PCI映射NVML后的身份，不按逻辑编号直接索引NVML |
| gpu_utilization_pct | NVML整卡忙碌统计，非进程独占、非achieved occupancy |
| memory_utilization_pct | NVML内存读写忙碌统计，不是used/total容量比；0可能是有效读数 |
| gpu_used_bytes / gpu_total_bytes | NVML整卡显存使用/总量，包含框架池外分配和context等 |
| process_gpu_used_bytes | NVML当前PID显存；不支持或unknown sentinel时null |
| compute_processes_json | NVML计算进程列表与PID显存，不能保证观测间隙内没有其他任务 |
| gpu_temperature_c / gpu_power_w | NVML整卡摄氏温度/瓦特功耗，非进程能耗 |
| sm_clock_mhz / memory_clock_mhz | 当前时钟，MHz；没有锁频 |
| cupy_active_bytes / cupy_reserved_bytes | 默认CuPy池used_bytes/total_bytes，不涵盖所有CUDA分配 |
| torch_allocated_bytes / torch_reserved_bytes | Torch allocator采样，显式目标device，不适用于HyTorch记录 |
| system_cpu_pct | psutil整机CPU，多核归一化；会受其他用户任务影响 |
| process_cpu_pct | psutil进程CPU；100%代表一个逻辑核，可超过100；首点null |
| process_rss_bytes / process_threads / pid | psutil进程驻留内存、线程数、PID |
| step_id / host_phase / scope | 采样开始时CPU标签/测量范围，不能当GPU kernel阶段归因 |
| status_json / status / error | 每指标失败或生命周期状态；正常0和缺失严格分开 |
| epoch / batch_size / data_bytes | 当前训练元信息；data_bytes为传输输入数组nbytes之和，非实际PCIe流量 |
| *_cpu_s | perf_counter主机阶段elapsed；含调用阻塞，不是纯Python指令时间 |
| *_gpu_event_s | 当前stream事件区间；可能包含GPU空闲/提交间隙，非kernel累加 |
| host_iteration_s | 包括显式取数至update的主机提交迭代时间，不等于异步GPU完成延迟 |
| step_wall_s | CPU完成迭代或latency模式step末同步完成时间；CUDA throughput留空 |
| step_gpu_interval_s / gpu_event_status | step流事件区间及可用性；跨流并发不完整归因 |
| start_monotonic_ns / end_monotonic_ns | step主机起止边界 |
| tag / model / framework / monitor_components / monitor_mode / repeat / run_dir | 聚合时添加的实验分组；step原mode为throughput/latency |
| train_start_ns / train_end_ns / inside_train | 正式训练窗口与样本开始时刻是否落入窗口；采集结束可能越界 |

## 汇总与其他 CSV

RUNS.csv：wall_s 包含monitor启动/停止/导出；training_window_s为训练循环加末尾同步；monitor_setup_finalize_export_s是两者差；samples_per_s用wall，training_window_samples_per_s用训练窗口。samples=处理总数（256×3），不是去重样本数。order为当轮顺序，loss为最终step loss。

MONITOR_OVERHEAD.csv：metric区分两种wall；mean/median/std/min/max单位秒；paired_delta为与同轮disabled的差（秒）；95%CI用df11配对t。ratio_of_means_overhead_pct与逐对百分比平均paired_overhead_mean_pct不是同一统计量。raw/OVERHEAD_PAIRS.csv保留每对差。

STEP_TIMINGS/CPU_SAMPLES/GPU_SAMPLES为逐条原始聚合。TIMING_SUMMARY、RESOURCE_SUMMARY、BOTTLENECK_SUMMARY为标注group/metric的汇总，不同分组不能合并；formal_all共2304steps，trend为30epochs。PROFILE_PHASES/PROFILE_KERNELS来自独立20-step Nsight CUPTI，时间为实际kernel或CUDA API主机elapsed；kernel数/step与time/step列已按20步归一化。raw/timeline.json活动并集按区间合并，不用kernel和API耗时相加解释wall。

summary JSON 的 sampled peaks 只是采样最大值。allocator hook峰值仅主线程CuPy默认pool分配点；Torch allocator maxima是重置后计数器范围。均不代表全卡毫秒级真实峰值。hook 的phase/step是CPU标签。CUDA API时间只在独立Nsight获得，普通TrainingMonitor不会伪造API分项。
