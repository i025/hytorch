# 课程汇报建议（六页）

1. **为什么监控**：GPU 算子快不等于完整训练快；区分主机提交、GPU 区间、纯 kernel 与硬件状态。目标：为 HyTorch 增加可复用、可关闭、可验证的监控能力。
2. **系统设计**：TrainingMonitor + 显式 DataLoader next + 当前 stream events + 后台 NVML/psutil。UUID/PCI 映射，独立 CSV/JSON，异常清理，缺失指标 null。默认关闭，不改变算法。
3. **硬件趋势**：用 figures/01_gpu_utilization_memory、02_gpu_temperature_power、03_cpu_utilization（均有 PNG/SVG）。GPU 均值 38.8%，28°C、43.94W；CPU 进程约 1.05 核。38 个训练内采样点/7.62s，不逐 step 归因。整卡采样峰值 987.44MiB 与 CuPy hook 活跃峰值16.16MiB 是不同范围。
4. **训练瓶颈**：用 figures/04_step_time_distribution、05_phase_breakdown。Backward 主机 2.23ms/step，独立 Nsight 72/142 kernels、809μs纯 GPU/step。主机提交间隙值得研究；65% trace 间隙不是 SM occupancy。吞吐等效3.809ms/step，诊断完成延迟4.101ms，不能混用。
5. **监控成本**：用 figures/06_monitor_overhead。12轮配对：完整生命周期 timing+5.34%、resources+1.40%（CI跨0）、all+8.50%；all仅训练窗口+4.98%。所有异常值保留，正式结果无 profiler。347测试通过、2已有CPUskip，1884数值比较无门限失败。
6. **结论与后续方向**：日常趋势仅开 resources，定位阶段再开 timing，吞吐验收关闭。下一步可研究 backward 的主机包装和 launch 合并、分块事件导出、更长资源采样；本轮未实施新算子优化。PyTorch FP32 参考0.430s vs HyTorch0.731s，说明仍有框架性能差距，不是本轮优化收益。

所有图片标明 A100、batch4、固定配置、采样间隔与测量方法。不要把 Event 区间称为纯 kernel 时间，或把采样最大值称为全卡精确峰值。
