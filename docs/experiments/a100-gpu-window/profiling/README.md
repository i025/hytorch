# Profiling：原始结果与权限限制

Nsight Systems **PASS**：实际 DonkeyNet combined synthetic 时间线采集成功，完整 CUDA/NVTX 汇总 CSV 保存在本目录。原始二进制与 SQLite 留在服务器 `/home/xymo/projects/hytorch/runs/nsys_window_01.nsys-rep`、`nsys_window_01.sqlite`，未上传二进制。

Nsight Compute Tile 16/32 均 **BLOCKED**：实际尝试返回 ERR_NVGPUCTRPERM，当前用户没有 GPU 性能计数器权限。不使用 sudo、不修改系统权限。没有测得 bandwidth、SM occupancy、warp scheduler/shared memory 效率，不从普通计时推测这些指标。完整错误与原始命令见 ncu-tile16-monitor-01/、ncu-tile32-monitor-01/。

Compute Sanitizer **PASS**：GPU 验收 37 passed / 0 failed / 0 skipped，ERROR SUMMARY: 0 errors。包含子进程检查；原始 tests.log/XML/environment.json 在 sanitizer-results-01/。

## 发现的实际 dtype 问题

初次 Systems 报告中出现 cupy_copy__float32_float64、float64 GEMM 与更新转换。虽然输入和权重 FP32，HyTorch MSE/Tanh 中转换为零维 float64/int64 的常数令 GPU 梯度提升为 FP64；因此修复前训练结果保留，但不能作为严格 FP32 对比。修复和复测另行保存。

nsys-analysis-before-fix/ 是从原始 SQLite 提取的时间线图和全部间隔 CSV，标注为日志可视化，不是真实终端截图。interval coverage 只是 NVTX 训练包络内 kernel+memcpy 区间的并集覆盖率，不是 SM 利用率。NVTX 范围主要反映主机提交开销，GPU 异步执行不能简单按主机范围归因。

初次 trace 全程包含初始化和 warmup，默认 kernel 汇总亦含这些部分；分析脚本单独给出训练 NVTX 包络内数量。插桩耗时不混入正常 benchmark。

## 修复后复测

nsys_stats_fixed_01_* CSV、nsys-monitor-after-fix/ 和 nsys-analysis-after-fix/ 保留物理 GPU 2 上重新采集的同参数训练。全 trace kernel 2664 次，训练 NVTX 包络内 2368 次；包络 161.090596 ms，kernel+memcpy 并集 23.611117 ms（覆盖率 14.657%，不是 SM 利用率）。未观察到 float64/f64/d884 kernel 名称。主机 backward NVTX 占已标注阶段时间约 61.8%；独立 phases 训练也应一并查看。数据表明主机提交/同步与很多短 kernel 值得下一轮调查，不能仅据此断言某硬件瓶颈。

原始修复后报告 /home/xymo/projects/hytorch/runs/nsys_dtype_fixed_01.nsys-rep 与 .sqlite。与修复前物理卡/时段不同，插桩时间差不能当作严格修复收益。内存带宽、occupancy、warp scheduler 仍受 NCU 权限限制。
