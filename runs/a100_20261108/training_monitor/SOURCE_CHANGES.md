# 本阶段源码变化

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

本轮未改 GEMM/Col2im/dWeight/Pooling kernel、dispatch、core/自动求导或 SGD 数学。基线 SHA 与现 SHA 见 raw/source_integrity.json；变化的既有文件：hytorch/profiling.py, benchmarks/bench_cuda_training.py, requirements-cuda.txt.

诊断与复现脚本位于 diagnostics/；所有原始失败日志保留。没有 Commit、Push、PR。基线 .so 与 Nsight 二进制不进入压缩包，SHA/服务器路径另列。
