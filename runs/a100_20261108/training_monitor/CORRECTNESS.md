# 正确性与测试证据

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
