# 实际问题、修复与数值诊断

## 训练 FP32 类型提升：已修复

原始证据：profiling/nsys_stats_window_01_cuda_gpu_kern_sum.csv 出现 FP64 转换与 GEMM。dtype-before-02/dtypes.json 进一步复现：CPU Loss float64；GPU TinyNet 全连接层梯度 float64、DonkeyNet 所有参数梯度 float64。诊断脚本以断言失败结束，完整输出在 dtype-before-monitor-02/；此前拒绝使用忙碌 GPU 的记录仍在服务器 runs/dtype_before_monitor_01/，没有计算。

根因：hytorch/core.py 原有标量包装把 Python 常数变为零维 int64/float64 数组。MSE.backward 的 2/N、Tanh.backward 的 1 在 CuPy 上产生类型提升；MSE.forward 的 NumPy 标量除 Python 整数使 CPU Loss 变为 float64。修复仅在 hytorch/functions.py 的 MSE/Tanh 中显式创建与输入同 dtype 的常量，不改变公式、容差或 CUDA 内核，不改变全局混合类型运算语义。

修复前：dtype 审计失败，但旧测试没有检查这些类型；普通数值误差检查仍通过。
修复后：dtype-after-01/ 显示 CPU/GPU 两个模型的预测、Loss、全部参数梯度和更新后权重均为 float32。新增 CPU float32/float64 MSE+Tanh 梯度回归和两个 GPU 模型的梯度类型回归。主机 25 passed / 0 failed / 0 skipped；GPU 39 passed / 0 failed / 0 skipped（validation-after-dtype-fix-01/）。PyTorch 两项主机对齐再次实际执行，容差未改。

完整重跑 39 次 throughput + 13 次 phases，均保留在 training-comparison/*-fixed-*。Nsight 复测没有观察到 float64/f64/d884 kernel 名称；此证据加上实际参数 dtype 断言支持本次 FP32 链路验收。修复前后使用物理 GPU 0 与 2、不同时段，不能把两组耗时差全部归因为修复；最终框架对比只使用同一修复后时段、同一 GPU 2 的匹配结果。

## GEMM 2048/4096：仍未达到原定逐元素容差

该问题与训练标量类型提升独立，GEMM 输入和输出均 FP32。gemm-diagnosis-01/errors.json 独立重复完整矩阵比较：N=2048 每个 GPU 方法 10 个元素超差；N=4096 每个 GPU 方法 2418 个元素超差。CuPy 与自研各变体在这两个规模结果逐元素相同；独立 PyTorch supplement 也有相同误差统计，因此不是仅自研实现发生的偏差。

fp64-dot-samples.json 对每组最大的三个容差比位置以 FP64 dot 独立核查；fma-reproduction.json 再按递增 K 顺序用 libm.fmaf 做 IEEE FP32 FMA 累加。2048/4096 全部 36 个记录的采样结果与 GPU 精确相同，支持长 FP32 累加与 NumPy BLAS 归约顺序不同导致取消误差的判断。1024 自研的 15 个采样同样精确匹配，而 CuPy 三个采样不相同。采样诊断不是全矩阵高精度验收，也没有把失败重新标成 PASS。

例如 4096 的 [1821,4053]：GPU 0.2896142602，NumPy FP32 0.2899932861，FP64 dot 0.2899966041，允许差 0.0001289993；GPU vs FP64 约 0.0003823439。原定 atol=rtol=1e-4 确实未通过。未放宽阈值、未优化或重写内核。下一轮应先共同确定稳定的累加策略与高精度参考，再比较性能。

## PyTorch GEMM 补充

torch-gemm-01/ 是另一次同输入、同 FP32、TF32 关闭、预分配输出、5 warmup / 10 CUDA Event 测量。不要把该时段 CuPy 时间直接拼接到七配置 baseline。N=1024 Torch 0.246579 ms，CuPy 0.326861 ms，均通过；2048/4096 同样未通过既定容差，速度记录仅作诊断。

实际命令、设备、时间、监控及 stdout 均在各 *-monitor-* 目录。生成脚本位于 ../scripts/diagnose_gemm.py、diagnose_accumulation.py、torch_gemm.py、audit_training_dtype.py。
