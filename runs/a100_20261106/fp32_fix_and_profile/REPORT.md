# FP32 类型修复与 A100 性能复评

## 结论

已修复 MSE 与 Tanh 内部标量导致的非预期 FP64 梯度。实测 DonkeyNet 的全部 Conv2D/Linear 激活、dInput/dWeight/dBias、参数和 SGD 中间表达式为 FP32；显式 FP64 一阶/二阶导数测试通过。生产修改仅两个 backward 函数，未改 CUDA kernel、dispatch、全局类型规则或优化器。

同一 GPU 窗口 5 轮交替配对，Native wall 从 0.933279 s 降至 0.851444 s，减少 8.77%，均值比约 1.096×。这包含 FP64→FP32 计算/内存路径改变，不是 CUDA 算法加速。

conv4/5 的现有 register-tiled GEMM 在实际形状上明显比原 shared 慢；最终建议首先审查通用小矩形 GEMM 分块/dispatch，再研究 col2im 和 dWeight im2col/布局。

## 源码版本和根因

分支 main，HEAD `edb6fbe3bfaa9eed461addd29e6943b22e28216e`，开始即有此前未提交工作。前后完整 diff 与本轮增量分别保存。CUDA 源码/runtime/已编译库 SHA-256 保持不变。

`core.py:264 mul`、`:300 rsub` 经 `:170 as_array` 把 Python float/int 包成强类型 CuPy FP64/int64 零维数组。前向 loss 为 FP32，但旧 `MeanSquareError.backward` 的 `gy * diff * (2./len(diff))` 在 id=86 的第二次 Mul 首次制造 FP64；旧 Tanh 的 `1-y*y` 也会独立提升类型。父函数记录 id=83 的 FP64 输出是其返回结果，不是更早的算术执行。

旧梯度沿 MatMul、Conv2D backward、Add 累积传播；SGD 原地减法把 FP64 中间更新写回 FP32 参数，不能据参数 dtype 推断梯度正确。Native 仅用于前向；卷积/Linear backward 仍走 CuPy/cuBLAS。输入数据生成和 DataLoader 在 CPU，随后 H2D。

修复前后代码、适用范围和最终整数边界保护见 `SOURCE_CHANGES.md`。最后的 result_type 元数据保护保持本次 GPU 实测 FP32/FP64 系数类型不变；此收尾版本未重新占用 GPU，明确保留 GPU 已测源码供审计。

同类但未修改：全局 Python 标量与 Variable 的强数组提升、Pow.backward/简单 loss/其他激活中的常量可能有相同风险。它们不在本次 DonkeyNet 的实际触发路径，扩大类型规则需要独立设计。显式 FP64 输入并未被强制降精度。

## 正确性

| 检查 | 结果 |
|---|---|
| 原有主机测试 | 22 passed |
| 最终 dtype/整数边界 CPU 回归 | 9 passed，12 deselected，未用 skip 冒充通过 |
| 新增 GPU dtype/数值/DonkeyNet 回归 | 10 passed，10 deselected |
| 原有 GPU 单元测试 | 45 passed，0 skipped |
| workload 插桩对比 | 修复前、修复后各自 loss/全部梯度/更新后权重逐位一致 |
| 5 轮训练重复 | 各模式自身最终权重 hash 一致 |
| 与 PyTorch 对齐 | 所有模式通过 atol=2e-5 / rtol=2e-4 权重检查；最终 loss 误差均远小于 2e-5 |

GPU 回归包含 CPU PyTorch 算子参考以及完整 CUDA DonkeyNet 参考。MSE 通用算子按 batch 归一化，与 PyTorch 对照使用 sum/batch；训练输出为 (4,1)，因此与 PyTorch mean MSE 完全一致。

`DTYPE_AUDIT.csv` 覆盖每层和 SGD；`raw/*_all_tensor_dtypes.csv` 保留完整算子张量审计。修复前 94 个已记录输出为 FP64，修复后同一 workload 的 178 个浮点输出均为 FP32。所有浮点输入/输出也已检查没有 FP64。float64 MSE/Tanh/Conv/Linear 与梯度保持 float64。

真实 shape 微基准的原始严格 gate（NumPy FP32，atol=rtol=1e-4）在 (36,600,2632)、(24,75,12152) 上的 original/current 两实现均失败。未放宽门限；这些失败保留在 CSV，不能称“全部严格 GEMM 数值验证通过”。两 native 实现所有 shape 逐位一致，三实现相对 FP64 L2 最大 1.90e-6（补充指标，不替代严格 gate）。此前 2048/4096 大方阵失败记录也保留在 previous_gemm_* 文件中。

## 配对训练

seed=42；相同初始化和数据 hash；256 个合成样本，batch=4，3 epochs，共 192 steps/768 sample visits；SGD lr=.001，无 momentum/weight decay/dropout；各模式 10 次预热；5 轮正反交替，每次恢复相同初始权重。

wall 包括 DataLoader、pageable H2D、前后向、更新与最终同步，无逐步 loss 读取/日志/sync。所有模式共用轻量 phase NVTX；loss 与 snapshot 在计时后读取。

| 实现 | wall s，均值±样本标准差 | samples/s 均值 | 最终 loss | 权重最大绝对误差 vs Torch |
|---|---|---|---|---|
| before_native | 0.933279 ± 0.015155 | 823.08 | 0.000255663384451 | 2.98e-08 |
| after_native | 0.851444 ± 0.014637 | 902.21 | 0.000255663209828 | 2.98e-08 |
| after_cupy | 0.847630 ± 0.011519 | 906.19 | 0.000255663617281 | 4.19e-08 |
| torch | 0.444945 ± 0.023079 | 1729.80 | 0.000255662802374 | 0 |

逐轮配对加速比 1.09619、1.10264、1.08867、1.09155、1.10165，均值±样本 SD=1.09614±0.00611。没有将旧窗口绝对时间拿来直接计算此次收益。历史三框架数据原样归档；原来所谓等 FP32 比较应撤回。

PyTorch 2.9.1+cu128：参数/梯度/SGD FP32，matmul TF32=False、cuDNN TF32=False、benchmark=False、deterministic=True；本进程另设 NVIDIA_TF32_OVERRIDE=0。CuPy 使用默认 FP32 cuBLAS math，未启用 fast-math。cuDNN 保持启用，与 HyTorch 显式 im2col+GEMM 实现不同；匹配的是模型/数据/精度/训练配置，不是相同底层算法。

## 新的 profiling

另取相同窗口各模式 10-step Nsight trace；这些不是 5 次重复的 kernel 均值。TRAINING_COMPARISON.csv 每一轮携带同一份独立 trace 汇总，有明确 note，不能把重复列当 5 份 profile。

| 实现 | GPU kernel 总计 μs/step | Forward | Backward | SGD | CUDA API CPU elapsed | 同步 API CPU elapsed |
|---|---|---|---|---|---|---|
| before_native | 2036.276 | 752.764 | 1157.937 | 125.575 | 722.904 | 2.875 |
| after_native | 1750.836 | 752.412 | 880.311 | 118.113 | 753.382 | 3.916 |
| after_cupy | 1419.622 | 408.827 | 891.105 | 119.690 | 697.587 | 2.509 |
| torch | 2666.985 | 545.029 | 2094.088 | 27.869 | 1899.703 | 1332.501 |

GPU 列为 CUPTI kernel 实际 duration 的累计，CPU API/同步列为主机 elapsed，彼此重叠，不可相加。CPU phase elapsed 与 API 分项见 CSV/JSON；没有开启 CPU 栈采样，不能定量声称剩余 CPU 时间全是 Python 调度。

旧 Native 有 160 次 FP32→FP64 copy、100 个 FP64 命名 GEMM/相关 kernel（10 steps）；修复后 Native 和 CuPy 均为 0，并由实际 dtype 审计交叉验证。Native kernel 数从 161/step 减至 147/step，GPU kernel 总计下降约 14.02%，backward 从 1157.937 降至 880.311 μs/step。前向保持约 752 μs。

PyTorch autograd 的 430 个 launch 出现在 worker thread；通过 correlationId 找到对应 CPU API，再关联唯一包围它的主线程 backward NVTX 区间，全部为 backward；这是明确的跨线程关联，非 GPU 时间戳落在 Python 区间的错误匹配。各 trace 无未归因 kernel。

profile 内 GPU kernel+memcpy 时间并集占窗口：before_native 36.87%、after_native 33.19%、after_cupy 28.15%、torch 70.69%。这是活动时间比例，不是 occupancy 或 nvidia-smi utilization。PyTorch 这个 trace 的 kernel 累计更高却 wall 更短，说明累加 kernel 与端到端提交/重叠是不同维度；禁止把 profiled GPU 累计与未 profile 的 wall 作差。PyTorch 的 pageable H2D 与默认流排队可能把等待体现在 transfer/API 中，不应误称纯数据拷贝成本。

## 反向细分与当前热点

用带逐函数 NVTX 的 native_trace 关联操作，再对只有 phase NVTX 的 native_plain 校验 kernel 序列/phase/grid/block/regs/shared：1470/1470 匹配。细分 shape 标签继承，实际 duration 来自 native_plain。细插桩 CPU wall 不能当真实训练速度。

| 修复后 Native 分类 | GPU μs/step | GPU kernel 占比 |
|---|---|---|
| conv_forward | 605.183 | 34.56% |
| conv_dInput | 341.710 | 19.51% |
| conv_dWeight | 340.299 | 19.43% |
| other_forward | 146.432 | 8.36% |
| SGD_update | 118.474 | 6.77% |
| activation_backward | 73.770 | 4.21% |
| conv_dBias | 57.207 | 3.27% |
| linear_backward | 40.038 | 2.29% |
| other_backward | 17.702 | 1.01% |
| linear_bias_backward | 10.310 | 0.59% |

以上分类互不重叠。dInput=341.710，其中 col2im=233.735、BLAS=82.086、layout copy=25.888 μs。dWeight=340.299，其中 im2col=107.469、layout copy=107.700、BLAS+辅助=125.130 μs。前向卷积=605.183，其中 native GEMM=436.271、im2col=111.280 μs。

修复后 conv4/5 register GEMM 训练实际时间 113.268 / 148.583 μs；其 block 为 4 / 2、每 block 256 线程、64 registers/thread、8256 bytes shared。原 shared 为 32 registers/thread、2048 shared bytes。低 block 数是结构事实；没有实测 occupancy/带宽计数器。

一次诊断 step 的 pool 请求从 147 次/124,776,518 bytes 降至 131 次/64,723,654 bytes，指累计请求量，不是显存峰值。修复后这次带插桩/状态恢复诊断还观测到一次 6,317,056-byte 底层 allocation，不能宣称永远零 cudaMalloc。Pooling 不在 DonkeyNet 实际调用路径。

## 真实 GEMM 形状

从实际 workload 收集 24 次 GEMM/step（8 前向、16 反向），对逻辑 M/N/K 的 contiguous FP32 预打包随机矩阵比较原 shared、当前 dispatch、cuBLAS；相同输入、预分配输出、seed=4200+shape_id，10 warmups，20 次正反交替 CUDA event。总训练 GEMM FLOPs=645,021,168，前向 register 两调用仅占 2.8804%。后向现行实现仍是 cuBLAS；微基准不意味着它实际用了 native。

| M,N,K | 实现 | CUDA event 中位 μs | CUPTI kernel 总计/调用 μs | 实际 blocks |
|---|---|---|---|---|
| 240,64,432 | original | 55.136 | 25.248 | 60 |
| 240,64,432 | current | 128.624 | 95.253 | 4 |
| 240,64,432 | cublas | 50.224 | 17.917 | 96;30 |
| 72,64,576 | original | 61.280 | 32.365 | 20 |
| 72,64,576 | current | 157.248 | 124.936 | 2 |
| 72,64,576 | cublas | 45.552 | 15.858 | 54;10 |

这里 cuBLAS 某些形状包含计算+归约多个 kernel，blocks 用分号列出。事件时间包含 Python 提交和 Nsight 开销；CUPTI 列是同一批调用的实际 kernel duration 总计。因此用于解释 block 并行度的主证据采用 CUPTI 列，事件原值全部保留，绝不把高出部分当 kernel 计算。

conv4 原 shared 60 blocks、当前 4 blocks；conv5 原 20、当前 2。当前相对原实际 kernel 分别慢 3.773×、3.860×。大方阵有充足输出 blocks，64×64 tile 可以利用数据复用；小矩形把输出块数压缩后，设备并行度不足可能抵消复用优势。这是代码结构+配对时间支持的解释，非 occupancy 测量结论。没有改任何 dispatch 或 kernel。

所有实际形状：

| M | N | K | 每步调用 | original/current/cuBLAS CUPTI μs |
|---|---|---|---|---|
| 12152 | 24 | 75 | 1 | 30.637 / 30.608 / 16.741 |
| 2632 | 36 | 600 | 1 | 79.679 / 79.586 / 42.779 |
| 440 | 48 | 900 | 1 | 49.984 / 49.874 / 24.885 |
| 240 | 64 | 432 | 1 | 25.248 / 95.253 / 17.917 |
| 72 | 64 | 576 | 1 | 32.365 / 124.936 / 15.858 |
| 4 | 100 | 1152 | 1 | 58.686 / 58.626 / 16.170 |
| 4 | 50 | 100 | 1 | 8.611 / 8.589 / 6.643 |
| 4 | 1 | 50 | 1 | 5.550 / 5.538 / 4.694 |
| 4 | 50 | 1 | 1 | 4.011 / 3.995 / 3.187 |
| 50 | 1 | 4 | 1 | 4.030 / 4.027 / 5.002 |
| 4 | 100 | 50 | 1 | 6.336 / 6.323 / 5.994 |
| 100 | 50 | 4 | 1 | 4.152 / 4.141 / 8.058 |
| 4 | 1152 | 100 | 1 | 9.314 / 9.310 / 7.786 |
| 1152 | 100 | 4 | 1 | 6.230 / 6.254 / 8.302 |
| 576 | 72 | 64 | 1 | 8.146 / 18.290 / 10.504 |
| 64 | 576 | 72 | 1 | 8.643 / 21.168 / 11.810 |
| 432 | 240 | 64 | 1 | 10.902 / 18.349 / 12.314 |
| 64 | 432 | 240 | 1 | 16.176 / 55.199 / 17.165 |
| 900 | 440 | 48 | 1 | 21.379 / 21.317 / 16.947 |
| 48 | 900 | 440 | 1 | 29.296 / 29.274 / 23.723 |
| 600 | 2632 | 36 | 1 | 67.253 / 67.218 / 25.293 |
| 36 | 600 | 2632 | 1 | 142.605 / 142.594 / 41.583 |
| 75 | 12152 | 24 | 1 | 30.394 / 30.378 / 15.831 |
| 24 | 75 | 12152 | 1 | 602.085 / 602.196 / 28.165 |

原始 tensor stride/axes/transpose 存在 CSV origins 字段；逻辑 M/N/K 不等于已经捕获的 cuBLAS transa/transb。微基准已预打包，不能用于推断端到端 packing 成本。`original` 指此前保存的 baseline .so，`current` 指现有 .so；小形状 current 仍可能使用原 shared，不能把每行都叫 register-tiled 实测。

## 限制与未完成的补充项

GPU 主窗口为北京时间 2026-11-06 11:14:28 至 11:16:19，起始 GPU1 空闲，结束无本轮进程/显存占用；GPU0 VLLM 服务未触碰。后续拟补无 profiler 的 CUDA Graph 事件计时，空闲检查发现 GPU1 已由另一 Python 任务占用约 16.9 GB，保护性检查阻止启动。所有要求的原始事件配对和 profiler 采集已完成；无 profiler 的 graph 补充数据**未获得**，脚本标为 optional_unexecuted，不将计划当实验结果。

不确定项：GPU clocks 未锁定；没有 Nsight Compute occupancy/带宽/stall；CPU 无栈采样；微基准与训练的缓存/布局/事件开销不同；严格长 K gate 仍失败；最后整数元数据 guard 没有重新跑 GPU（同质 FP32/FP64 类型等价已核对，整数 CPU 测试通过）。不能据本轮估计下一轮训练加速比，也没有跑真实数据集准确率训练。

下一轮三个候选的证据、风险与建议验证见 NEXT_OPTIMIZATION.md。本轮到此停止，不进入 CUDA 优化。
