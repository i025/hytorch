# 阶段 4A 诊断插桩

`profile_training.py` 通过文件路径导入原 `benchmarks/bench_cuda_training.py`，把除 `--operator-nvtx`、`--memory-stats` 之外的参数原样交给其 `main()`。要求显式传 `--no-fingerprints`。不改模型、输入、初始权重、训练公式、Kernel 或默认 dispatch。

唯一修改的原文件是 benchmark 对 `--nvtx` 的参数校验：允许 Torch CUDA 复用已存在的 `StepProfiler` 五个 phase。CPU 仍不允许 NVTX。普通未插桩 benchmark 路径不安装任何 wrapper。

## 启用方式

```bash
python docs/experiments/donkeynet-backward-profiling/scripts/profile_training.py \
  --operator-nvtx --memory-stats \
  --framework hytorch --device cuda --gpu 0 --model donkey \
  --matmul shared --conv im2col --pool cupy --tile 16 \
  --samples 32 --batch 4 --steps 8 --warmup 3 --seed 42 \
  --mode phases --nvtx --no-fingerprints --no-monitor \
  --output runs/CHOOSE_A_NEW_OUTPUT_DIRECTORY
```

这是参数示例；实际执行命令及输出路径见本实验命令记录。GPU 0 是进程内逻辑设备，须事先将获准使用的物理设备映射到 `CUDA_VISIBLE_DEVICES`。Torch 使用 `--framework torch --matmul cupy --conv cupy --pool cupy --operator-nvtx`，不传 `--memory-stats`。

## 稳定标签格式与归因

| 标签 | 范围 |
|---|---|
| `OP/bwd:Conv2d\|in=4x3x66x200;24x3x5x5;24` | Function 的 backward，shape 是其原始前向输入，用于识别不同层 |
| `OP/fwd:Deconv2d\|in=...` | 卷积输入梯度内部调用 |
| `OP/fwd:Conv2DGradW\|in=...` | 卷积权重梯度内部调用 |
| `OP/fwd:MatMul\|in=...` | 全连接前向，或 backward 中的矩阵乘；由外层 phase 区分 |
| `OP/internal:im2col_array\|in=...` / `col2im_array` | Python 数组转换函数 |
| `OP/internal:cupy.tensordot\|in=...` | CuPy 高层运算及其中布局转换、矩阵乘等全部提交 |
| `OP/internal:cupy.dot\|in=...` / `cupy.ascontiguousarray` / `cupy.rollaxis` | 可见 Python 调用，不宣称覆盖全部 Cython 内部操作 |

聚合时按第一个 `|` 分离算子和 shape。所有已经载入的 `Function` 子类中，直接定义的 backward 均加标签；选定 forward 为 Conv2d、Deconv2d、Conv2DGradW、MatMul、Linear、Transpose、Reshape、Im2col、Col2im、ReLU、Tanh。修改在 context 退出时按逆序恢复。嵌套范围都使用 `finally` 弹出。

Torch 使用其原生 `torch.autograd.profiler.emit_nvtx(record_shapes=True)`；其节点名和 seq 元数据由 Torch 提供，不能假设采用 HyTorch `OP/` 标签。外层 phase 名仍是 `data`、`transfer`、`forward`、`backward`、`update`。

NVTX 表示 **CPU 主机执行与提交区间**。GPU 执行是异步的，应以 CUDA API / kernel correlation ID 关联到发起调用的最内层范围，不能仅凭 kernel 时间是否落在 Python 范围内归因。父子范围有包含关系，父子耗时不能相加；内存分配计数也归属最内层标签，避免重复计算。

## 内存统计

可选 `--memory-stats` 使用已安装 CuPy 12.3 `MemoryHook` 的 `malloc_postprocess`、`alloc_postprocess`、`free_postprocess`。方法与参数经安装包的 `cupy/cuda/memory_hooks/debug_print.py` 和 `line_profile.py` 核对。

计数只在内存中更新，训练结束后一次写入 `instrumentation.json`，不在每次分配时进行文件 I/O。按 phase、最内层算子、类型记录次数和累计字节：

- `malloc`：向 CuPy 内存池申请，包含池复用；另保留原始请求字节与取整字节。
- `alloc`：池需要向底层申请新 backing allocation。
- `free`：归还内存池，不等同立即释放到驱动。

累计字节不是峰值或存活内存，`malloc` 与 `alloc` 也不能相加作为总需求；Torch allocator 不在该 hook 范围内。`phase=outside` 包含初始化、warmup、恢复权重、最终快照；只分析正式训练时筛选五个 phase。

## 计时口径和开销边界

HyTorch 的新增 wrapper 不插入 CUDA Event、不调用 synchronize，不增加 tensor 转换、数据复制或数值读取。shape 只读取元数据。Torch 原生 `emit_nvtx` 在整个 context 进入/退出时各调用一次 `torch.cuda.synchronize()`（已核对本机 Torch 2.4.1 的 `torch/autograd/profiler.py`），位置在正式训练循环之外，不是在每个算子之间同步。插桩会增加 Python 包装调用、字符串构造、NVTX push/pop、计数更新以及可选 MemoryHook 回调开销，因而插桩运行不能替代正式无插桩计时。

已有 `--mode phases` 在 phase 边界同步，该同步来自原 `StepProfiler`；阶段 4A 不再在微小算子间增加同步。完整插桩涵盖初始化和 warmup，分析正式训练必须筛选 phase 范围。

`--mode throughput --nvtx --operator-nvtx` 则沿用原 benchmark 的 step 边界同步，phase 及算子标签本身都不增加同步；其 phase JSON 不提供同步耗时。两种模式都必须分别保留无插桩计时与插桩采集，不能混用数值。

## CPU 验证

`test_profile_training.py` 在 `CUDA_VISIBLE_DEVICES=''` 下测试：

1. TinyNet：插桩前后 loss、全部梯度和 SGD 后全部权重逐元素相等。
2. DonkeyNet：同上，批量 2、输入 3×66×200、seed 42、Dropout 关闭。
3. 运算异常时嵌套 NVTX 全部弹出，全部原方法恢复。
4. phase 归因及内存累计计数准确，phase 退出后恢复 outside。
5. shape 标签只读取元数据，不触发数组转换。
6. 缺少 `--no-fingerprints`、CPU 算子 NVTX、未启用算子标签却启用内存统计，均在训练前拒绝。
7. 正常参数原样转交 benchmark，调用结束恢复原 `sys.argv`。
8. 使用假的 Torch/CuPy 对象验证 Torch `emit_nvtx(record_shapes=True)` context 的进入/退出和 wrapper 参数剥离；此项不加载真实 GPU 或执行真实 Torch profiler。

首次五项测试为 **5 passed in 2.11s**；补充 CLI 和 Torch context 合约检查后，第二次完整测试为 **10 passed in 0.91s / 0 failed / 0 skipped**，完整日志与 JUnit XML 位于 `../logs/instrumentation-cpu-tests-02.*`，首次结果亦保留。三组无效 CLI 参数各计一个测试。

本轮 GPU 空闲预检查未通过，**GPU 插桩验证状态为 NOT_RUN**。实际 CuPy NVTX、MemoryHook GPU 回调、真实 Torch `emit_nvtx`、GPU loss/权重等值及插桩开销均尚未实测。CPU 测试只确认 CPU 数值透明性、上下文合约和生命周期，不能代替 GPU 验证。
