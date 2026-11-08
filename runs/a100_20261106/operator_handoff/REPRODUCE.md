# 环境、命令、原始证据与复现

## 已执行环境

- 仓库 `/home/xieminhui_pubuse/hytorch`，Python `/home/xieminhui_pubuse/hytorch-env/bin/python`。
- 分支 main，HEAD `edb6fbe3bfaa9eed461addd29e6943b22e28216e`，带此前未提交的寄存器 GEMM 优化与文档/测试；`raw/working_tree.patch` 保存 tracked diff，`source/` 保存本轮所需当前源码（含 untracked benchmark）。
- Python 3.10.12 / NumPy 2.2.6 / CuPy 14.2.0；复用系统 PyTorch 2.9.1+cu128。完整 pip freeze 为 `raw/packages.txt`，不是要求重新安装系统全部软件。
- CUDA 编译器 12.4.99；CuPy 实际 runtimeGetVersion=12090；driverGetVersion=13000；驱动 580.95.05。不同层版本分别记录，不混作同一 CUDA 版本。
- Nsight Systems **2025.1.3.140**，路径 `/opt/nvidia/nsight-systems/2025.1.3/bin/nsys`。PATH 默认 nsys 是旧 2022.4.2，故显式使用新版。
- 环境：`CUDA_VISIBLE_DEVICES=1`、`CUDA_PATH=/usr/local/cuda`、`LD_LIBRARY_PATH=/usr/local/cuda/lib64`、`OPENBLAS_NUM_THREADS=4`、`OMP_NUM_THREADS=4`、`PYTHONHASHSEED=0`。
- 物理 GPU 1 UUID `GPU-0bc09ee6-967a-b53d-bf80-38379585cf08`，A100-PCIE-40GB，PCI `0000:E1:00.0`。固定为逻辑 GPU 0。GPU 0 原 VLLM 服务未触碰。
- 用户已经授权在当前服务器执行 GPU 实验；工具在正常主机执行环境运行，沙箱本身不暴露 GPU。没有 sudo、时钟锁定、驱动参数改变、终止他人进程或新权限申请。
- 使用前查所有 GPU：GPU 1 为 0 MiB、0% 且无进程。结束后也恢复 0 MiB、0%，GPU 0 服务仍存在。保存 `raw/gpu_before.log`、`raw/gpu_after_initial.log`。

## GPU 窗口内已执行命令

所有完整 argv、返回码、时间戳和环境选择记录在 `raw/window_commands.json`，stdout/stderr 在同名 `.log`。
依次执行，不并行占用 GPU：

1. `diagnostics/diagnose.py --mode workload --output .../raw/workload`：10 次 warmup，恢复初始参数，未插桩参考 1 step，再恢复初始参数，插桩 1 step。loss/所有参数梯度/更新后的权重逐位一致。记录完整 operator 描述和 CuPy memory hook。
2. Nsight `native_trace`：10 warmup + 10 记录 steps，native shared/im2col/native，逐函数诊断 NVTX，记录区间为 cudaProfilerStart/Stop。
3. Nsight `native_plain`：同样的 workload，只有 phase/step NVTX，无逐函数 wrapper。用于较少 CPU 扰动的 kernel timing。
4. Nsight `cupy_trace`：同样的 workload，CuPy 后端，逐函数 NVTX。
5. `diagnostics/compare_overhead.py`：native 后端，64 样本、batch 4、3 epochs、每组 48 steps、10 次 warmup；5 轮交替执行 logged/no_json/boundary_only。每次恢复相同初始化；三种模式的最终 SHA-256 完全相同。
6. 检查 GPU 状态并结束。后面的导出、分析、源码索引、文档与打包均为 CPU/文件操作。

Nsight 核心选项：

```bash
/opt/nvidia/nsight-systems/2025.1.3/bin/nsys profile \
  --sample=none --cpuctxsw=none --trace=cuda,nvtx,osrt \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --output <trace-prefix> \
  /home/xieminhui_pubuse/hytorch-env/bin/python <diagnostics/diagnose.py> \
  --mode trace --backend native --steps 10 --output <fresh-result-directory>
```

`plain` 模式同上但 `--mode plain`；CuPy 改为 `--backend cupy`。
本轮没有运行 Nsight Compute。其 launch 指标已由 Nsight Systems 获得，未测 occupancy/带宽/stall；没有猜测这些数值。

## 原始 trace 与摘要

二进制文件和 SQLite 留在服务器，不包含在上传包：

- `runs/a100_20261106/operator_handoff_traces/native_trace.nsys-rep`、`.sqlite`
- `runs/a100_20261106/operator_handoff_traces/native_plain.nsys-rep`、`.sqlite`
- `runs/a100_20261106/operator_handoff_traces/cupy_trace.nsys-rep`、`.sqlite`

SHA-256、大小、绝对路径见 `EXTERNAL_ARTIFACTS.json`。包内已包含足够进行源码/性能分析的 CSV：

- `raw/*_kernels.csv`：所有 kernel 的实际 start/end、correlation、phase、shape、grid/block/regs/shared。
- `raw/*_kernel_summary.csv`：按算子/layer/shape/kernel 汇总。
- `raw/*_operators.csv`：完整诊断 NVTX 内的 inclusive GPU 累计与 CPU elapsed，父子项重叠，不能相加。
- `raw/*_api.csv`：CUDA API 次数和 CPU elapsed。
- `raw/profile_summary.json`：phase/category 的不重叠 kernel 汇总，以及 memcpy/时间并集。
- `raw/workload/operators.json`、`memory.json`：完整 one-step 输入输出/属性/分配记录。
- `raw/workload_summary.json`、`gemm_workload.csv`：排除嵌套 wrapper 双计数的 FLOPs。
- `raw/observer_overhead/timings.csv`、`method.json`：5 次重复的 wall 与 event span。event span 不是 kernel 累计。
- `raw/prior_training_summary.json`、原始 GEMM 表、已有 memcheck/racecheck 日志：只复用，不重复大型实验。

导出命令（对三份 trace 各运行一次）：

```bash
/opt/nvidia/nsight-systems/2025.1.3/bin/nsys export --type sqlite \
  --output runs/a100_20261106/operator_handoff_traces/native_plain.sqlite \
  runs/a100_20261106/operator_handoff_traces/native_plain.nsys-rep
```

## 归因与正确性约束

`diagnose.py` 仅包装现有函数，并记录 NVTX、shape/stride/dtype；没有替换算法。memory hook 仅观察，不替换 allocator。
`analyze.py` 通过 `correlationId` 把 GPU kernel 关联到 CUDA launch，再用相同线程中包含该 API 的 NVTX 范围定位算子；不是根据 GPU kernel 时间是否恰好落在 CPU 函数区间做错误的异步归因。

较轻的 native_plain 与详细 native_trace 的 kernel 顺序、phase、名称、grid/block/regs/shared 全部匹配，1,610/1,610。native_plain 的细分标签/shape 使用这一验证后的映射；实际 duration 来自 native_plain 本身。
所有 scope 都有口径：PROFILE 的 phase/category/operator/kernel 四种汇总是重叠视图，不能相加；CPU 与 GPU 时间不能相加。

初版诊断给 `MatMul.forward` 写死了一个 CuPy 说明标签，后按**已记录的 attrs.native 和实际 native.gemm 子调用**更正该说明字段，未改变 shape/dtype/时间或执行。审计说明在 `raw/metadata_corrections.json`，诊断脚本也已修正。源码快照和生产代码未改动。

原始训练配置写 FP32，但本轮实测 backward 为 FP64。因此本轮是对现状的观测，不是修复之后的 FP32 性能基线，也不能与 PyTorch 原记录直接称为等精度加速比。

## 在新目录重复采集

先重新检查所有 GPU 的任务、显存、利用率与授权。不要假设 GPU 1 一直空闲。当前 runner 明确只允许 GPU 1 在 0 MiB/0% 时开始，若换卡需先按新资源条件修改诊断 runner 的设备选择，而不是终止已有任务。

源码包是分析快照，不是完整独立 Python distribution；在原仓库 checkout 上使用。异地复现时先获取原提交，再应用 `raw/working_tree.patch` 和 source 中相关新文件；不要覆盖别人已有的改动。

从仓库根目录运行，使用全新路径，保留本次证据：

```bash
PY=/home/xieminhui_pubuse/hytorch-env/bin/python
DIAG=runs/a100_20261106/operator_handoff/diagnostics
# 执行前人工/工具确认 GPU 空闲与授权；runner 还会再次检查。
"$PY" "$DIAG/run_window.py" \
  --handoff runs/operator_handoff_repeat_01 \
  --traces runs/operator_handoff_repeat_01_traces

# 对 native_trace、native_plain、cupy_trace 分别导出 SQLite。
# 示例：
/opt/nvidia/nsight-systems/2025.1.3/bin/nsys export --type sqlite \
  --output runs/operator_handoff_repeat_01_traces/native_plain.sqlite \
  runs/operator_handoff_repeat_01_traces/native_plain.nsys-rep

# 三份 SQLite 均导出后，仅用 CPU 做关联分析：
"$PY" "$DIAG/analyze.py" \
  --handoff runs/operator_handoff_repeat_01 \
  --traces runs/operator_handoff_repeat_01_traces
```

如果 Nsight 在别的环境不可用，`diagnose.py --mode workload --events --output <fresh-dir>` 提供 CUDA event 包围的 inclusive span；该 span 仍包含主机提交间隙和嵌套工作，不应冒充单 kernel 时间。本轮 Nsight 可用，因此未使用此回退，也没有填补未测数据。

## 本轮改动与打包

只在 `runs/a100_20261106/operator_handoff/` 创建诊断、源码快照与报告，在相邻 `operator_handoff_traces/` 保留 trace，最终创建 `hytorch_operator_handoff.tar.gz`。
源码/working tree 状态与开始时逐项核对，详见 `raw/final_integrity.json`。
打包脚本为 `diagnostics/package.py`，排除 `.pyc`/缓存、数据集、权重、动态库和二进制 trace。
本次没有 commit/push/PR。采集完毕后停止，等待 ChatGPT 的下一步方案。
