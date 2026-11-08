# 环境与复现

工作目录 `/home/xieminhui_pubuse/hytorch`，Python `/home/xieminhui_pubuse/hytorch-env/bin/python`。Python 3.10.12、NumPy 2.2.6、CuPy 14.2.0、PyTorch 2.9.1+cu128；nvcc 12.4.99，驱动 580.95.05，CuPy runtime 12090 / driver API 13000。环境记录见 workload/summary.json、training/method.json、previous_packages.txt、previous_build_optimized.json。

物理 GPU1 为 A100-PCIE-40GB、PCI 0000:E1:00.0，108 SM。设置 CUDA_VISIBLE_DEVICES=1 后是逻辑0。GPU0 的 VLLM 服务未改变，未终止他人进程，未锁定共享GPU时钟，未申请性能计数器权限。

## 已执行命令

`raw/window_commands.json` 保存所有实际 argv、环境、UTC 起止时间、返回码；各阶段 stdout/stderr 在 raw/*.log。

GPU窗口北京时间 11:14:28–11:16:19；最终 query GPU1 0MiB、无进程，利用率 3% 是查询采样值。之后只做CPU分析。11:18:07 查到别人 python3 PID3133818 占用 GPU1 16874MiB，额外 graph 基准被 idle guard 阻止；见 supplementary_not_run.json。

主要环境：
```bash
export CUDA_VISIBLE_DEVICES=1
export CUDA_PATH=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export PYTHONHASHSEED=0
export HYTORCH_REQUIRE_CUDA=1
export NVIDIA_TF32_OVERRIDE=0
```

正确性：
```bash
/home/xieminhui_pubuse/hytorch-env/bin/python -m pytest tests/test_cuda_host.py tests/test_conv2d_gemm_cpu.py tests/test_gemm_blocked_packed.py -q
/home/xieminhui_pubuse/hytorch-env/bin/python -m pytest tests/test_backward_dtype.py -k 'not cuda and not donkey' -q
/home/xieminhui_pubuse/hytorch-env/bin/python -m pytest tests/test_backward_dtype.py -k cuda -q
/home/xieminhui_pubuse/hytorch-env/bin/python -m pytest tests/test_cuda_device.py -q
```
GPU tests require real CUDA; HYTORCH_REQUIRE_CUDA=1 prevents unavailable-device skips from masquerading as success. 最终 dtype 文件新增了整数 CPU test，因此重新选择 -k cuda 会有额外 deselected，不能把计数差误认为GPU测试数量变化。

## 在新结果目录运行

先查询所有 GPU 的进程、显存和利用率，再确认可使用。runner 只接受 GPU1 的 0MiB/0% 状态；不要直接假设此刻空闲。使用新目录，不覆盖本轮或历史记录。

```bash
PY=/home/xieminhui_pubuse/hytorch-env/bin/python
DIAG=runs/a100_20261106/fp32_fix_and_profile/diagnostics
"$PY" "$DIAG/run_window.py" \
  --output runs/fp32_repeat_01 \
  --traces runs/fp32_repeat_01_traces
```

runner 按顺序运行新/旧 GPU tests、before/after workload、5轮交替训练、3份细分profile、4份训练profile和真实shape GEMM profile。前后GPU状态保存。每份 nsys 使用：
```text
/opt/nvidia/nsight-systems/2025.1.3/bin/nsys profile
--sample=none --cpuctxsw=none --trace=cuda,nvtx,osrt
--capture-range=cudaProfilerApi --capture-range-end=stop
--output <prefix> <python> <diagnostic-script> <args>
```

然后仅用 CPU 导出8份 trace。对每个 native_trace/native_plain/cupy_trace/train_before_native/train_after_native/train_after_cupy/train_torch/gemm 执行：
```bash
/opt/nvidia/nsight-systems/2025.1.3/bin/nsys export --type sqlite \
  --output runs/fp32_repeat_01_traces/native_plain.sqlite \
  runs/fp32_repeat_01_traces/native_plain.nsys-rep
```
所有 SQLite 导出完后：
```bash
"$PY" "$DIAG/analyze.py" --handoff runs/fp32_repeat_01 --traces runs/fp32_repeat_01_traces
"$PY" "$DIAG/extra_analysis.py" --output runs/fp32_repeat_01 --traces runs/fp32_repeat_01_traces
```
报告生成器/打包脚本针对本轮目录固定，不要在新实验中无意识重写旧报告。新实验另行生成报告并区分版本。

## 训练和计时口径

训练：seed42、固定顺序256样本、batch4、3epochs、SGD.001、无dropout，10warmup、5轮交替、每轮恢复初始状态；同一模式最终 hash一致。阶段 NVTX没有每步同步；计时前后统一同步，loss/weights在结束后读取。

before_native 通过 legacy.py 从 raw/functions_before.py AST 载入旧 backward 方法，只在实验进程内切换。不是修改磁盘源码/切换CUDA库/不同随机初始化。

GPU细分：diagnose.py 的 detailed/plain 匹配1470个kernel；CPU函数时间不是GPU时间。PyTorch worker launch关联唯一主线程backward范围。Nsight捕获有扰动，记录的是profile窗口的kernel/API指标，与未profile五轮wall分开。

GEMM：24组逻辑M/N/K，固定随机FP32输入（不是训练激活值），row-major预打包，输出预分配；10warmup、20交替轮，CUDA event start→提交→end。因为Python和Nsight可能让GPU等待，event区间是含提交空隙的延迟；CUPTI kernel总计另列。CuPy dot的多个内部kernel按同一个NVTX调用求和。没有为benchmark强制改变current库dispatch。

可选 `gemm_graph_optional_unexecuted.py --graph-timing` 旨在graph内64次重复后平均，**未运行/未验证**，不作为此次结果。需以后GPU可用时独立验证与标注窗口。

## 源码、构建、归档

source/ 是必要源码快照，并非完整独立发行版。在原仓库main提交上参考 full patch及 source 文件恢复现有状态；只应用本轮修复则使用 raw/this_task.patch。不要覆盖其他人的未提交工作。

当前/原始动态库保留服务器，路径/大小/SHA见 EXTERNAL_ARTIFACTS.json。构建配置是 sm_80、-O3、-std=c++14、-lineinfo、shared cudart，无fast-math。原 baseline CUDA源码另存 source/runs/a100_20261106/kernels_baseline.cu，可使用同一构建参数重建；当前 kernels.cu 在 source/hytorch/backends/cuda/。GEMM脚本明确加载这两个保存的ABI相容库，不依靠猜测哪个库在PATH。

最终 result_type 元数据保护比GPU测量版多一处CPU类型推导，两个版本完整源码和diff均保留。FP32/FP64同质路径等价通过CuPy纯类型API核对，整数路径仅CPU测试。未经这项说明不能声称最终源码字节版本已重跑GPU。

二进制 .nsys-rep/.sqlite 在 `runs/a100_20261106/fp32_fix_and_profile_traces/`，包内不放二进制trace/库/数据集/权重；包含CSV导出、原始文本日志、源码和诊断脚本，SHA清单可验证。
