# 环境、命令与复现

## 环境和文件

仓库 `/home/xieminhui_pubuse/hytorch`，Python `/home/xieminhui_pubuse/hytorch-env/bin/python`。main HEAD edb6fbe3bfaa9eed461addd29e6943b22e28216e，开始时有未提交FP32/GEMM修改，详见raw/before.patch和before_status.txt。

本轮旧库：`runs/a100_20261106/gemm_shape_dispatch/lib_before.so`；新库：`lib_after.so`；强制原shared库：`runs/a100_20261106/libhytorch_cuda_baseline.so`。三库不装入上传包，EXTERNAL_ARTIFACTS.json记录绝对路径/大小/SHA。旧源码 raw/kernels_before.cu、新源码 source/hytorch/backends/cuda/kernels.cu、原shared源码 source/runs/a100_20261106/kernels_baseline.cu 均包含。

构建参数见raw/build_after.json。使用nvcc12.4.99、-O3、-std=c++14、-lineinfo、shared cudart、sm_80+compute_80；未加fast-math。CuPy14.2.0、NumPy2.2.6、PyTorch2.9.1+cu128、驱动580.95.05。

## 实际执行记录

- raw/baseline_commands.json：最初合并测试动态库装载错误，失败日志保留。
- raw/baseline_retry_commands.json：先import torch的成功基线，56GPU、22host、9host dtype。
- raw/window_commands.json：应用条件、构建、31host、102GPU测试及第一次graph基准尝试。
- raw/benchmark_resume_commands.json：修正诊断cuBLAS捕获方式后的完整GEMM、7轮训练、3份Nsight采集和GPU结束状态。

这些JSON记录实际argv、起止UTC时间、返回码；时间展示换算到北京时间为约11:43–11:49。CPU分析/打包随后进行，不重新占GPU。两个短暂诊断失败没有引入生产算法修改或扩大权限；重启阶段前再次检查GPU1空闲。GPU0 VLLM原服务一直保留。

测试进程先导入PyTorch，避免旧libcudart先被native库加载导致PyTorch缺少cudaGetDriverEntryPointByVersion：

```bash
export CUDA_VISIBLE_DEVICES=1
export CUDA_PATH=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export PYTHONHASHSEED=0
export HYTORCH_REQUIRE_CUDA=1
export NVIDIA_TF32_OVERRIDE=0
PY=/home/xieminhui_pubuse/hytorch-env/bin/python

# 使用GPU前必须重新核对全部GPU任务/显存/利用率。
# 当前GPU1是否空闲不能由本报告推断；不得终止别人的进程。
"$PY" -c 'import torch,pytest,sys;sys.exit(pytest.main(sys.argv[1:]))' \
  tests/test_cuda_device.py tests/test_backward_dtype.py tests/test_gemm_dispatch.py \
  -k 'not cpu' -q
"$PY" -m pytest tests/test_cuda_host.py tests/test_conv2d_gemm_cpu.py tests/test_gemm_blocked_packed.py -q
"$PY" -m pytest tests/test_backward_dtype.py -k 'not cuda and not donkey' -q
```

HYTORCH_REQUIRE_CUDA=1使设备缺失成为失败，不把skipped算通过。HYTORCH_DISPATCH_EVIDENCE可设置为新目录以保存42组测试DOT/数值JSON；DOT来自实际捕获的kernel节点。

## 重新测量，保留现有结果

各测量脚本的--output必须给新目录，会用exist_ok=False保护旧结果。脚本从本轮固定路径加载保存的库，不重建或覆盖它们。

```bash
DIAG=runs/a100_20261106/gemm_shape_dispatch/diagnostics
"$PY" "$DIAG/gemm_bench.py" --output runs/dispatch_repeat_01_gemm
"$PY" "$DIAG/train.py" --output runs/dispatch_repeat_01_training
```

GEMM：47组shape×2tiles×4实现；固定seed9000+shape_id；输出预分配；5次普通warmup+graph预上传/回放；12轮交替；图内1/8/32次相同GEMM，CUDA event总时长除以次数。大方阵图1次；小形状批次用于降低CPU提交空隙污染。CSV明确提供batch、每次原值、中位数、SD等。H2D/D2H和验证不在事件区间。

cuBLAS组使用CuPy创建的handle及其同一已加载libcublas.so，ctypes调用标准cublasSgemm_v2；没有通过CuPy不支持capture的wrapper。默认math mode、TF32 override0，C行主序通过C^T=B^T@A^T列主序调用实现。graph DOT保存实际kernel，独立Nsight再验证直接调用的名字和配置。

训练：seed42、256samples、batch4、3epochs、192steps、SGD.001，无dropout，10warmup、7轮交替同进程，每轮恢复同一权重。相同CPU DataLoader、H2D、phase NVTX，无逐步同步/日志/loss read，末尾同步。只有rt._library选择不同，旧新库以不同路径同时装载，避免同路径dlopen缓存。

## Nsight独立采集

```bash
NSYS=/opt/nvidia/nsight-systems/2025.1.3/bin/nsys
"$NSYS" profile --sample=none --cpuctxsw=none --trace=cuda,nvtx,osrt \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --output runs/dispatch_repeat_01_gemm_trace \
  "$PY" "$DIAG/gemm_bench.py" --profile --output runs/dispatch_repeat_01_gemm_profile
```

训练分别将被测命令替换为：
```bash
"$PY" "$DIAG/train.py" --trace old --output runs/dispatch_repeat_01_train_old
"$PY" "$DIAG/train.py" --trace new --output runs/dispatch_repeat_01_train_new
```
为每条命令分别选择新的nsys输出前缀。GEMM --profile仅2轮直接调用，用于核对kernel及配置，不作12轮事件统计替代。训练profile各10step，独立于未profile七轮wall。

三份二进制trace当前位于`runs/a100_20261106/gemm_shape_dispatch_traces/`，未入上传包。导出只用CPU：
```bash
"$NSYS" export --type sqlite --output <trace>.sqlite <trace>.nsys-rep
```

diagnostics/analyze.py的目录常量针对本轮，处理新目录时先复制脚本并调整O/T及raw输入目录；不要无意识重写旧报告。分析以correlationId连接GPU kernel和CPU launch，再用同线程NVTX包围launch来归因，绝不直接把Python函数计时当GPU时间。

## 构建/源码复原

上传包是必要源码和证据快照，不是完整独立Python发行版。在原仓库完整checkout中使用。git_diff.patch包含原有未提交修改；本轮生产变化单独在raw/dispatch_only.patch。不要覆盖其他人的修改。

当前工作区已经应用新条件，**不要再次运行apply_dispatch.py/run_window.py**；其一次性assert用于防止重复应用。若需要从头重建，在独立副本先恢复raw/kernels_before.cu及前轮FP32状态，确认基线后应用dispatch_only.patch，按build_after.json的相同参数构建三个独立输出路径。run_window.py/resume_benchmarks.py是此次执行记录脚本，包含固定输出和阶段前提，不能直接用于覆盖现有目录。

## 正确性门限和局限

原严格allclose atol=rtol=1e-4全部原样报告，22行失败保留；old/new/forced_shared所有输出逐位一致不能抹去门限失败。FP64最大绝对和相对L2误差另列。

未锁频，未采集Nsight Compute occupancy/bandwidth/stall，也无CPU栈采样。graph热缓存时间、直接CUPTI时间、真实训练时间是不同口径。每个CPU/GPU指标有单独范围，不能相加。
