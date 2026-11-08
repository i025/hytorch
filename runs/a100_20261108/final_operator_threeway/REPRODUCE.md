# 复现命令与文件口径

工作目录 `/home/xieminhui_pubuse/hytorch`。使用已有 `/home/xieminhui_pubuse/hytorch-env/bin/python`，不安装/升级组件。生产源码和三个 .so 的逐文件 SHA 在 raw/source_hashes_before.json；原始状态 raw/git_status_before.txt、raw/git_diff_before.patch，结束状态 raw/git_status_after.txt。

所有 CPU/GPU 环境：

```bash
cd /home/xieminhui_pubuse/hytorch
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export NVIDIA_TF32_OVERRIDE=0
```

脚本 common.py 在导入 NumPy 前也设置线程和 TF32 环境，并明确配置 O1/F1。raw/environment.json 中 `scipy_openblas_get_num_threads64_()` 实际返回 4。

## 建立独立目录

**不要原地运行覆盖本次产物。** 新建唯一目录，例如 `runs/a100_20261108/final_operator_threeway_recheck_YYYYMMDD_HHMMSS`，复制 diagnostics/，创建 raw/source/traces/figures/。脚本 O 从所在目录解析、R 从项目层级解析；下列路径中的 final_operator_threeway 替换为新目录。

## 静态、CPU 检查

```bash
/home/xieminhui_pubuse/hytorch-env/bin/python runs/a100_20261108/final_operator_threeway/diagnostics/snapshot.py
/home/xieminhui_pubuse/hytorch-env/bin/python runs/a100_20261108/final_operator_threeway/diagnostics/cpu_check.py
/home/xieminhui_pubuse/hytorch-env/bin/python -m pytest -q tests/test_cuda_host.py tests/test_col2im_host.py tests/test_conv2d_gemm_cpu.py tests/test_gemm_blocked_packed.py
nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
```

确认物理 GPU 1 无其他进程且有权使用后才设置 CUDA_VISIBLE_DEVICES=1。本次 UUID 是 GPU-0bc09ee6-967a-b53d-bf80-38379585cf08。不要触碰 GPU 0 VLLM，不锁频、不改驱动/权限。无空闲授权 GPU 时停止 GPU 部分。

## GPU 运行

```bash
export CUDA_VISIBLE_DEVICES=1
export HYTORCH_REQUIRE_CUDA=1
/home/xieminhui_pubuse/hytorch-env/bin/python -m pytest -q tests
/home/xieminhui_pubuse/hytorch-env/bin/python runs/a100_20261108/final_operator_threeway/diagnostics/workload.py
/home/xieminhui_pubuse/hytorch-env/bin/python runs/a100_20261108/final_operator_threeway/diagnostics/benchmark.py --smoke
/home/xieminhui_pubuse/hytorch-env/bin/python runs/a100_20261108/final_operator_threeway/diagnostics/benchmark.py
/home/xieminhui_pubuse/hytorch-env/bin/python runs/a100_20261108/final_operator_threeway/diagnostics/extra.py
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=false -o runs/a100_20261108/final_operator_threeway/traces/operators /home/xieminhui_pubuse/hytorch-env/bin/python runs/a100_20261108/final_operator_threeway/diagnostics/benchmark.py --trace
nvidia-smi
```

Nsight Systems 只作独立诊断；不使用 NCU，也不使用 profiler 下 wall time 作为正式结果。所有进程结束即释放 GPU，后续仅 CPU 处理产物。

本次操作日志包括首次简化 dWeight/Pool backward 构造器和修正版，为保留原始证据，最终将两个算子的所有模式/口径一起替换为以下补测，其他算子没有替换。归档中的 common.py 已是修正版；新复现完整跑一次即可，不需要补测或 merge。

```bash
/home/xieminhui_pubuse/hytorch-env/bin/python runs/a100_20261108/final_operator_threeway/diagnostics/benchmark.py --ops dweight,pool_backward --prefix raw/interface_corrected_
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=false -o runs/a100_20261108/final_operator_threeway/traces/interface_corrected /home/xieminhui_pubuse/hytorch-env/bin/python runs/a100_20261108/final_operator_threeway/diagnostics/benchmark.py --trace --ops dweight,pool_backward
```

## CPU 分析、图表、报告

```bash
nsys export --type sqlite --output runs/a100_20261108/final_operator_threeway/traces/operators.sqlite runs/a100_20261108/final_operator_threeway/traces/operators.nsys-rep
/home/xieminhui_pubuse/hytorch-env/bin/python runs/a100_20261108/final_operator_threeway/diagnostics/profile.py
/home/xieminhui_pubuse/hytorch-env/bin/python runs/a100_20261108/final_operator_threeway/diagnostics/analyze.py
/home/xieminhui_pubuse/hytorch-env/bin/python runs/a100_20261108/final_operator_threeway/diagnostics/metadata.py
/home/xieminhui_pubuse/hytorch-env/bin/python runs/a100_20261108/final_operator_threeway/diagnostics/figures.py
/home/xieminhui_pubuse/hytorch-env/bin/python runs/a100_20261108/final_operator_threeway/diagnostics/report.py
```

report.py 包含本次实际结论的中文叙述及动态表格；复现新时窗时须审核/更新固定叙述，不能将旧结论自动当作新结果。本次补测还执行了：

```bash
nsys export --type sqlite --output runs/a100_20261108/final_operator_threeway/traces/interface_corrected.sqlite runs/a100_20261108/final_operator_threeway/traces/interface_corrected.nsys-rep
/home/xieminhui_pubuse/hytorch-env/bin/python runs/a100_20261108/final_operator_threeway/diagnostics/profile.py --input interface_corrected --prefix raw/corrected_profile/
/home/xieminhui_pubuse/hytorch-env/bin/python runs/a100_20261108/final_operator_threeway/diagnostics/merge.py
```

merge.py 只合并本轮同 case 的完整配对组，不混算不同窗口的 CuPy/Native。首批正式数据保留 raw/formal_firstpass/。历史严格失败仅复制到 raw/historical_strict_gate/，不进入计时分析。

## 结果字段

- RAW_TIMINGS.csv：逐 repeat，scope=array/function，mode，loops、order、stream_ptr、预分配开关；sync_us/submit_us/event_us 分开。CPU 没有 event/submit 字段。
- *_RESULTS.csv：每 case/scope/mode/metric 的均值、中位数、样本SD、最小/最大、n。单位 μs。
- SPEEDUP_SUMMARY.csv：同口径 ratio_of_means、逐对比值中位数、逐对差值均值和近似95%区间。正差值/比值>1有利于Native。
- PROFILE_SUMMARY.csv：15次独立CUPTI计时的 kernel 总和、launch/API CPU elapsed、NVTX CPU elapsed。API仅统计算子NVTX范围内，测量者的外部同步不计入该算子的API；与正常sync_us不是同一口径。
- KERNEL_DETAILS.csv：实际名称、次数、grid/block/registers；DISPATCH_DECISIONS.csv仅Native GEMM array消融。不依据名字推断NCU occupancy或stall。
- WORKLOAD.csv：所有shape、几何、dtype、输入strides；真实调用路径/次数来自 raw/workload.json。GEMM独立比较统一连续合成矩阵，只复用真实M/N/K，**不声称重放了实际训练所有A/B布局**。真实权重转置packing由完整Conv测量包含。
- ALLOCATIONS.csv：内存池预热后的malloc请求数/请求字节，包含临时数组，不是driver cudaMalloc计数/峰值内存。
- ACCURACY.csv：独立FP64参考、MaxAbs、RelativeL2、混合容差失败元素数、近零分母floor明确。

冻结原Shared、旧Register、当前库在同进程用ctypes加载；rt._library仅诊断进程内切换，不写生产文件。生产代码未改，默认backend未变。所有现有测试门限不变。

## 二进制复现来源

服务器保留：

- `/home/xieminhui_pubuse/hytorch/runs/a100_20261106/libhytorch_cuda_baseline.so`
- `/home/xieminhui_pubuse/hytorch/runs/a100_20261106/gemm_shape_dispatch/lib_before.so`
- `/home/xieminhui_pubuse/hytorch/hytorch/backends/cuda/build/libhytorch_cuda.so`

原CUDA源码归档在 source/frozen/ 和 source/hytorch/backends/cuda/kernels.cu；当前构建命令在 raw/build.json。需要在其他服务器重建时输出至新的诊断目录，示例：

```bash
/usr/local/cuda/bin/nvcc -O3 -std=c++14 -lineinfo --shared --cudart shared -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_80,code=compute_80 -Xcompiler -fPIC source/frozen/shared.cu -o /tmp/hytorch_diag_shared.so
```

同步更改诊断脚本库路径；工具链不同的二进制SHA自然可能不同，应重新记录。不要覆盖现有库或历史产物。

原始 .nsys-rep/.sqlite 路径和SHA见 EXTERNAL_ARTIFACTS.json；为了归档轻便不放入tar，但解析后的逐kernel/逐调用CSV全部包含。打包脚本 diagnostics/package.py 生成文件清单及归档，不包含数据集、权重、动态库、pyc。
