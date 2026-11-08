# 复现命令

仅在已确认空闲且获授权的GPU执行。不要覆盖当前已测量目录；以下创建独立workspace，无数据集、权重依赖。真实命令、UTC起止与返回码见 raw/window_commands.json；初始诊断命令还记录在 diagnostics/diagnose.py、diagnose_cache.py、pilot.py，原执行版本保留 raw/executed_*.py。GPU基准主脚本run_window.py的Graph stream问题已在bench.py修正；原失败与resume记录未删除。

```bash
cd /home/xieminhui_pubuse/hytorch
# 替换成不存在的新路径，脚本会拒绝覆盖。
/home/xieminhui_pubuse/hytorch-env/bin/python \
  runs/a100_20261106/col2im_binding_optimization/diagnostics/prepare_reproduction.py \
  /home/xieminhui_pubuse/hytorch-binding-reproduction
cd /home/xieminhui_pubuse/hytorch-binding-reproduction
export CUDA_VISIBLE_DEVICES=1 CUDA_PATH=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
export PATH=/usr/local/cuda/bin:$PATH
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONHASHSEED=0
export HYTORCH_REQUIRE_CUDA=1 NVIDIA_TF32_OVERRIDE=0
nvidia-smi
/home/xieminhui_pubuse/hytorch-env/bin/python scripts/build_cuda.py --arch 80
# 所有比较使用同一当前库，old模式仅将Python wrapper换成baseline版本。
# 可选重建保存旧库（本轮原旧库还留在服务器，SHA见清单）：
nvcc -O3 -std=c++14 -lineinfo --shared --cudart shared \
  -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_80,code=compute_80 \
  -Xcompiler -fPIC \
  runs/a100_20261106/col2im_binding_optimization/baseline/hytorch/backends/cuda/kernels.cu \
  -o runs/a100_20261106/col2im_binding_optimization/baseline/libhytorch_cuda.so
/home/xieminhui_pubuse/hytorch-env/bin/python runs/a100_20261106/col2im_binding_optimization/diagnostics/diagnose.py
/home/xieminhui_pubuse/hytorch-env/bin/python runs/a100_20261106/col2im_binding_optimization/diagnostics/diagnose_cache.py
/home/xieminhui_pubuse/hytorch-env/bin/python runs/a100_20261106/col2im_binding_optimization/diagnostics/pilot.py
/home/xieminhui_pubuse/hytorch-env/bin/python runs/a100_20261106/col2im_binding_optimization/diagnostics/run_window.py
/home/xieminhui_pubuse/hytorch-env/bin/python runs/a100_20261106/col2im_binding_optimization/diagnostics/analyze.py
```

run_window一次执行：204项GPU回归、24+9项host、两种Sanitizer、独立CPU profiling/低开销计时、36case×15次三方benchmark、10轮三方训练、单算子及三组10步Nsight trace与SQLite export。默认要求物理GPU1读到0MiB/0%；若检查不通过应停止并重新确认，不允许跳过对其他用户任务的保护。脚本使用 `/opt/nvidia/nsight-systems/2025.1.3/bin/nsys`、`/usr/local/cuda/bin/compute-sanitizer`。CPU profiler不在正式训练进程中运行。

示例独立验证（在fresh workspace按需执行，不应覆盖已有raw输出）：

```bash
/home/xieminhui_pubuse/hytorch-env/bin/python -c \
 'import torch,pytest,sys;sys.exit(pytest.main(sys.argv[1:]))' \
 tests/test_cuda_device.py tests/test_backward_dtype.py tests/test_gemm_dispatch.py \
 tests/test_col2im_cuda.py tests/test_col2im_binding.py -k 'not cpu' -q
```

先import torch用于避开该服务器已知的CUDA runtime导入顺序问题，不影响HyTorch数值或门限。旧wrapper与新wrapper使用bindings.py在同进程加载，未提供生产unchecked接口。CPU cProfile self统计含C扩展时间，不可当作纯Python主动执行；正式训练最终验收只看raw/training/training.csv的无profiler记录。

源码快照source包含重建必要依赖，baseline包含阶段开始时源码。原库、候选库、Nsight-rep/SQLite文件仍在服务器，路径及哈希见EXTERNAL_ARTIFACTS.json；包中有其导出的CSV和必要原始计时数据，不含这些二进制。参考reference_kernel_order.csv仅用来对齐层/类别，当前时间来自本轮CUPTI，绝不复用旧计时。

磁盘空间有限，复现会产生编译库与profiler文件，请使用有足够空间的新目录；本轮未删除其他实验、数据或用户文件。
