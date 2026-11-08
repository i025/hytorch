# 完整复现信息

## 已执行环境

工作目录 `/home/xieminhui_pubuse/hytorch`，Python `/home/xieminhui_pubuse/hytorch-env/bin/python`，main HEAD edb6fbe3bfaa9eed461addd29e6943b22e28216e。保留前轮全部未提交修改，baseline/含必要完整源码，raw/before.patch与before_status.txt记录开始状态。

A100-PCIE-40GB物理GPU1，108SM，CUDA_VISIBLE_DEVICES=1后逻辑0。GPU0 VLLM服务未触碰。北京时间2026-11-06 16:13:50–16:14:44集中完成本轮GPU工作。前置查询GPU1空闲；结束时0MiB且无本轮进程，最后utilization查询值12%反映采样窗口，不声称其瞬时必须为0。未锁定共享时钟、修改权限或终止他人进程。

nvcc12.4.99、driver580.95.05、CuPy14.2.0、NumPy2.2.6、PyTorch2.9.1+cu128。CUDA编译命令见raw/build.json（-O3,-std=c++14,-lineinfo,sm_80/compute_80,shared cudart，无fast-math）。完整GPU执行argv/环境/返回码/UTC起止在raw/window_commands.json。

## 构建与开关

```bash
cd /home/xieminhui_pubuse/hytorch
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_PATH=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export PYTHONHASHSEED=0
export HYTORCH_REQUIRE_CUDA=1
export NVIDIA_TF32_OVERRIDE=0
PY=/home/xieminhui_pubuse/hytorch-env/bin/python
"$PY" scripts/build_cuda.py --arch 80
```

col2im backend默认cupy。启用时：
```python
from hytorch.backends.cuda import backend
with backend(matmul="shared", conv="im2col", pool="native", col2im="native"):
    loss.backward()
```
显式FP64由原CuPy路径处理；输入超出支持的index metadata或旧ABI-1库无新symbol也fallback。非法geometry、非法输出/别名明确报错，不属于合法输入fallback。

## 正确性命令

任何GPU命令前必须重新查询全部GPU的任务/显存/利用率并确认可使用；不要假定GPU1现在仍空闲。确认后设置CUDA_VISIBLE_DEVICES=1。不要为了复现终止别人的任务。

```bash
export CUDA_VISIBLE_DEVICES=1
"$PY" -m pytest tests/test_cuda_host.py tests/test_conv2d_gemm_cpu.py tests/test_gemm_blocked_packed.py tests/test_col2im_host.py -q
"$PY" -m pytest tests/test_backward_dtype.py -k 'not cuda and not donkey' -q
"$PY" -c 'import torch,pytest,sys;sys.exit(pytest.main(sys.argv[1:]))' \
  tests/test_cuda_device.py tests/test_backward_dtype.py tests/test_gemm_dispatch.py tests/test_col2im_cuda.py -k 'not cpu' -q
```
先import torch统一动态库装载顺序，避免旧libcudart先被加载时与PyTorch缺符号冲突。HYTORCH_REQUIRE_CUDA=1保证无设备直接失败，不能把skipped算通过。

Compute Sanitizer：
```bash
/usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 99 --target-processes all \
  "$PY" -c 'import torch,pytest,sys;sys.exit(pytest.main(sys.argv[1:]))' \
  tests/test_col2im_cuda.py -k 'training or negative or zero_and_nonunit or validation or fold' -q
/usr/local/cuda/bin/compute-sanitizer --tool racecheck --error-exitcode 99 --target-processes all \
  "$PY" -c 'import torch,pytest,sys;sys.exit(pytest.main(sys.argv[1:]))' \
  tests/test_col2im_cuda.py -k 'training or negative or zero_and_nonunit or validation or fold' -q
```
两种工具各67项，包含实际非连续/负stride、零/切片stride、输出边界以及FP32/FP64高阶测试。日志原样保留。

## 算子与训练测量

必须在同一空闲GPU会话集中完成，输出使用全新目录以保留旧证据：
```bash
DIAG=runs/a100_20261106/native_col2im/diagnostics
"$PY" "$DIAG/bench.py" --output runs/col2im_repeat_01_bench
"$PY" "$DIAG/train.py" --output runs/col2im_repeat_01_training
```

bench.py：36case、每case15次交替配对，seed501+case_id；10warmup，输出预分配。CUDA event包围32次相同kernel的graph回放后平均，H2D/D2H和验证在计时外。common.py通过AST取原CuPy ElementwiseKernel表达式，在诊断中传预分配out；每case与未改生产_col2im_gpu输出再做逐位核对。graph是隔离Python提交空隙的热缓存微基准，不冒充完整训练。

train.py：同进程、同库、同FP32/GEMM128，只切换col2im配置。256样本、batch4、3epochs/192steps、seed42、SGD.001、dropout0，10warmup、7轮正反交替，每次恢复相同权重。wall包括DataLoader/H2D/计算/最终同步，两组NVTX相同，无每步日志/loss read/显式同步。最终参数梯度/权重快照均在计时外。

本轮run_window.py是固定目录的执行记录脚本，不能直接重跑覆盖本轮：T.mkdir(exist_ok=False)和测量输出exist_ok=False会阻止覆盖。重复实验请复制runner并将O/T改为新路径，或按上面单独命令集中运行。

## Profiling及CPU分析

```bash
NSYS=/opt/nvidia/nsight-systems/2025.1.3/bin/nsys
"$NSYS" profile --sample=none --cpuctxsw=none --trace=cuda,nvtx,osrt \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --output runs/col2im_repeat_01_trace \
  "$PY" "$DIAG/bench.py" --profile --output runs/col2im_repeat_01_profile
```

训练两组分别用新nsys输出前缀，并替换被测命令：
```bash
"$PY" "$DIAG/train.py" --trace cupy --output runs/col2im_repeat_01_train_cupy
"$PY" "$DIAG/train.py" --trace native --output runs/col2im_repeat_01_train_native
```

算子profile每case3次直接调用，训练每模式10step；它们是独立profile，不能把训练CSV附带的同一profile汇总当成7份profile重复。原二进制在runs/a100_20261106/native_col2im_traces/。CPU导出：
```bash
"$NSYS" export --type sqlite --output <trace>.sqlite <trace>.nsys-rep
```

analyze.py固定O/T路径服务本轮，复制到新分析脚本再修改路径，不要覆盖本报告。通过correlationId关联kernel到CPU launch，再按相同线程NVTX包围launch归因。backward细分类继承已验证的FP32序列metadata，除col2im替换外全部名字/grid/block一致；时间来自本轮trace。

Python主动执行时间没有栈采样，只有NVTX CPU elapsed和CUPTI API时间。二者与GPU时间重叠，不可直接相加。没有Nsight Compute occupancy/带宽/stall；未锁频，单份trace的其他不变kernel也有漂移，详见报告限制。

## 包与源码

raw/this_task.patch是本轮增量；git_diff.patch保留完整tracked工作区变化并附本轮新测试。源码包不是完整Python发行版，应在完整仓库checkout中使用。不要把已有GEMM/FP32改动当成本轮新增，也不要覆盖用户未提交工作。

baseline/保存35个必要旧源码文件；source/保存37个当前文件。lib_baseline.so、lib_native_col2im.so和当前build库在服务器保留；包内不放动态库、dataset、weights或二进制trace。EXTERNAL_ARTIFACTS.json记录绝对路径/大小/SHA，PACKAGE_FILES.json可逐项验证上传包。

本轮默认仍为cupy；不提交、推送、创建PR或进入下一套优化。
