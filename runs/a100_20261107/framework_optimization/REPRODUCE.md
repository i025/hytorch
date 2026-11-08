# 在新目录复现

不要直接重跑并覆盖本目录。GPU1必须空闲且获授权；不得使用GPU0或终止其它进程。原始真实执行命令/退出码/结束时间在raw/commands.jsonl；最初资源阻塞与最终外部任务情况在raw/preparation与raw/GPU_WINDOW.md。

## 准备与阶段A

同机虚拟环境 `/home/xieminhui_pubuse/hytorch-env/bin/python`，CUDA路径 `/usr/local/cuda`，Nsight2025.1.3。所有GPU子进程由run_tasks.py设置CUDA_VISIBLE_DEVICES=1、NVIDIA_TF32_OVERRIDE=0、OPENBLAS/OMP线程4、PYTHONHASHSEED=0、HYTORCH_REQUIRE_CUDA=1。必须先import torch，再运行pytest，以沿用本机CUDA导入顺序。

```bash
cd /home/xieminhui_pubuse/hytorch
/home/xieminhui_pubuse/hytorch-env/bin/python \
 runs/a100_20261107/framework_optimization/diagnostics/prepare_reproduction.py \
 /home/xieminhui_pubuse/hytorch-framework-repro
cd /home/xieminhui_pubuse/hytorch-framework-repro
nvidia-smi
/home/xieminhui_pubuse/hytorch-env/bin/python \
 runs/a100_20261107/framework_optimization/diagnostics/run_tasks.py \
 workload cpu baseline trace:B0:a trace:torch:a tests operators optrace15
/home/xieminhui_pubuse/hytorch-env/bin/python \
 runs/a100_20261107/framework_optimization/diagnostics/analyze_profiles.py
/home/xieminhui_pubuse/hytorch-env/bin/python \
 runs/a100_20261107/framework_optimization/diagnostics/freeze_stage_a.py
```

prepare_reproduction只接受不存在的目录，先复制原始生产基线，包括原动态库、测试与必要脚本。GEMM诊断旧库也随包提供。当前无CUDA改动，不必重编译；换机器或CUDA环境不兼容时，用scripts/build_cuda.py --arch 80按保存编译参数重建并记录SHA，不能冒充原库二进制。

## 应用唯一候选并验证

```bash
git apply --check runs/a100_20261107/framework_optimization/git_diff.patch
git apply runs/a100_20261107/framework_optimization/git_diff.patch
/home/xieminhui_pubuse/hytorch-env/bin/python \
 runs/a100_20261107/framework_optimization/diagnostics/run_tasks.py \
 tests:final framework cpu:F1 balanced supplement memory \
 trace:B0:b trace:F1:b trace:torch:b trace:F1:c trace:B0:c \
 memcheck racecheck workload:complete allocations:history
/home/xieminhui_pubuse/hytorch-env/bin/python \
 runs/a100_20261107/framework_optimization/diagnostics/analyze_profiles.py
/home/xieminhui_pubuse/hytorch-env/bin/python \
 runs/a100_20261107/framework_optimization/diagnostics/summarize.py
/home/xieminhui_pubuse/hytorch-env/bin/python \
 runs/a100_20261107/framework_optimization/diagnostics/plots.py
```

`balanced`为正式12轮、6种排列各2遍；`main`仅为本次保留的初轮固定中间位置pilot，不应作为正式命令。正式训练无NVTX push/pop或profiler（phase包装器为相同空上下文），所有配置均在结束同步之后采集loss/grad/weight。profiling单独进行；不要把profiler开销算作训练收益。

复现版run_tasks增加逐命令启动前GPU1占用检查，发现任何现有GPU1进程即停止，不终止它们；该检查不是资源预约，不能保证运行中永不出现共享。此保护是在本次末尾观察到allocator诊断重叠后添加，不能反过来宣称本次每条命令都做过该检查。

run_tasks拒绝覆盖同名日志或已有trace。失败后先保存原日志，再选择新tag或新目录；不能删掉失败记录只保留成功记录。optrace15及allocator补充强制从baseline/source加载生产代码；其PYTHONPATH不传播到后续任务。

分配诊断使用PyTorch版本私有allocator history接口，仅独立记录allocation requests，结束后立即关闭history；旧CPU profiler不支持CUDA分配事件的零值不可用。CuPy计数为MemoryHook。MEMORY_COMPARISON.csv包围整个run，包含恢复参数和结果收集阶段的分配，不是把每步精确请求数直接乘步数；两组统计范围相同。活跃峰值只覆盖CuPy池，不包含所有库私有内存。

report.py会生成本次交付说明与scope校验，但测试计数、异常记录等文字针对这次实际运行；新环境复现后应按新的真实日志审查更新，不应盲目宣称与本次相同。正式数值、图表由CSV生成。配对CI为t(df11)，不把CPU和GPU重叠时间相加。

## 默认与推荐

```python
from hytorch.backends.cuda import backend
from hytorch.optimizers import SGD
optimizer = SGD(lr=0.001, backend="legacy").setup(model)
with backend(matmul="shared", conv="im2col", pool="native",
             col2im="native", dweight="packed", metadata_cache=True):
    # 原有 forward / loss.backward() / optimizer.update()
    ...
```

只在本轮已验证的配置下推荐显式开启metadata_cache；默认False。没有F2，没有启用融合SGD，不改GEMM阈值、kernel算法、dWeight数值路径或高阶求导语义。
