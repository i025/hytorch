# 从交付源码复现

不要在原实验目录重复运行覆盖证据。以下创建不存在的新目录，不需要下载数据集或权重；沿用同一虚拟环境。确认GPU1空闲且获授权后运行；不动GPU0、不锁频、不改权限。

```bash
cd /home/xieminhui_pubuse/hytorch
/home/xieminhui_pubuse/hytorch-env/bin/python \
 runs/a100_20261106/final_training_optimization/diagnostics/prepare_reproduction.py \
 /home/xieminhui_pubuse/hytorch-final-repro
cd /home/xieminhui_pubuse/hytorch-final-repro
export CUDA_VISIBLE_DEVICES=1 CUDA_PATH=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
export PATH=/usr/local/cuda/bin:$PATH
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 HYTORCH_REQUIRE_CUDA=1
export NVIDIA_TF32_OVERRIDE=0 PYTHONHASHSEED=0
nvidia-smi
/home/xieminhui_pubuse/hytorch-env/bin/python scripts/build_cuda.py --arch 80
/home/xieminhui_pubuse/hytorch-env/bin/python \
 runs/a100_20261106/final_training_optimization/diagnostics/run_gpu.py
/home/xieminhui_pubuse/hytorch-env/bin/python \
 runs/a100_20261106/final_training_optimization/diagnostics/full_wrappers.py
/home/xieminhui_pubuse/hytorch-env/bin/python \
 runs/a100_20261106/final_training_optimization/diagnostics/torch_dispatch.py
/home/xieminhui_pubuse/hytorch-env/bin/python \
 runs/a100_20261106/final_training_optimization/diagnostics/analyze.py
/home/xieminhui_pubuse/hytorch-env/bin/python \
 runs/a100_20261106/final_training_optimization/diagnostics/plots.py
```

当前source已包含最终几何范围保护，因此新复现不必再运行guard_retest.py，也不必重做初轮数据。run_gpu.py会在源目录创建traces，要求新的profile文件名；它依次执行专项、针对性sanitizer、168组数值、单步/实际xgy验证、算子、12轮四请求配置、补充网络、CuPy/PyTorch5轮参考、memory hooks、交替短profile和全量测试。以上命令会把SGD生产接口保持回退；私有SGD原型仅在明确标记的Sanitizer和operators/full_wrappers诊断中暂时调用，不能将其计时纳入正式训练推荐。

本次真实执行经过一次边界保护修正：raw/commands.json是初轮，raw/guard_commands.json是最终受影响训练/profile重测；原始较早结果封存在pre_range_guard，最终TRAINING.csv是后者，不能择优混用。文件名operators.log曾用于独立microprofile输出，正式初轮内部微基准原始计时保留OPERATORS.csv；完整Function计时使用FULL_WRAPPERS.csv。

独立专项：

```bash
/home/xieminhui_pubuse/hytorch-env/bin/python -c \
 'import torch,pytest,sys;sys.exit(pytest.main(sys.argv[1:]))' \
 tests/test_final_optimization.py -q
```

全回归使用 `tests -q -rs`。原本两项CPU DonkeyNet被显式skip，GPU对应测试实际执行。先import torch用于该机器CUDA导入顺序兼容，不修改数值断言。

S失败复现：`diagnostics/sgd_edge.py`输出当前CuPy乘法/减法subnormal位模式，`raw/special_initial.log`保存原型失败；`raw/prototype_tests.py`是未将S生产回退视为实现成功的原测试副本。验证私有原型需显式临时替换rt.sgd_multi=rt._sgd_multi_candidate，不改变生产默认。不要为了通过而设FTZ或放宽断言。

Nsight路径：`/opt/nvidia/nsight-systems/2025.1.3/bin/nsys`；CUDA12.4工具`/usr/local/cuda/bin/compute-sanitizer`。raw命令JSON保存每条argv/UTC时间/返回码。analyze.py依据CUPTI correlationId关联host launch与NVTX，按训练phase/dweight/dinput聚合；CPU elapsed、CUDA API elapsed、GPU时间分别保留，不相加。两轮trace反向排序以减轻顺序偏差，但未锁频且样本有限。

训练精度门限来自ACCEPTANCE.md与scripts/report_cuda.py（最终权重1e-4/1e-3）；训练保留GPU loss标量引用，计时结束后读出完整轨迹。默认CuPy dWeight/legacy SGD；推荐配置示例：

```python
from hytorch.backends.cuda import backend
from hytorch.optimizers import SGD
optimizer = SGD(lr=0.001, backend="legacy").setup(model)
with backend(matmul="shared", conv="im2col", pool="native",
             col2im="native", dweight="packed"):
    # normal forward / loss.backward() / optimizer.update()
    ...
```

SGD native参数当前代表“尝试，但因本轮验收拒绝而回退”；不是可用fused SGD承诺。不要将S/DS列当作两个有效优化的2×2消融。

包内包含必要源码、最终/基线动态库、测试、全部CSV和原始Nsight-rep/SQLite、图表与脚本，没有数据集和训练权重。SHA.json为源码和库校验，PACKAGE_FILES.json为交付文件校验。初步版本数据单独封存。旧严格GEMM失败文件在raw/previous_gemm_*中。
