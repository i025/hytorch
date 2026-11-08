# 复现命令与停止条件

本轮未实施生产复用。独立实验先检查数值准入，不能把脚本退出0或Sanitizer无错误当作数值准入成功：失败数量保存在CSV字段 `donkey_B_A_failed` 等。

在新的目录复现，避免覆盖原始记录；`source/`有当前基线代码与测试依赖，`diagnostics/`有脚本，使用相同服务器虚拟环境。以下路径需未存在：

```bash
cd /home/xieminhui_pubuse/hytorch
mkdir /home/xieminhui_pubuse/hytorch-dweight-gate-repro
cp -a runs/a100_20261106/dweight_im2col_reuse/source/. /home/xieminhui_pubuse/hytorch-dweight-gate-repro/
mkdir -p /home/xieminhui_pubuse/hytorch-dweight-gate-repro/runs/a100_20261106/dweight_im2col_reuse/raw
cp -a runs/a100_20261106/dweight_im2col_reuse/diagnostics /home/xieminhui_pubuse/hytorch-dweight-gate-repro/runs/a100_20261106/dweight_im2col_reuse/
cd /home/xieminhui_pubuse/hytorch-dweight-gate-repro
export CUDA_VISIBLE_DEVICES=1 CUDA_PATH=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
export PATH=/usr/local/cuda/bin:$PATH
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 NVIDIA_TF32_OVERRIDE=0
export HYTORCH_REQUIRE_CUDA=1 PYTHONHASHSEED=0
nvidia-smi
# 确认GPU1空闲且获授权，再构建同一已保存基线。
/home/xieminhui_pubuse/hytorch-env/bin/python scripts/build_cuda.py --arch 80
/home/xieminhui_pubuse/hytorch-env/bin/python runs/a100_20261106/dweight_im2col_reuse/diagnostics/math_gate.py
# 首次实际执行使用旧文件名failure_examples.json；当前脚本tag命名为math_gate_failure_examples.json。
cp runs/a100_20261106/dweight_im2col_reuse/raw/math_gate_failure_examples.json runs/a100_20261106/dweight_im2col_reuse/raw/failure_examples.json
/home/xieminhui_pubuse/hytorch-env/bin/python runs/a100_20261106/dweight_im2col_reuse/diagnostics/check_reference.py
/home/xieminhui_pubuse/hytorch-env/bin/python runs/a100_20261106/dweight_im2col_reuse/diagnostics/validate_baseline.py
```

validate_baseline.py记录env、实际argv、UTC时间、返回码，执行全部原主机/GPU测试及memcheck/racecheck。`math_gate.py memcheck`或`racecheck`会写各自独立CSV，保留最初数据。`raw/executed_math_gate.py`保存首次执行版本，唯一后来差异是为Sanitizer重复结果增加文件名tag，不改变运算。

CPU参考脚本使用NumPy CPU im2col、np.longdouble累加复核，不需要GPU。已有GPU回归强制HYTORCH_REQUIRE_CUDA，preload torch是该服务器的CUDA导入顺序约定。CUDA编译仍sm80、FP32、无fast-math。

本轮共有90个case（10几何×3layout×3seed），三个layout是相同逻辑值，勿把重复失败当作独立随机统计样本。示例准入门限与原测试一致，但比较对象改为旧结果；原单元测试仍全部通过。决定是否接受“新结果更贴近FP64、却与旧结果超过门限”的情况属于后续审查，本轮不自行放宽门限或继续实现。

没有训练、性能trace或峰值显存复现命令，因为这些阶段明确未执行。相关CSV中NOT_RUN及空值表示缺失测量，不是零耗时。已保存动态库仅留服务器，路径与SHA见EXTERNAL_ARTIFACTS.json。
