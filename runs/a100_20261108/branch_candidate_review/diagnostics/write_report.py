from pathlib import Path
import csv,json,hashlib,difflib,shutil,subprocess,statistics
R=Path('/home/xieminhui_pubuse/hytorch');O=R/'runs/a100_20261108/branch_candidate_review'
def put(n,s):(O/n).write_text(s.strip()+'\n')
put('REPORT.md','''# 对 experiment/gpu0-autonomous-7h 的选择性移植与 A100 验证

## 结论

有两项适合用于当前优化分支：**原生 ReLU backward 融合**和**显式不计算训练数据输入梯度**。两者组合相对修改前精确提交，在 10 对独立进程实验中均更快：192-step DonkeyNet 训练均值 **0.737014s → 0.705258s，耗时下降 4.31%，吞吐提高 4.50%**。loss、参数梯度和最终权重哈希逐位一致。

Heap 调度适合宽图，保留可选接口；DonkeyNet 单独开启 heap 的差异不显著，不默认启用。没有整体 merge 实验分支，没有替换当前 GEMM、Col2im、packed dWeight、F1/O1、SGD 或 TrainingMonitor。

## 源码与证据审查

- 目标分支：a100-optimizations-and-monitoring-20261108，基线 `4317bf55b22e362c9106fe40ec37ec3b62f2327d`。
- 来源分支：experiment/gpu0-autonomous-7h，审查提交 `d58c581aa3602e450cd736fd14cce5ceda039de3`。
- 共同祖先：`edb6fbe3bfaa9eed461addd29e6943b22e28216e`。
- 来源分支自己的最终报告明确 GPU 窗口为 **NOT_RUN**，没有完成 GPU ReLU/Sum 或训练实验。219 项 CPU 回归与宽图调度收益不能当作 A100 验收。
- 两分支独立发展，整体合并会替换/冲突现有优化与监控，因此只移植可隔离验证的机制。来源报告快照在 baseline/source_*.md，来源核心文件由 detached worktree 读取。

| 候选 | 处理与理由 |
|---|---|
| ReLU backward | 移植现有 FP32 mask-multiply kernel，验证后保留显式开关；减少布尔 mask 分配及一次 launch |
| 输入梯度需求 | 引入 requires_grad=False 的显式停止语义；Conv2d/Conv2DGradW 跳过不需要的 dInput，不剪裁参数梯度、不按层名硬编码 |
| Heap 调度 | 以 (-generation,-serial) 保持旧排序和 LIFO 平局顺序；通过高阶/非默认 stream/宽图测试，默认仍 list |
| Sum 归约 | 不移植：来源报告存在强消去误差负样本，GPU 验收缺失；本轮没有声称其 GPU 性能或通用精度已验证 |
| fused dInput/dWeight/SGD、conv auto dispatch | 不移植：当前分支已有独立验证过的路径，不能用来源未完成的新窗口替代；本轮未逐个 benchmark 这些候选 |
| CPU 预取/DataLoader | 本轮没有移植；当前验证是合成数据，不可用来判断真实 JPEG I/O 收益。需要独立真实数据协议 |

## 实现与接口

ReLU 使用现有来源的 `__fmul_rn(gy, x>0 ? 1 : 0)`，保留 NaN/Inf 和 signed-zero 的原 mask multiply 行为，不启用 fast-math/TF32。加入 C ABI 空指针/size 保护，沿用当前 runtime 的 FP32、device、size、连续化和 current stream 检查。非连续输入显式物化，费用纳入完整调用测量。FP64、CPU、空输入和 mixed-dtype 梯度保留原路径；create_graph 时保留可微分的原操作。

`Variable(data, requires_grad=False)` 表示不接收梯度、不继续遍历该输入；操作输出仍默认 True，**不是 PyTorch 风格的自动全图 requires-grad 推断**。默认 True 保留现有行为；不要在构图后随意修改该标志。只有调用者明确不需要训练数据梯度时才启用。中间激活仍反传，参数梯度和高阶权重导数通过验证。顺便支持返回 None 的梯度与 leaf.backward()；没有修改 loss、优化器、权重梯度计算方法。

CLI 新增 --relu-backward original/fused、--no-input-grad、--backward-schedule list/heap，均保持原行为为默认。已有 --final-config 的含义不变。推荐训练显式启用前两项，heap 仅按计算图实验选择。详见 REPRODUCE.md。

## 实验设计

实际执行物理 GPU 1（A100-PCIE-40GB，UUID GPU-0bc09ee6-967a-b53d-bf80-38379585cf08），CUDA_VISIBLE_DEVICES=1 / 逻辑0。开始时查询两张卡均无进程；从未使用 GPU0，结束快照中 GPU0 的 VLLM 为其他任务，未干预。GPU1 最终0MiB，实验进程已退出。

FP32、256个合成样本、batch4、3epochs/192steps、seed42、固定初始化、legacy SGD .001、dropout0；shared GEMM 128-CTA dispatch、native im2col/pool/col2im、packed dWeight、F1+O1 不变。相同数据顺序、10step warmup，恢复初始化后训练，末尾同步。正式 wall 不包含 Python进程启动/初始化/结果传回CPU，不开 profiler、监控或逐phase同步。模型输出/参数/所有192个loss的读取在计时外。

12轮同进程六模式平衡交替；TinyNet另外6轮三模式检查通用性。再以修改前完整源码 worktree+原动态库对照新组合，做10对独立进程、交替顺序验证净收益，排除新增开关基础设施对 baseline 的影响。后者不是同进程实验，所有逐对值完整保留。没有删除异常值；95% CI 使用配对 t（df11或9），共享服务器、未锁频和运行时段仍是限制。

## 同进程消融（每模式12轮）

| 模式 | 192-step wall均值/s | 相对关闭候选的耗时下降 | 配对 Δwall 95%CI/ms |
|---|---:|---:|---:|
| 候选均关闭 | 0.742186 | — | — |
| 仅ReLU融合 | 0.719195 | 3.10% | [-25.50,-20.48] |
| 仅不计算输入梯度 | 0.722578 | 2.64% | [-21.69,-17.53] |
| 仅heap | 0.740471 | 0.23% | [-3.63,0.20] |
| ReLU + 输入梯度跳过 | 0.703332 | 5.24% | [-40.97,-36.74] |
| 三项全部 | 0.701473 | 5.49% | [-43.04,-38.39] |

组合改善不能简单相加。三项全部对 baseline 的显著收益不能用于证明 heap 的独立收益，因此推荐组合不包含 heap。TinyNet baseline0.284582s，ReLU+输入梯度跳过0.257519s，耗时下降9.51%，配对差CI[-28.21,-25.92]ms；加入heap为0.258334s，略慢于前两项组合。

## 相对原始精确提交的净收益

| 对照 | 10次均值/s | samples/s |
|---|---:|---:|
| 原始4317bf5源码/动态库 | 0.737014 | 1042.04 |
| 新ReLU+输入梯度跳过 | 0.705258 | 1088.96 |

10/10对更快，平均差 **-31.756ms/run**，95%CI **[-42.389,-21.123]ms**。耗时下降4.31%、速度比1.0450×。该结果比同进程消融更保守，作为本轮最终净收益结论。两组每对的参数、梯度及全部loss哈希均相同。训练最终loss为0.00025566318072378635；仅为合成回归，不代表自动驾驶精度。

## 单算子与 CPU heap

ReLU 完整调用覆盖标量、小数组、DonkeyNet七个激活shape、大数组、转置/负stride。每组固定输入、预热20次、20轮交替；每轮30次调用，记录CUDA Event区间、CPU提交时间、末尾同步wall。两组都包括输出分配；Native额外布局物化不隐藏。Event区间包含主机提交间隙，不能称纯kernel时间。七个真实形状的Event中位数改善约1.49–1.53×；原始明细及不利case全部见 RELU_BENCHMARK.csv、RELU_SUMMARY.csv。

CPU heap独立运行在GPU试验结束后：平衡二叉聚合图、每宽度20对，只测backward（构图不计时）。宽度1约1.00×、8约1.01×、64约1.15×、256约1.63×；256宽度的中位数35.59ms→21.77ms，梯度逐位相等。不是CNN训练加速，也不表示所有大节点数图都受益。

## Nsight：收益来源

独立20-step trace，正式性能不取profiler wall；主机同线程NVTX包含关系+CUDA correlationId归因。

| 阶段 | 原kernel数/step | 新kernel数/step | 原纯kernel μs/step | 新纯kernel μs/step |
|---|---:|---:|---:|---:|
| Forward | 34 | 34 | 545.18 | 544.23 |
| Loss | 4 | 4 | 13.11 | 12.98 |
| Backward | 72 | 62 | 658.93 | 581.09 |
| Optimizer | 32 | 32 | 114.87 | 114.74 |
| 合计 | 142 | 132 | 1332.09 | 1253.04 |

原每step七个mask kernel消失，新增七个relu_backward_kernel，且少了七个独立乘法launch；col2im从5次到4次，另少第一层dInput的矩阵/布局操作，合计减少10个kernel。Forward、dWeight的算法与SGD未变。Backward NVTX主机区间2584.85→2416.11μs，launch API312.61→291.69μs；这些是独立profiler诊断，不能与纯kernel时间相加解释正式wall。没有采集NCU，不声称提高occupancy。

pool_reserved_end_bytes只表示每次训练结束时CuPy池保留量，不是峰值。本轮未测全GPU显存峰值，不据此宣称显存峰值降低。

## 正确性与负结果

完整回归402 passed、2个原有CPU skip；新增候选GPU专门检查49 passed（随后新增6个GPU heap case纳入完整回归）；CPU定向重跑32 passed。共2538条loss/梯度/权重对比全部逐位相等。独立原版对照10对的三类哈希全部一致。Compute Sanitizer memcheck26 passed/0 errors；racecheck26 passed/0 errors/0 warnings。实际优化CLI完整训练通过。

原始失败保留：移植测试中SGD枚举名称需从来源接口适配为当前native；新测试曾错误要求既有pow高阶路径FP32，将其改为与原路径dtype/数值直接比较，没有修改生产pow或放宽数值门限。系统/usr/bin旧Sanitizer缺少注入库，改用已安装/usr/local/cuda/bin工具后通过，未安装软件或改权限。详见CORRECTNESS.md。

本轮只证明两个组合在上述合成DonkeyNet/TinyNet上的收益；没有跑真实图片完整训练，没有声称来源整个分支都更快。保留原有路径和显式开关，不合并尚缺GPU证据的其他原型。
''')
put('SOURCE_CHANGES.md','''# 源码变更与来源

来源：experiment/gpu0-autonomous-7h@d58c581；目标基线4317bf5。采用选择性移植，未进行整分支合并。

| 文件 | 修改 |
|---|---|
| hytorch/backends/cuda/relu_backward.cu | 来自来源同名kernel，额外加入非法指针/size检查；FP32乘法/current stream一致 |
| hytorch/backends/cuda/kernels.cu | 只在末尾include新kernel，既有内容逐字保留 |
| hytorch/backends/cuda/runtime.py | 新增可选relu_backward模式、ctypes ABI与wrapper；复用现有安全检查，默认original |
| hytorch/functions.py | ReLU.backward在显式fused+FP32且非create_graph时调用新kernel；其他路径保留 |
| hytorch/core.py | 移植显式requires_grad、None梯度/leaf处理、可选heap；增加bool参数校验；保留O1 |
| hytorch/functions_conv.py | 只有显式不需要输入梯度时跳过Conv2d与Conv2DGradW的dInput；保留packed dWeight |
| benchmarks/bench_cuda_training.py | 三个CLI开关、predict包装及配置记录；默认不变，monitor/helper共用 |
| tests/test_backward_schedule.py | 来源的调度顺序/高阶/异常测试 |
| tests/test_conv_dinput_pruning.py | 来源的输入梯度/参数/高阶/Torch对齐测试；SGD接口适配本分支 |
| tests/test_relu_fusion.py | 集成测试：dtype/fallback、layout、stream、特殊值、边界、GPU heap高阶 |

新增参数为keyword-only，默认requires_grad=True；该标志是显式梯度接收/遍历边界，不实现全图自动推断。默认调度list、默认ReLU original、CLI默认计算输入梯度。原有--final-config不偷偷改变含义。

原有kernel内容SHA验证见raw/source_integrity.json。git_diff.patch仅本轮生产/测试代码相对4317bf5的差异。baseline/包含原源码快照；编译库保留服务器，归档和Git上传不含.so。source/保存修改后源码。
''')
put('CORRECTNESS.md','''# 正确性证据

| 检查 | 结果 | 原始日志 |
|---|---|---|
| 首次CPU定向 | 30pass/2fail | raw/host_candidates.log |
| CPU修正后定向 | 32pass、17 deselected | raw/host_candidates_retry.log |
| GPU候选专项（含CPU调度测试） | 49pass、无skip | raw/gpu_candidates.log |
| 完整回归，HYTORCH_REQUIRE_CUDA=1 | 402pass、2已有CPUskip | raw/full_tests.log |
| Sanitizer memcheck | 26pass、0 errors | raw/memcheck.log |
| Sanitizer racecheck | 26pass、0 errors、0 warnings | raw/racecheck.log |
| 优化CLI完整DonkeyNet | 3epochs/192steps完成 | raw/cli_optimized.log |
| 同进程训练loss/参数/梯度 | 2538条逐位相同 | NUMERICAL_ALIGNMENT.csv |
| 原版精确源码对照 | 10对三类SHA全部一致 | ORIGINAL_COMPARISON.csv |

首次失败为测试适配错误：来源SGD参数与当前backend枚举不同；以及测试对既有pow高阶标量路径作了错误FP32假设。后者改为检查原路径和融合路径的相同dtype/数值；没有修改生产pow，也没有降低allclose门限。失败日志保留。

移植后的参数/输入/偏置梯度与PyTorch比较沿用来源atol1e-5/rtol1e-4或更严门限，其他已有测试不改。ReLU有限数逐位一致、NaN一致、signed zero单独检查；FP64/CPU/empty/mixed dtype回退；transpose/负stride/广播、非默认stream、重复调用、None梯度、leaf、双层conv、权重高阶导数、非默认stream宽GPU图均覆盖。

初始/usr/bin/compute-sanitizer启动失败为缺失libsanitizer-collection.so，未执行kernel检查；保存在raw/memcheck_initial_environment_failure.log。改用现有完整CUDA Toolkit工具后真实执行通过；不将初次失败当通过或静默删除。

GPU专项后又加入6个宽图GPU heap案例，最终全量回归与两种Sanitizer均包含它们。完整回归中的pruning参数测试默认CPU；GPU专项显式HYTORCH_PRUNING_DEVICE=cuda，不用CPU通过替代GPU通过。两项既有CPUskip不计为GPU测试通过。
''')
put('REPRODUCE.md','''# 复现

先确认GPU空闲且有权限。本次使用物理GPU1，结束时0MiB；禁止影响GPU0上的服务。以下GPU命令只在再次确认GPU1可用后运行。

```bash
cd /home/xieminhui_pubuse/hytorch
nvidia-smi
export CUDA_VISIBLE_DEVICES=1 NVIDIA_TF32_OVERRIDE=0
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=.
export PATH=/usr/local/cuda/bin:$PATH
PY=/home/xieminhui_pubuse/hytorch-env/bin/python
$PY scripts/build_cuda.py --arch 80
HYTORCH_REQUIRE_CUDA=1 $PY -m pytest -q tests
HYTORCH_REQUIRE_CUDA=1 HYTORCH_PRUNING_DEVICE=cuda $PY -m pytest -q tests/test_relu_fusion.py tests/test_conv_dinput_pruning.py tests/test_backward_schedule.py
/usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 99 $PY -m pytest -q tests/test_relu_fusion.py
/usr/local/cuda/bin/compute-sanitizer --tool racecheck --error-exitcode 99 $PY -m pytest -q tests/test_relu_fusion.py
```

## 推荐开启方式

```bash
$PY benchmarks/bench_cuda_training.py --device cuda --gpu 0 --model donkey \
  --samples 256 --batch 4 --epochs 3 --final-config \
  --relu-backward fused --no-input-grad \
  --monitor --monitor-components disabled --output runs/candidate_training_fresh
```

--monitor --monitor-components disabled使用统一异步训练helper，但不启用监控；不传--monitor则使用旧StepProfiler入口，其默认phase同步会改变性能口径。需要资源监控时选resources/all并单独报告监控开销。heap可通过--backward-schedule heap显式试验，DonkeyNet不推荐默认加上。输入梯度用于对抗训练/解释等时，不要指定--no-input-grad。

Python API：`Variable(x, requires_grad=False)`；CUDA配置上下文新增`backend(..., relu_backward="fused")`；调度`using_config("backward_schedule", "heap")`。上下文恢复原配置，不缓存设备指针。

## 完整配对实验

不要覆盖现有记录。将diagnostics复制到同深度新目录（例如runs/a100_20261108/branch_candidate_review_rerun/diagnostics），创建raw子目录后运行：

```bash
$PY NEW/diagnostics/experiment.py > NEW/raw/training.log 2>&1
$PY NEW/diagnostics/micro.py > NEW/raw/micro.log 2>&1
```

脚本从自身路径推导仓库和输出；NEW需替换成上述同深度目录。experiment默认12轮DonkeyNet六模式、6轮TinyNet三模式；固定数据/seed/init，每组10step预热，不修改门限。结果完整读回CPU在正式计时外。

精确原版对照需原始4317bf5工作区及其CUDA库。本次保留在/home/xieminhui_pubuse/hytorch-before-candidate-review。新机器可`git worktree add --detach /path/to/baseline 4317bf5`，用相同CUDA编译器/flags在该目录构建（原版build脚本--arch80）。修改paired_processes.py中的roots为实际两个工作区，再运行该脚本。其子进程设置HYTORCH_REVIEW_ROOT，保持同一数据与初始化；该环境变量也可用于独立experiment.py --worker baseline。baseline .so的原SHA见baseline/hashes.json。

Nsight完整命令见diagnostics/gpu_validation.sh；原版和候选各20steps，cudaProfilerApi截取，NVTX+CUDA correlation归因。旧系统/usr/bin sanitizer不可用，本次使用/usr/local/cuda/bin；Nsight使用/opt/nvidia/nsight-systems/2025.1.3/bin/nsys。

GPU实验结束后再独立运行heap_cpu.py，避免CPU负载干扰正式训练；它不使用GPU。已有CSV与SQLite齐全后运行analyze.py、figures.py。CSV单位在列名中：wall/delta/CI为秒，CPU/event/micro为μs，HEAP_CPU为ms。micro是完整分配+调用的Event区间，不是纯kernel时间；PROFILE来自独立诊断不能替代正式wall。

CUDA运行环境与原构建参数沿用目标分支；PyTorch在CuPy首次设备调用之前导入，避免本机混合CUDA runtime加载顺序问题。二进制trace/SQLite、权重和.so只在服务器，不随Git发布；EXTERNAL_ARTIFACTS.json列出路径/SHA。
''')
# Capture exactly the modified code, preserving current and baseline provenance.
paths=['hytorch/core.py','hytorch/functions.py','hytorch/functions_conv.py','hytorch/backends/cuda/runtime.py','hytorch/backends/cuda/kernels.cu','hytorch/backends/cuda/relu_backward.cu','benchmarks/bench_cuda_training.py','tests/test_backward_schedule.py','tests/test_conv_dinput_pruning.py','tests/test_relu_fusion.py']
patch=[]
for p in paths:
 dest=O/'source'/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(R/p,dest)
 old=subprocess.run(['git','show','HEAD:'+p],cwd=R,capture_output=True)
 patch.extend(difflib.unified_diff(old.stdout.decode().splitlines(True) if old.returncode==0 else [],(R/p).read_text().splitlines(True),fromfile='a/'+p if old.returncode==0 else '/dev/null',tofile='b/'+p))
put('git_diff.patch',''.join(patch))
old=(O/'baseline/hytorch/backends/cuda/kernels.cu').read_bytes();new=(R/'hytorch/backends/cuda/kernels.cu').read_bytes();assert new.startswith(old)
info=dict(baseline='4317bf5',source='d58c581',existing_kernel_bytes_unchanged=True,new_kernel_suffix=new[len(old):].decode(),sha256={p:hashlib.sha256((R/p).read_bytes()).hexdigest() for p in paths})
put('raw/source_integrity.json',json.dumps(info,indent=2))
print('Reports written')
