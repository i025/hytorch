from pathlib import Path
import json,csv,statistics,shutil,subprocess,hashlib,difflib
ROOT=Path.cwd();O=ROOT/'runs/a100_20261106/fp32_fix_and_profile';R=O/'raw';OLD=ROOT/'runs/a100_20261106/operator_handoff'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
train=list(csv.DictReader((O/'TRAINING_COMPARISON.csv').open()));gemm=list(csv.DictReader((O/'GEMM_REAL_SHAPES.csv').open()))
prof=json.loads((R/'training_profile_summary.json').read_text());detail=json.loads((R/'profile_summary.json').read_text())['native_plain']
kr=list(csv.DictReader((R/'native_plain_kernel_summary.csv').open()))
def table(headers,rows):return '| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+'\n'.join('| '+' | '.join(str(x) for x in row)+' |' for row in rows)+'\n'
modes=['before_native','after_native','after_cupy','torch'];stats={}
for mode in modes:
 v=[r for r in train if r['mode']==mode];ts=[float(r['wall_s']) for r in v]
 stats[mode]=dict(wall_mean=statistics.mean(ts),wall_sd=statistics.stdev(ts),samples_per_s=statistics.mean(float(r['samples_per_s']) for r in v),loss=float(v[0]['loss']),weight_max_abs=max(float(r['weight_max_abs']) for r in v),loss_abs=max(float(r['loss_abs_vs_torch']) for r in v))
training_table=table(['实现','wall s，均值±样本标准差','samples/s 均值','最终 loss','权重最大绝对误差 vs Torch'],[[m,f"{stats[m]['wall_mean']:.6f} ± {stats[m]['wall_sd']:.6f}",f"{stats[m]['samples_per_s']:.2f}",f"{stats[m]['loss']:.12g}",f"{stats[m]['weight_max_abs']:.3g}"] for m in modes])
phase_table=table(['实现','GPU kernel 总计 μs/step','Forward','Backward','SGD','CUDA API CPU elapsed','同步 API CPU elapsed'],[[m,f"{prof[m]['kernel_us_per_step']:.3f}",*[f"{prof[m]['phase_gpu_us'][k]:.3f}" for k in ['forward','backward','update']],f"{sum(v['us_per_step'] for v in prof[m]['cuda_api'].values()):.3f}",f"{sum(v['us_per_step'] for k,v in prof[m]['cuda_api'].items() if 'Synchronize' in k):.3f}"] for m in modes])
cat_table=table(['修复后 Native 分类','GPU μs/step','GPU kernel 占比'],[[k,f"{v['us_per_step']:.3f}",f"{v['pct']:.2f}%"] for k,v in sorted(detail['by_category'].items(),key=lambda x:-x[1]['us_per_step'])])
shape_table=table(['M,N,K','实现','CUDA event 中位 μs','CUPTI kernel 总计/调用 μs','实际 blocks'],[[','.join(r[k] for k in ['M','N','K']),r['implementation'],f"{float(r['median_us']):.3f}",f"{float(r['profile_kernel_us_per_call']):.3f}",r['block_count']] for r in gemm if (r['M'],r['N'],r['K']) in [('240','64','432'),('72','64','576')]])
all_shape=table(['M','N','K','每步调用','original/current/cuBLAS CUPTI μs'],[[r['M'],r['N'],r['K'],r['calls_per_training_step'],' / '.join(f"{float(x['profile_kernel_us_per_call']):.3f}" for x in gemm if x['shape_id']==r['shape_id'])] for r in gemm if r['implementation']=='original'])
source_paths=[]
for entry in json.loads((OLD/'SOURCE_FILES.json').read_text()):
 rel=Path(entry['path']);original=ROOT/rel.relative_to('source')
 if not original.is_file():original=OLD/rel
 target=O/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(original,target);source_paths.append(target)
for rel in ['tests/test_backward_dtype.py','runs/a100_20261106/kernels_baseline.cu']:
 target=O/'source'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,target);source_paths.append(target)
(O/'SOURCE_FILES.json').write_text(json.dumps([dict(path=str(p.relative_to(O)),bytes=p.stat().st_size,sha256=sha(p)) for p in source_paths],indent=2))
for name in ['gemm_baseline_summary.json','gemm_paired_comparison.csv','prior_training_summary.json','packages.txt','build_baseline.json','build_optimized.json']:
 shutil.copyfile(OLD/'raw'/name,R/('previous_'+name))
(R/'working_tree_full.patch').write_bytes(subprocess.check_output(['git','diff']))
(R/'git_status.txt').write_bytes(subprocess.check_output(['git','status','--short']))
patch=subprocess.check_output(['git','diff','--','hytorch/functions.py'])
r=subprocess.run(['git','diff','--no-index','--','/dev/null','tests/test_backward_dtype.py'],capture_output=True);assert r.returncode==1
(R/'this_task.patch').write_bytes(patch+r.stdout)
# Preserve exact tested GPU implementation, before the CPU-only integer coefficient guard.
tested=(ROOT/'hytorch/functions.py').read_bytes().replace(b'dtype=xp.result_type(diff.dtype, gy.dtype)',b'dtype=diff.dtype')
(R/'functions_gpu_tested.py').write_bytes(tested)
(R/'final_metadata_guard.patch').write_text(''.join(difflib.unified_diff(tested.decode().splitlines(True),(ROOT/'hytorch/functions.py').read_bytes().decode().splitlines(True),fromfile='functions_gpu_tested.py',tofile='functions.py')))
(O/'SOURCE_CHANGES.md').write_text('''# 本轮源码修改与审计

只修改生产文件 `hytorch/functions.py`，新增回归测试 `tests/test_backward_dtype.py`。未修改 CUDA kernel、runtime、全局 `as_array`、梯度累积或 SGD 算法。

## Tanh.backward（functions.py:53）

之前：
```python
y = self.outputs[0]()
gx = gy * (1 - y*y)
```
之后：
```python
y = self.outputs[0]()
xp = cuda.get_array_module(y.data)
one = xp.array(1, dtype=y.dtype)
gx = gy * (one - y*y)
```
`Variable.__rsub__` 将 Python int 变为 CuPy int64 零维数组，参与 FP32 数组运算时提升为 FP64。内部单位常量跟随 tanh 输出 dtype，保持计算图和显式 FP64 行为。

## MeanSquareError.backward（functions.py:478）

之前：
```python
diff = x0 - x1
gx0 = gy * diff * (2. / len(diff))
```
之后：
```python
diff = x0 - x1
xp = cuda.get_array_module(diff.data)
scale = xp.array(2. / len(diff), dtype=xp.result_type(diff.dtype, gy.dtype))
gx0 = gy * diff * scale
```
内部缩放常量匹配 diff 与上游梯度的联合 dtype。FP32/FP32 保持 FP32，FP64/FP64 保持 FP64；显式 FP64 上游梯度不被缩窄。整数 diff 的浮点上游梯度也能保留浮点比例，避免把 2/3 截断为 0。未给所有梯度加后置 cast。

GPU 窗口测试及性能数据采用 `dtype=diff.dtype` 版本。窗口结束后 CPU 边界审查发现整数输入可能截断缩放系数，最终使用 `result_type(diff.dtype, gy.dtype)`。已用 CuPy 的纯 dtype API 证明本次实测所有同质 FP32/FP64 路径的系数 dtype 完全相同，并重新通过 9 项 CPU 回归；没有重新申请 GPU 或冒充最终字节版本重跑 GPU。保存 `raw/functions_gpu_tested.py`、`raw/final_metadata_guard.patch`、`raw/final_scale_dtype_metadata.json`。最终整数边界路径仅经 CPU 验证。

## 回归测试

覆盖 CPU/CUDA 的 FP32、FP64 MSE 与 Tanh 一阶/二阶导数；PyTorch 数值参考；梯度累加；显式 FP64 数组提升；Conv2D/Linear dInput/dWeight/dBias；SGD 中间表达式；完整 DonkeyNet 的 native 和 CuPy 前向、反向、保留计算图 dtype、更新后参数对齐；另有整数输入 MSE CPU 边界测试。

`raw/this_task.patch` 是本轮两文件 patch；`raw/working_tree_full.patch` 含开始前已有未提交修改，不能将其都归为本轮。`raw/before.patch` 和 `before_status.txt` 保存开始状态。已有 README、kernels.cu、test_cuda_device.py、bench_cuda_revisions.py 和 docs 修改均保留。

诊断脚本与输出在 ignored runs 目录。`legacy.py` 从保存的原源码 AST 提取两个旧 backward 方法，仅在实验进程内临时替换；退出恢复，不覆盖工作区。没有 Commit、Push 或 PR。
''')
(O/'NEXT_OPTIMIZATION.md').write_text('''# 下一轮候选 CUDA 算子（按预期端到端影响排序）

下列顺序是基于已测 GPU 工作量与可行性的判断，不是已经得到的训练加速保证。CPU/GPU 异步重叠和主机提交间隙使 kernel 节省不能直接换算成 wall time。

## 1. Native 小矩形 GEMM：通用输出分块与 dispatch

位置：`hytorch/backends/cuda/kernels.cu:33`（gemm_shared）、`:53`（gemm_register）、`:93`（hy_gemm），调用入口 `hytorch/backends/cuda/runtime.py:105`。

事实：修复后训练 conv4/5 register GEMM 每步分别 113.268 / 148.583 μs，合计 261.851 μs，约占全部 GPU kernel 14.95%。实际 block 为 4 / 2，A100 有 108 SM；原 shared 则 60 / 20 blocks。相同随机 FP32 输入微基准 CUPTI：原/现分别 25.248 / 95.253 μs、32.365 / 124.936 μs，当前分别慢 3.77 / 3.86 倍。两 native 输出逐位相同。其他前向 native GEMM 也有开销，卷积 GEMM 总计 436.271 μs/step。

假设与方向：大输出 tile 的复用优势抵不过少量输出 block 的并行度限制；基于通用 M/N/K 工作量选择较小输出 tile、不同线程分块或通用 dispatch 值得设计，不应硬编码 conv4/5。没有测 occupancy、SM 活跃率、带宽，不能将结构推断冒充硬件计数器结论。此次没有改 dispatch。

风险/测试：尾块、非方阵、长 K 累积误差、非默认 stream、输入视图和输出别名。维持严格门限，保留当前 long-K 失败。需要在跨形状训练与配对端到端测试中确认收益。

## 2. dInput col2im

位置：`hytorch/functions_conv.py:95`（Deconv2d.forward）、`:442`（col2im_array）、`:507`（_col2im_gpu）。

事实：dInput 341.710 μs/step；其中 col2im 233.735 μs，占 dInput 68.40%、全部 GPU kernel 13.35%。conv2/1/3 分别 95.066 / 59.565 / 46.371 μs。FP64 已消失，该热点仍然存在。

假设与方向：现实现每个输出像素循环 kernel 窗口、做 stride 整除/边界判断并 gather；可设计通用索引计算、连续访存和工作划分优化。禁止以减少 dtype 精度换速度。未测带宽、stall，因此不断言它纯属带宽瓶颈。

风险/测试：stride/padding、非连续 col 布局、边界尺寸、batch、累积次序、FP32/FP64，以及卷积数值和梯度对齐。候选最大涉及上述 233.735 μs GPU 工作，实际训练收益待测。

## 3. dWeight im2col 与布局整理

位置：`hytorch/functions_conv.py:147`（Conv2DGradW.forward）、`:376`（im2col_array）、`:468`（_im2col_gpu）；CuPy tensordot 的外部实现也已随包提供。

事实：dWeight 340.299 μs/step；im2col 107.469 μs、FP32 布局复制 107.700 μs、BLAS 与辅助计算 125.130 μs。FP64 cast 已消失，但显式中间 col 和 tensordot 的布局复制仍在。当前 dWeight 走 CuPy/cuBLAS，不会自动使用 native GEMM。

假设与方向：以 cuBLAS 需要的布局直接生成 col、通用 workspace 复用、减少重复物化。先明确实际转置与 stride，不能仅将逻辑 M/N/K 当作 cuBLAS transa/transb。该候选主要影响约 215.169 μs 的 im2col/布局工作，是否值得与 BLAS 融合需另外测量。

风险/测试：多 stream 并发、buffer 生命周期、不同 kernel/stride/padding/batch、布局正确性、高阶梯度与 FP64 保留。此次只观察到 memory pool 请求，不能把逻辑新数组一律认定为每步 cudaMalloc。
''')
report=f'''# FP32 类型修复与 A100 性能复评

## 结论

已修复 MSE 与 Tanh 内部标量导致的非预期 FP64 梯度。实测 DonkeyNet 的全部 Conv2D/Linear 激活、dInput/dWeight/dBias、参数和 SGD 中间表达式为 FP32；显式 FP64 一阶/二阶导数测试通过。生产修改仅两个 backward 函数，未改 CUDA kernel、dispatch、全局类型规则或优化器。

同一 GPU 窗口 5 轮交替配对，Native wall 从 {stats['before_native']['wall_mean']:.6f} s 降至 {stats['after_native']['wall_mean']:.6f} s，减少 {(1-stats['after_native']['wall_mean']/stats['before_native']['wall_mean'])*100:.2f}%，均值比约 {stats['before_native']['wall_mean']/stats['after_native']['wall_mean']:.3f}×。这包含 FP64→FP32 计算/内存路径改变，不是 CUDA 算法加速。

conv4/5 的现有 register-tiled GEMM 在实际形状上明显比原 shared 慢；最终建议首先审查通用小矩形 GEMM 分块/dispatch，再研究 col2im 和 dWeight im2col/布局。

## 源码版本和根因

分支 main，HEAD `edb6fbe3bfaa9eed461addd29e6943b22e28216e`，开始即有此前未提交工作。前后完整 diff 与本轮增量分别保存。CUDA 源码/runtime/已编译库 SHA-256 保持不变。

`core.py:264 mul`、`:300 rsub` 经 `:170 as_array` 把 Python float/int 包成强类型 CuPy FP64/int64 零维数组。前向 loss 为 FP32，但旧 `MeanSquareError.backward` 的 `gy * diff * (2./len(diff))` 在 id=86 的第二次 Mul 首次制造 FP64；旧 Tanh 的 `1-y*y` 也会独立提升类型。父函数记录 id=83 的 FP64 输出是其返回结果，不是更早的算术执行。

旧梯度沿 MatMul、Conv2D backward、Add 累积传播；SGD 原地减法把 FP64 中间更新写回 FP32 参数，不能据参数 dtype 推断梯度正确。Native 仅用于前向；卷积/Linear backward 仍走 CuPy/cuBLAS。输入数据生成和 DataLoader 在 CPU，随后 H2D。

修复前后代码、适用范围和最终整数边界保护见 `SOURCE_CHANGES.md`。最后的 result_type 元数据保护保持本次 GPU 实测 FP32/FP64 系数类型不变；此收尾版本未重新占用 GPU，明确保留 GPU 已测源码供审计。

同类但未修改：全局 Python 标量与 Variable 的强数组提升、Pow.backward/简单 loss/其他激活中的常量可能有相同风险。它们不在本次 DonkeyNet 的实际触发路径，扩大类型规则需要独立设计。显式 FP64 输入并未被强制降精度。

## 正确性

| 检查 | 结果 |
|---|---|
| 原有主机测试 | 22 passed |
| 最终 dtype/整数边界 CPU 回归 | 9 passed，12 deselected，未用 skip 冒充通过 |
| 新增 GPU dtype/数值/DonkeyNet 回归 | 10 passed，10 deselected |
| 原有 GPU 单元测试 | 45 passed，0 skipped |
| workload 插桩对比 | 修复前、修复后各自 loss/全部梯度/更新后权重逐位一致 |
| 5 轮训练重复 | 各模式自身最终权重 hash 一致 |
| 与 PyTorch 对齐 | 所有模式通过 atol=2e-5 / rtol=2e-4 权重检查；最终 loss 误差均远小于 2e-5 |

GPU 回归包含 CPU PyTorch 算子参考以及完整 CUDA DonkeyNet 参考。MSE 通用算子按 batch 归一化，与 PyTorch 对照使用 sum/batch；训练输出为 (4,1)，因此与 PyTorch mean MSE 完全一致。

`DTYPE_AUDIT.csv` 覆盖每层和 SGD；`raw/*_all_tensor_dtypes.csv` 保留完整算子张量审计。修复前 94 个已记录输出为 FP64，修复后同一 workload 的 178 个浮点输出均为 FP32。所有浮点输入/输出也已检查没有 FP64。float64 MSE/Tanh/Conv/Linear 与梯度保持 float64。

真实 shape 微基准的原始严格 gate（NumPy FP32，atol=rtol=1e-4）在 (36,600,2632)、(24,75,12152) 上的 original/current 两实现均失败。未放宽门限；这些失败保留在 CSV，不能称“全部严格 GEMM 数值验证通过”。两 native 实现所有 shape 逐位一致，三实现相对 FP64 L2 最大 1.90e-6（补充指标，不替代严格 gate）。此前 2048/4096 大方阵失败记录也保留在 previous_gemm_* 文件中。

## 配对训练

seed=42；相同初始化和数据 hash；256 个合成样本，batch=4，3 epochs，共 192 steps/768 sample visits；SGD lr=.001，无 momentum/weight decay/dropout；各模式 10 次预热；5 轮正反交替，每次恢复相同初始权重。

wall 包括 DataLoader、pageable H2D、前后向、更新与最终同步，无逐步 loss 读取/日志/sync。所有模式共用轻量 phase NVTX；loss 与 snapshot 在计时后读取。

{training_table}
逐轮配对加速比 1.09619、1.10264、1.08867、1.09155、1.10165，均值±样本 SD=1.09614±0.00611。没有将旧窗口绝对时间拿来直接计算此次收益。历史三框架数据原样归档；原来所谓等 FP32 比较应撤回。

PyTorch 2.9.1+cu128：参数/梯度/SGD FP32，matmul TF32=False、cuDNN TF32=False、benchmark=False、deterministic=True；本进程另设 NVIDIA_TF32_OVERRIDE=0。CuPy 使用默认 FP32 cuBLAS math，未启用 fast-math。cuDNN 保持启用，与 HyTorch 显式 im2col+GEMM 实现不同；匹配的是模型/数据/精度/训练配置，不是相同底层算法。

## 新的 profiling

另取相同窗口各模式 10-step Nsight trace；这些不是 5 次重复的 kernel 均值。TRAINING_COMPARISON.csv 每一轮携带同一份独立 trace 汇总，有明确 note，不能把重复列当 5 份 profile。

{phase_table}
GPU 列为 CUPTI kernel 实际 duration 的累计，CPU API/同步列为主机 elapsed，彼此重叠，不可相加。CPU phase elapsed 与 API 分项见 CSV/JSON；没有开启 CPU 栈采样，不能定量声称剩余 CPU 时间全是 Python 调度。

旧 Native 有 160 次 FP32→FP64 copy、100 个 FP64 命名 GEMM/相关 kernel（10 steps）；修复后 Native 和 CuPy 均为 0，并由实际 dtype 审计交叉验证。Native kernel 数从 161/step 减至 147/step，GPU kernel 总计下降约 14.02%，backward 从 1157.937 降至 880.311 μs/step。前向保持约 752 μs。

PyTorch autograd 的 430 个 launch 出现在 worker thread；通过 correlationId 找到对应 CPU API，再关联唯一包围它的主线程 backward NVTX 区间，全部为 backward；这是明确的跨线程关联，非 GPU 时间戳落在 Python 区间的错误匹配。各 trace 无未归因 kernel。

profile 内 GPU kernel+memcpy 时间并集占窗口：before_native 36.87%、after_native 33.19%、after_cupy 28.15%、torch 70.69%。这是活动时间比例，不是 occupancy 或 nvidia-smi utilization。PyTorch 这个 trace 的 kernel 累计更高却 wall 更短，说明累加 kernel 与端到端提交/重叠是不同维度；禁止把 profiled GPU 累计与未 profile 的 wall 作差。PyTorch 的 pageable H2D 与默认流排队可能把等待体现在 transfer/API 中，不应误称纯数据拷贝成本。

## 反向细分与当前热点

用带逐函数 NVTX 的 native_trace 关联操作，再对只有 phase NVTX 的 native_plain 校验 kernel 序列/phase/grid/block/regs/shared：1470/1470 匹配。细分 shape 标签继承，实际 duration 来自 native_plain。细插桩 CPU wall 不能当真实训练速度。

{cat_table}
以上分类互不重叠。dInput=341.710，其中 col2im=233.735、BLAS=82.086、layout copy=25.888 μs。dWeight=340.299，其中 im2col=107.469、layout copy=107.700、BLAS+辅助=125.130 μs。前向卷积=605.183，其中 native GEMM=436.271、im2col=111.280 μs。

修复后 conv4/5 register GEMM 训练实际时间 113.268 / 148.583 μs；其 block 为 4 / 2、每 block 256 线程、64 registers/thread、8256 bytes shared。原 shared 为 32 registers/thread、2048 shared bytes。低 block 数是结构事实；没有实测 occupancy/带宽计数器。

一次诊断 step 的 pool 请求从 147 次/124,776,518 bytes 降至 131 次/64,723,654 bytes，指累计请求量，不是显存峰值。修复后这次带插桩/状态恢复诊断还观测到一次 6,317,056-byte 底层 allocation，不能宣称永远零 cudaMalloc。Pooling 不在 DonkeyNet 实际调用路径。

## 真实 GEMM 形状

从实际 workload 收集 24 次 GEMM/step（8 前向、16 反向），对逻辑 M/N/K 的 contiguous FP32 预打包随机矩阵比较原 shared、当前 dispatch、cuBLAS；相同输入、预分配输出、seed=4200+shape_id，10 warmups，20 次正反交替 CUDA event。总训练 GEMM FLOPs=645,021,168，前向 register 两调用仅占 2.8804%。后向现行实现仍是 cuBLAS；微基准不意味着它实际用了 native。

{shape_table}
这里 cuBLAS 某些形状包含计算+归约多个 kernel，blocks 用分号列出。事件时间包含 Python 提交和 Nsight 开销；CUPTI 列是同一批调用的实际 kernel duration 总计。因此用于解释 block 并行度的主证据采用 CUPTI 列，事件原值全部保留，绝不把高出部分当 kernel 计算。

conv4 原 shared 60 blocks、当前 4 blocks；conv5 原 20、当前 2。当前相对原实际 kernel 分别慢 3.773×、3.860×。大方阵有充足输出 blocks，64×64 tile 可以利用数据复用；小矩形把输出块数压缩后，设备并行度不足可能抵消复用优势。这是代码结构+配对时间支持的解释，非 occupancy 测量结论。没有改任何 dispatch 或 kernel。

所有实际形状：

{all_shape}
原始 tensor stride/axes/transpose 存在 CSV origins 字段；逻辑 M/N/K 不等于已经捕获的 cuBLAS transa/transb。微基准已预打包，不能用于推断端到端 packing 成本。`original` 指此前保存的 baseline .so，`current` 指现有 .so；小形状 current 仍可能使用原 shared，不能把每行都叫 register-tiled 实测。

## 限制与未完成的补充项

GPU 主窗口为北京时间 2026-11-06 11:14:28 至 11:16:19，起始 GPU1 空闲，结束无本轮进程/显存占用；GPU0 VLLM 服务未触碰。后续拟补无 profiler 的 CUDA Graph 事件计时，空闲检查发现 GPU1 已由另一 Python 任务占用约 16.9 GB，保护性检查阻止启动。所有要求的原始事件配对和 profiler 采集已完成；无 profiler 的 graph 补充数据**未获得**，脚本标为 optional_unexecuted，不将计划当实验结果。

不确定项：GPU clocks 未锁定；没有 Nsight Compute occupancy/带宽/stall；CPU 无栈采样；微基准与训练的缓存/布局/事件开销不同；严格长 K gate 仍失败；最后整数元数据 guard 没有重新跑 GPU（同质 FP32/FP64 类型等价已核对，整数 CPU 测试通过）。不能据本轮估计下一轮训练加速比，也没有跑真实数据集准确率训练。

下一轮三个候选的证据、风险与建议验证见 NEXT_OPTIMIZATION.md。本轮到此停止，不进入 CUDA 优化。
'''
(O/'REPORT.md').write_text(report)
(R/'training_statistics.json').write_text(json.dumps(stats,indent=2))
print('reports and source snapshots written',len(source_paths))
