from pathlib import Path
import csv,json,hashlib,collections,statistics,subprocess,shutil
O=Path(__file__).resolve().parents[1];R=O.parents[2]
def read(n):return list(csv.DictReader((O/n).open()))
def table(head,rows):return '| '+' | '.join(head)+' |\n|'+'|'.join(['---']*len(head))+'|\n'+'\n'.join('| '+' | '.join(str(v) for v in r)+' |' for r in rows)+'\n'
def write(n,t):(O/n).write_text(t)
agg=read('OPERATOR_AGGREGATE.csv');rs=read('RESULTS_ALL.csv');acc=read('ACCURACY.csv');sp=read('SPEEDUP_SUMMARY.csv');prof=read('PROFILE_SUMMARY.csv');alloc=read('ALLOCATIONS.csv');env=json.loads((O/'raw/environment.json').read_text())
labels={'gemm':'GEMM','im2col':'Im2col','col2im':'Col2im','conv':'Conv forward','dweight':'dWeight（混合）','pool_forward':'MaxPool forward','pool_backward':'MaxPool backward（混合）'}
ops=list(labels)
def get(op,case,mode,scope='array',metric='sync_us'):return next(r for r in rs if r['op']==op and r['case']==case and r['scope']==scope and r['mode']==mode and r['metric']==metric)
def pg(op,case,mode):return next(r for r in prof if r['op']==op and r['case']==case and r['scope']=='array' and r['mode']==mode)
def summary(scope):
 return table(['算子','case 数','NumPy CPU μs','CuPy GPU μs','Native GPU μs','CPU/Native','CuPy/Native'],[[labels[op],r['case_count'],*[f'{float(r[k]):.3f}' for k in ['numpy_us','cupy_us','native_us']],f"{float(r['native_vs_numpy']):.3f}×",f"{float(r['native_vs_cupy']):.3f}×"] for op in ops for r in agg if r['op']==op and r['scope']==scope])
def accuracytable():return table(['算子','Native MaxAbs','Native RelativeL2','混合容差失败元素'],[[labels[o],f"{max(float(r['max_abs']) for r in acc if r['op']==o and r['mode']=='native'):.6g}",f"{max(float(r['relative_l2']) for r in acc if r['op']==o and r['mode']=='native'):.6g}",sum(int(r['mixed_fail_count']) for r in acc if r['op']==o and r['mode']=='native')] for o in ops])
# Verify all production and baseline library bytes remain unchanged.
old=json.loads((O/'raw/source_hashes_before.json').read_text());changed=[]
for row in old:
 if hashlib.sha256((R/row['path']).read_bytes()).hexdigest()!=row['sha256']:changed.append(row['path'])
assert not changed,changed
status=subprocess.run(['git','status','--short'],cwd=R,text=True,stdout=subprocess.PIPE,check=True).stdout
(O/'raw/git_status_after.txt').write_text(status);assert status==(O/'raw/git_status_before.txt').read_text()
with (O/'SOURCE_STATUS.md').open('a') as f:f.write(f'\n结束后逐一复核 {len(old)} 个源码/测试/动态库条目，SHA 全部不变，Git status 与执行前逐字一致。仅新增被 Git ignore 的 runs/ 评测产物。\n')
# Frozen algorithm sources, no binaries.
for name,src in [('shared.cu','runs/a100_20261106/kernels_baseline.cu'),('register.cu','runs/a100_20261106/gemm_shape_dispatch/raw/kernels_before.cu')]:
 target=O/'source/frozen'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(R/src,target)

correct=f'''# 正确性与口径说明

- 当前生产测试：`pytest -q tests`，**319 passed，2 skipped**（raw/current_tests.log）。两项 skip 为 CPU 全量 DonkeyNet 重复项，定义见 tests/test_backward_dtype.py；GPU DonkeyNet 回归确实执行，设置 HYTORCH_REQUIRE_CUDA=1，未将 GPU skip 当作通过。
- 窗口前主机专项：23 passed；12 组独立 CPU 窗口展开/折叠几何检查通过。
- 本轮诊断：{len(acc)} 组输出与独立 CPU FP64 参考比较，失败元素总数 **{sum(int(r['mixed_fail_count']) for r in acc)}**。包括全部 GEMM 到 **4096³ 的完整 FP64 乘法**，并非抽样；两种接口、冻结 Shared/Register 也逐项验证。
- benchmark 诊断使用 `abs(error) <= 5e-5 + 2e-5*abs(reference)`。这是新增诊断，不替换原测试的各自严格门限。GEMM 输入标准差 0.25、FP32，精度结论只覆盖本轮输入，不能推出所有数据分布均通过。
- RelativeL2 分母 `max(norm(reference),1e-30)`；最大逐元素相对误差分母 `max(abs(reference),1e-8)`，没有对近零直接除零。所有原始值见 ACCURACY.csv。
- Im2col 使用独立 NumPy sliding_window_view 的 FP64 排列参考；Conv/dWeight 用 FP64 NumPy BLAS；Col2im 用独立 PyTorch CPU FP64 fold，对非对称 padding 采用显式裁剪；Pool 用零 padding、kh/kw 顺序第一个最大值、独立 fold 梯度。
- Pooling 包含全负输入、零 padding 与相等最大值；四例 Native index 与独立参考完全相同（raw/POOL_INDEX_CHECKS.csv）。不把 PyTorch 默认负无穷 padding 的池化直接当作等价参考。
- 五个真实 Col2im 的 shape/dtype/byte strides 与一次新训练采集逐项 assert 相符；36 个 case 覆盖 training-strided、连续、负 strides。
- 无本轮 UNSUPPORTED case。旧 Register 库对于 M/N/K<64 会实际回退 Shared；DISPATCH_DECISIONS.csv 给实际 kernel，不把回退称为 Register。

{accuracytable()}

完整 API 差异：NumPy Conv/dWeight 为合理的 NumPy BLAS 评测适配器，避免将生产 CPU 的自研 C GEMM 误标成 NumPy；原生单独 Im2col 无公开 Function，使用保留安全检查的诊断 ABI adapter + 原 Function.__call__ 包装。其余 Function 使用生产类；dWeight 和 Pool backward 最终数据已补测原始构造器。所有 Function 测试保持 enable_backprop=True / O1=True，仅计该算子调用，不执行额外 backward。

历史严格失败没有被“修复”或隐藏：raw/historical_strict_gate/ 原样保存前轮长 K、2048/4096 GEMM 的失败证据，仅作为验收历史，**不参与本轮性能比值**。当前测试与随机输入通过不能覆盖或撤销这些历史失败。

执行中保留的诊断失败：首次无 tests/ 参数的 pytest 误收集多个历史 source/tests，120 个模块同名 collection errors，raw/full_tests.log 完整保留；指定当前 tests/ 后通过。workload.py 首次在写 WORKLOAD.csv 时出现重复关键字（真实采集和 shape/stride assert 已完成）；CPU 修复写表不影响数据。extra.py 首次缺少 threadpoolctl，改为 ctypes 查询已加载 OpenBLAS 的线程数，重跑成功；没有安装软件。首次简化构造器数据保留 raw/formal_firstpass/，最终 dWeight/Pool backward 使用 raw/interface_corrected_*。
'''
write('CORRECTNESS.md',correct)

text='''# HyTorch 最终算子三方评测（2026-11-08）

## 结论与实验边界

本轮仅评测，不修改生产源码、CUDA kernel 或优化算法，不运行真实驾驶训练、不提交代码。所有表格来自本轮重新运行；原库只用于同进程消融，并非拼接历史计时。Native 并不普遍优于 CuPy：**GEMM 的大方阵仍明显慢于 cuBLAS；Im2col、Col2im 和 MaxPool forward 的最终完整接口表现较好；Packed dWeight 的收益来自 Native Im2col 与 cuBLAS 的组合以及减少布局复制。**

共 122 个算子 case、两个接口口径；36 个 GEMM shape（一次真实 DonkeyNet 前后向中的 24 次不同 GEMM + 12 个扩展），14 个 Im2col、14 个 Conv forward、14 个 dWeight、36 个 Col2im、各 4 个 Pool forward/backward。真实 workload 使用一个固定 seed 的 batch=4 合成 step，记录前后向及 SGD；没有重新做全训练时序验收。因此本轮不能量化新增训练加速，也不能用微基准时间相加代替训练 wall time。

## 环境、开关与公平性

- 物理 GPU 1：A100-PCIE-40GB，UUID GPU-0bc09ee6-967a-b53d-bf80-38379585cf08；驱动 580.95.05。开始、接口补测前均确认为 0 MiB、无计算进程；结束回到 0 MiB。GPU 0 VLLM 未触碰，没有终止他人任务。
- NumPy 2.2.6 / OpenBLAS 0.3.29，实际线程查询为 4；CuPy 14.2.0；PyTorch 2.9.1+cu128（独立 CPU 参考/环境加载）；nvcc 12.4。环境全记录 raw/environment.json 和 static_environment.txt。
- TF32：所有进程 NVIDIA_TF32_OVERRIDE=0、PyTorch matmul/cuDNN TF32=False；无 fast-math，保留当前 .so。未锁频。当前 native matmul=shared + 128 CTA dispatch、conv=im2col、col2im=native（POD+metadata）、dweight=packed、pool=native；F1 metadata_cache=True，O1 compact_function=True，tile=16。此设置仅实验上下文，未修改全局默认。
- 同一 case 固定 seed=20261108+case 哈希、完全相同逻辑 FP32 输入；值分布 N(0,0.25²)，Pooling 另有全负/相等值例。数据在计时前驻留各自设备，不计 H2D/D2H，也不将 CPU/GPU 的存储边界混为一谈。
- 每种先预热 4 次，短算子每个样本连续调用 5 次、长算子 1 次；常规 18 次、五路 GEMM array 消融 30 次正式重复，平衡轮换并交替正反顺序。CPU 每 case/mode/scope 预声明 30 秒预算，本次没有触发降重复次数。全部原始样本保留 RAW_TIMINGS.csv。
- **sync_us**：调用前 stream synchronize 不计时，perf_counter_ns 包含 Python 提交、GPU 执行及调用后 synchronize；短算子为 5 次调用共享末尾同步的摊销成本，不能称“每次都同步的单次响应延迟”。NumPy 完整同步调用同样循环摊销。主比值只用相同 sync_us。
- **submit_us**：上述正常提交循环到同步前的 CPU elapsed，可能包括 runtime 内部阻塞，不能称纯 Python 指令时间。
- **event_us**：独立调用循环两 CUDA Event 之间的时间，可能含 CPU 提交造成的 GPU 空洞，不等于纯 kernel。
- **kernel_us**：另一轮 Nsight Systems/CUPTI、每模式 15 次 NVTX 采集，通过 correlationId 将实际 kernel 归属到 CPU launch。CUDA API CPU elapsed、NVTX CPU elapsed、kernel 累计各自列出，**不相加**；这些带 profiler 的数据不替换正式 wall time。
- GEMM array 所有实现预分配输出；Function GEMM 包含输出分配。其他算子均包括必要输出/临时分配、packing、layout copy；输入准备、误差计算在计时外。内存池已预热；ALLOCATIONS.csv 是分配请求量，不是 cudaMalloc 次数或峰值显存。
- NumPy Conv/dWeight 为向量化窗口+NumPy BLAS，生产 CPU 自研 C GEMM 没混入这个基线。单独 Native Im2col 是诊断 ABI 适配器；完整接口例外详见 CORRECTNESS.md。
- GPU 实验在同一张卡集中完成；主批次结束复核时 GPU 已释放，随后为纠正 Function 构造口径，重新检查空闲后补测 dWeight/Pool backward。三种实现始终在各自同进程配对内比较；首批/补测不交叉计算同一算子的比值。此实际流程包含一次短暂释放后补测，非人为宣称全程持有 GPU。

## 主要时间结果

以下为每个 case 的重复均值再作等权算术平均，**不是训练调用次数加权，也不是某个固定 shape 的延迟**。GEMM 平均明显受 4096³ 影响；应结合逐 shape CSV、方阵曲线和优势区域图。每行 CPU/Native 是跨硬件实现比较，不能作为自研算法比 cuBLAS 好的证据。

### 数组接口：resident-data，同步完成 μs

'''+summary('array')+'''
### Function 接口：包含正常包装/构图，同步完成 μs

'''+summary('function')+'''
完整均值、中位数、样本标准差、最小/最大、重复次数分别在各 *_RESULTS.csv；每种 scope、metric 独立成行。SPEEDUP_SUMMARY.csv 同时给逐对差值的近似 95% t 区间，使用保守系数 2.12；共享机顺序样本不保证独立同分布，区间仅作局部稳定性诊断。

## 正确性

当前测试 **319 passed / 2 CPU-specific skipped**；全部 804 组诊断对独立 FP64 参考的混合容差失败元素数为 0（包括大矩阵完整参考）。测试门限未改变。最大绝对误差和相对 L2 的 worst case 不必来自同一个 case；不能把最大绝对误差单独与 atol 比较而忽略 rtol。

'''+accuracytable()+'''
Native GEMM 最大绝对误差约 1.34e-4、最大相对 L2 约 2.09e-6；相对误差很小但不是逐位相等。全三方、旧库和各接口的细项见 ACCURACY.csv；历史严格失败副本仅作追溯，不被当前诊断容差替代。

## GEMM：自研优化有效，但不能等同于超过 cuBLAS

'''
rows=[]
for m,n,k in [(240,64,432),(72,64,576),(256,256,256),(512,512,512),(1024,1024,1024),(2048,2048,2048),(4096,4096,4096)]:
 c=('real_' if m!=n else 'ext_')+f'{m}_{n}_{k}';v={mode:float(get('gemm',c,mode)['mean']) for mode in ['numpy','cupy','shared_old','register_old','native']}
 rows.append([f'{m}×{n}×{k}',*[f'{v[mode]:.2f}' for mode in ['numpy','cupy','shared_old','register_old','native']],f"{v['shared_old']/v['native']:.2f}×",f"{v['register_old']/v['native']:.2f}×"])
text+=table(['M×N×K','NumPy','CuPy','旧 Shared','旧 Register dispatch','最终 dispatch','Shared/最终','旧 register/最终'],rows)
text+='''
上表均为同步完成 μs、同一最终 Python wrapper 装载不同冻结 ABI，30 轮同进程平衡交替。当前源码/库 SHA 见 SOURCE_STATUS.md。实际 kernel、grid/block、registers 从 Nsight Systems 读取，见 DISPATCH_DECISIONS.csv，非仅根据 if 条件猜测。

- 240×64×432、72×64×576 最终恢复 Shared，输出 64×64 Register 分块仅 4、2 个 CTA；原 Shared 对应 60、20 个 CTA。最终完整调用相对旧 register dispatch 分别约 3.10×、3.30×；相对旧 Shared 约 0.99×，没有宣称超过 Shared 算法本身。
- 1024³/2048³/4096³ 仍走 Register，256/1024/4096 个 CTA，相对旧 Shared 约 1.81×/2.26×/2.47×，保留大矩阵收益；但相对 CuPy 只有约 0.53×/0.64×/0.63×。CuPy 实际为 ampere_sgemm 等 cuBLAS kernel。
- 窄矩阵和长 K 的最差例为 24×75×12152，最终自研仅约 0.076× CuPy。这个形状来自真实 **dWeight**，但生产 Packed dWeight 实际用 cuBLAS，本 microbenchmark 是“将该形状分别送给所有 GEMM”的诊断，不能误读成当前训练在这里调用了慢 Native GEMM。
- 256³/512³ 的 register 输出 CTA 太少，最终 dispatch 回退 Shared；128 是保守通用门限，当前数据不能证明它对所有矩形最优。
- 真实形状比大型方阵更窄，最终 dispatch 和利用率不同；而整个 Conv 还包括 Im2col、权重 transpose packing 与输出 pack。大方阵更快不能直接推出卷积或训练更快。

## Im2col、Col2im、Conv 与 dWeight 的证据

所有 Im2col 输出统一为 (T=N×OH×OW,D=C×KH×KW)，T 顺序(n,oh,ow)，D 顺序(c,kh,kw)，三方布局与数值一致。CuPy 的 6D 输出必须转置再物化矩阵，已计入完整时间。独立 raw/IM2COL_LAYOUT_SAMPLES.csv 分离 6D、仅 layout、完整 CuPy、完整 Native；这些独立测量不能直接相加为整条路径。

以下是五层 array 口径的 **CUPTI 纯 kernel 累计 μs**，每模式 15 次。与前述无 profiler wall time 分开看：

'''
for op in ['im2col','col2im','conv','dweight']:
 rows=[]
 for i in range(1,6):
  c=f'conv{i}_'+('training' if op=='col2im' else 'contiguous');a,b=pg(op,c,'cupy'),pg(op,c,'native')
  rows.append([f'conv{i}',f"{float(a['kernel_us_mean']):.3f}",f"{float(b['kernel_us_mean']):.3f}",f"{float(a['kernel_us_mean'])/float(b['kernel_us_mean']):.3f}×",a['kernel_count_mean']+'→'+b['kernel_count_mean'],a['copy_named_kernels_mean']+'→'+b['copy_named_kernels_mean']])
 text+='\n### '+labels[op]+'\n\n'+table(['层','CuPy kernel μs','Native kernel μs','CuPy/Native','kernel 次数','名称含 copy 的 kernel 次数'],rows)
text+='''
Copy 的识别基于实际 kernel 名包含 copy，详细名称、grid、寄存器和次数在 KERNEL_DETAILS.csv，可人工审计，不把所有带宽型 kernel 都归为 copy。dWeight 的五层每次均少一次 copy 和一次分配请求；conv2 申请字节由 13,099,008 降至 6,782,208、请求从 4 降为 3。减少的是临时布局拷贝，不是本轮实施了 forward workspace reuse。

Native **dWeight 是 Native Im2col + CuPy/cuBLAS GEMM**；最终路径并没有用自研 GEMM 计算 dW。代码证据为 runtime.py::packed_dweight 中 gm @ col，不能把该乘法速度归为自研 C++ GEMM。Native Pool backward 也是 CuPy scatter + Native Col2im 的混合路径。

Native Conv forward 的五层 kernel 总时间都比 CuPy 长，然而正常完整调用在除 conv2 外的四层更快。这符合减少/简化 Python 提交路径可抵消某些 kernel 差距的解释，但本轮没有 Python 栈采样来作独立因果拆分；PROFILE_SUMMARY.csv 里的 CPU NVTX 与 CUDA API 数据来自 profiler，只是辅助证据。conv2 完整 array 比值 0.929×，属于必须展示的回退。

### tiny、holes、非连续与规模

- Col2im 36 个完整 array 调用中 Native 都更快，范围 1.675–2.761×。但 **holes 的 kernel 本身慢约 14–18%**：例如 contiguous 的 CuPy/Native 为 7.403/8.747 μs；完整同步为 42.009/23.773 μs。tiny kernel 基本相当，Native 完整调用约快 1.8×。这说明 kernel 更快与 binding 更快是不同结论，不能用一项替代另一项。
- tiny/holes 的 CPU Col2im 是 NumPy 竞争力很强的例子：tiny contiguous 6.448 μs、holes contiguous 9.944 μs，均快于相应 GPU 完整调用；当输入本就在 GPU，额外 D2H/H2D 不在本表中，不能直接据此切到 CPU。
- 单独 Native Im2col 必须把非连续 NCHW 输入 pack 成连续后交给 ABI，CuPy 原 ElementwiseKernel 可以按逻辑 strides 读取。tiny_sliced 的 Native/CuPy优势降至约 1.21×；不能从连续 shape 的结果外推所有 view。
- Col2im 的 training、contiguous、negative 三种真实物理布局全部测量；Native 使用实际 signed strides，没有把 view 当连续数组。结果在 COL2IM_RESULTS.csv 全部保留。
- GEMM 从 256³ 到 4096³，CPU 时间从约 0.17 ms 增至约 250 ms（以逐 shape 表为准），Native 在大型矩阵相对 CPU 的优势放大，但始终不等于优于 cuBLAS。卷积相关曲线同时改变几何/形状/元素数，是覆盖性趋势，不是控制其他变量不变的 scaling law。

## 哪些场景更合适

'''
rows=[]
for op in ops:
 z=[r for r in sp if r['op']==op and r['scope']=='array' and r['metric']=='sync_us' and r['baseline']=='cupy'];wins=sum(float(r['ratio_of_means'])>1 for r in z);positive=sum(float(r['paired_delta_ci95_low_us'])>0 for r in z);negative=sum(float(r['paired_delta_ci95_high_us'])<0 for r in z)
 rows.append([labels[op],f'{wins}/{len(z)}',positive,negative])
text+=table(['算子','Native 均值更快 cases','配对差值区间全为正','区间全为负'],rows)
text+='''
Im2col、Col2im 和 MaxPool forward 的完整接口在本覆盖内优势明显；Packed dWeight 的“布局优化+cuBLAS”组合也有稳定收益。MaxPool backward 只有中等收益，其中 asymmetric 的区间跨零，不宜宣称所有形状都稳定提升。

CuPy/cuBLAS 更适合大型/长 K GEMM；真实 Conv2 的完整调用也应选择 CuPy 作为比较优势方。少数小 GEMM 的 Native 完整调用略快，部分差异在波动内，且 GPU kernel 时间常不更快，不能宣称通用 GEMM 算法超过 cuBLAS。tiny/holes CPU Col2im 保持竞争力。

## 与此前训练收益的关系

1. 新数据确认 shape-aware dispatch 避免低 CTA 的 register 回退，支持此前设计动机；但真实 dI/dW 多数走 cuBLAS，不能把所有 GEMM microbenchmark 的收益都加进训练。
2. 本轮最终 Col2im kernel 与同步完整调用分开测，能说明“kernel 收益可能被 CPU 路径抵消”的机制；此前最初版本训练退化来自历史报告，本轮未重跑旧 binding + 全训练，不将历史数值与本轮数据拼接。
3. dWeight 少一次 copy/分配是本轮直接测得，能支持其减少数据搬运的解释；没有重新训练，所以不能给出本轮端到端提升百分比。
4. O1/F1 本轮一直显式开启但没有再做独立消融，不能把本轮 Native/CuPy差值完全归因于 O1/F1。

## 局限与交付索引

共享服务器未锁频，GPU时钟/CPU调度会变化；无 NCU occupancy/stall 计数器，不能断言精确占用率或 stall 原因。无 transfer-inclusive 或真实训练实验。正式 sync 时间的 short-op 末尾同步是摊销口径；不同 metric 不混算。只比较 FP32；高阶、非默认 stream、FP64 fallback 等生产语义由现有 GPU回归覆盖。没有增加本轮新的 sanitizer；本轮未修改 kernel。

七组图见 figures/01...07，每组 PNG/SVG；数字、图例、误差条、硬件与口径均标注。PPT_SUMMARY.md 给五页安排。完整复现命令在 REPRODUCE.md；环境/源码状态在 SOURCE_STATUS.md；原始样本、kernel详情、分配请求和所有回退 case 均保留。traces 的原始 Nsight 二进制与冻结动态库只记录服务器路径/SHA，不进入压缩包。报告生成后对源码和库逐一复核，未发现改变；不继续进行下一轮优化。
'''
# Replace the indicative small GEMM value with the measured value to avoid hard-coded approximation.
v=float(get('gemm','ext_256_256_256','numpy')['mean'])/1000
text=text.replace('约 0.17 ms',f'约 {v:.3f} ms')
write('REPORT.md',text)

ppt='''# PPT 五页建议

本文件中的所有表格和图片来自本轮最终评测；未使用历史时窗的性能数值。

1. **平台与三种实现**：A100-PCIE-40GB、CPU OpenBLAS 4线程、FP32/TF32关闭、F1/O1显式开启。用 SOURCE_STATUS.md 与 WORKLOAD.csv。说明 NumPy是CPU，CuPy/cuBLAS和Native是同GPU；dWeight与Pool backward是混合实现。每case18轮、GEMM array30轮，短操作5次循环摊销；主口径为驻留数据同步完成时间。
2. **正确性**：319 passed、2 CPU跳过；804组独立FP64诊断0失败，4096³完整参考；Native GEMM MaxAbs 1.34e-4 / RelativeL2 2.09e-6。放 figures/06_accuracy.png 与 CORRECTNESS.md，保留“近似一致、并非逐位一致”和历史严格失败未撤销。
3. **GEMM与规模**：放 figures/03_gemm_size_curve.png、05_optimization_ab.png 左图。240×64×432 / 72×64×576 最终相对旧register约3.10×/3.30×；1024³–4096³相对旧Shared约1.81–2.47×，但仍慢于cuBLAS。用 DISPATCH_DECISIONS.csv 标注实际CTA。
4. **Im2col / Col2im / Conv**：放 figures/01_threeway_latency.png 左图、04_im2col_col2im_scale.png。Im2col 6D→matrix copy 必须计入；Col2im holes kernel更慢但完整接口更快；Conv2回退必须保留。不要把 kernel 时间与正常 wall time 混在同一加速比。
5. **优化价值与适用范围**：放 figures/02_native_vs_cupy.png、07_advantage_regions.png，搭配 OPERATOR_AGGREGATE.csv、KERNEL_DETAILS.csv。dWeight五层少一个Copy与分配，conv2请求字节13.10MB→6.78MB。强调使用cuBLAS的是dWeight乘法；本轮不跑完整训练，不能宣称新的端到端训练加速。

所有图均有SVG可无损编辑；全case的退化、负stride、tiny、holes见CSV，不建议仅截取最快shape。图1/2聚合是等case权重均值，不是DonkeyNet训练权重；图3/5给shape级例子更适合讲算法机制。
'''
write('PPT_SUMMARY.md',ppt)
print('REPORT and correctness written; production hashes unchanged',len(old))
