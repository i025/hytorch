"""Generate human-reviewable handoff using completed measurements only."""
from pathlib import Path
import json,csv,statistics,hashlib,subprocess,shutil,difflib
ROOT=Path.cwd();O=ROOT/'runs/a100_20261106/gemm_shape_dispatch';R=O/'raw';PREV=ROOT/'runs/a100_20261106/fp32_fix_and_profile'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
g=list(csv.DictReader((O/'GEMM_COMPARISON.csv').open()));tr=list(csv.DictReader((O/'TRAINING_COMPARISON.csv').open()));dec=list(csv.DictReader((O/'DISPATCH_DECISIONS.csv').open()));stats=json.loads((R/'training_statistics.json').read_text());pro=json.loads((R/'profile_summary.json').read_text())
def table(h,rows):return '| '+' | '.join(h)+' |\n|'+'|'.join(['---']*len(h))+'|\n'+'\n'.join('| '+' | '.join(str(v) for v in r)+' |' for r in rows)+'\n'
small=table(['M,N,K','tile','实现','无 profiler event μs','独立 CUPTI kernel μs','CTAs'],[[','.join(r[k] for k in ['M','N','K']),r['tile'],r['implementation'],f"{float(r['median_us']):.3f}",f"{float(r['profile_direct_kernel_us']):.3f}",next(d['actual_ctas'] for d in dec if (d['shape_id'],d['tile'],d['implementation'])==(r['shape_id'],r['tile'],r['implementation']))] for r in g if (r['M'],r['N'],r['K']) in [('240','64','432'),('72','64','576')] and r['tile']=='16'])
big=table(['size','tile','旧 μs','新 μs','forced shared μs','新相对 shared 加速'],[[r['M'],r['tile'],f"{float(next(x['median_us'] for x in g if x['shape_id']==r['shape_id'] and x['tile']==r['tile'] and x['implementation']=='old')):.3f}",f"{float(r['median_us']):.3f}",f"{float(next(x['median_us'] for x in g if x['shape_id']==r['shape_id'] and x['tile']==r['tile'] and x['implementation']=='forced_shared')):.3f}",f"{float(r['speedup_vs_forced_shared']):.3f}×"] for r in g if r['implementation']=='new' and r['group']=='square'])
regressions=[r for r in g if r['implementation']=='new' and float(r['speedup_vs_old'])<.95]
reg=table(['M,N,K','tile','register candidate CTAs','旧 μs','新 μs','新版延迟增加'],[[','.join(r[k] for k in ['M','N','K']),r['tile'],((int(r['M'])+63)//64)*((int(r['N'])+63)//64),f"{float(next(x['median_us'] for x in g if x['shape_id']==r['shape_id'] and x['tile']==r['tile'] and x['implementation']=='old')):.3f}",f"{float(r['median_us']):.3f}",f"{(1/float(r['speedup_vs_old'])-1)*100:.2f}%"] for r in regressions])
train_table=table(['实现','wall s（均值±SD）','samples/s 均值','最终 loss','与旧版最终权重'],[[m,f"{stats[m]['wall_mean']:.6f} ± {stats[m]['wall_sd']:.6f}",f"{stats[m]['samples_per_s']:.2f}",f"{stats[m]['loss']:.12g}",'逐位相同'] for m in ['old','new']])
pt=table(['实现','GPU kernel 总计','Forward GPU','Backward GPU','SGD GPU','Native GEMM GPU','CUDA API CPU elapsed','同步 API CPU elapsed'],[[m,f"{pro[m]['kernel_us_per_step']:.3f}",*[f"{pro[m]['phase_gpu_us'][k]:.3f}" for k in ['forward','backward','update']],f"{pro[m]['native_gemm_us_per_step']:.3f}",f"{sum(v['us_per_step'] for v in pro[m]['cuda_api'].values()):.3f}",f"{sum(v['us_per_step'] for k,v in pro[m]['cuda_api'].items() if 'Synchronize' in k):.3f}"] for m in ['old','new']])
fail=[r for r in g if r['original_strict_gate']=='False']
failtable=table(['M,N,K','tile','实现','FP64 相对 L2'],[[','.join(r[k] for k in ['M','N','K']),r['tile'],r['implementation'],f"{float(r['fp64_relative_l2']):.3g}"] for r in fail])
# Validate only the requested production dispatch differs from the saved baseline.
before=(R/'kernels_before.cu').read_text();after=(ROOT/'hytorch/backends/cuda/kernels.cu').read_text()
start=after.index('    // Conservative shape-only heuristic;');end=after.index('    cudaStream_t s=(cudaStream_t)stream;',start)
reconstructed=after[:start]+after[end:];reconstructed=reconstructed.replace('else if (use_register)','else if (variant==2 && m>=64 && n>=64 && k>=64)')
assert reconstructed==before
hashes=json.loads((R/'before_hashes.json').read_text());assert sha(ROOT/'hytorch/functions.py')==hashes['hytorch/functions.py'];assert sha(ROOT/'hytorch/backends/cuda/runtime.py')==hashes['hytorch/backends/cuda/runtime.py'];assert sha(O/'lib_before.so')==hashes['hytorch/backends/cuda/build/libhytorch_cuda.so']
(R/'dispatch_only.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='before/hytorch/backends/cuda/kernels.cu',tofile='after/hytorch/backends/cuda/kernels.cu')))
# Full working tree diff includes prior edits; new tests included via no-index diff.
patch=subprocess.check_output(['git','diff'])
for f in ['tests/test_gemm_dispatch.py','tests/test_backward_dtype.py']:
 r=subprocess.run(['git','diff','--no-index','--','/dev/null',f],capture_output=True);assert r.returncode==1;patch+=r.stdout
(O/'git_diff.patch').write_bytes(patch)
(R/'git_status.txt').write_bytes(subprocess.check_output(['git','status','--short']))
for f in ['SOURCE_CHANGES.md','CORRECTNESS.md','REPORT.md']:pass
(O/'SOURCE_CHANGES.md').write_text('''# 本轮源码修改

## 生产代码：hytorch/backends/cuda/kernels.cu:98

只修改 hy_gemm 的选择条件，保留 ABI、全部参数检查、stream 参数、variant0/1 路径、gemm_shared/register 数学实现以及 FP32 累加语义。

```cpp
const int64_t grid_m = ((int64_t)m + 63) / 64;
const int64_t grid_n = ((int64_t)n + 63) / 64;
const int64_t register_blocks = grid_m * grid_n;
constexpr int64_t kMinRegisterBlocks = 128;
const bool use_register = variant == 2 && m >= 64 && n >= 64 && k >= 64 &&
                          register_blocks >= kMinRegisterBlocks;
```

原先仅 m/n/k>=64 即用 register；现在还需 register 候选 CTA>=128。否则 variant2 沿原有 else 使用 shared。计算采用 int64，未增加设备查询、同步、数据拷贝或新的 C ABI。

128 是用户指定待验证初值；报告已列出 120/127 CTA 的反例，本轮没有擅自调参或改 kernel。构建继续 sm_80、-O3、-std=c++14、-lineinfo、shared cudart，无 TF32/fast-math。

通过把新增条件还原后与 raw/kernels_before.cu 整文件比较，确认除此之外无任何变化；hytorch/functions.py 与 runtime.py SHA 完全不变。前一轮 FP32 修复和整数 dtype guard 保留。

## 测试

- tests/test_backward_dtype.py：将现有整数输入 MSE 测试参数化到 CPU/CUDA，完成上轮未验证路径。
- tests/test_gemm_dispatch.py：21个明确形状 × tile16/32=42项实际选择与数值测试；另4项 variant0/1 不重定向测试。CUDA Graph debug DOT 是运行时实际 kernel 节点，断言预期名字，并执行图后核对 FP64 参考。包含预分配输出、非默认 stream、连续3次调用、CTA127/128/129、维度63/64/65、极不对称、K尾块和不规则矩形。大型1024/2048/4096方阵另外在性能基准逐位/FP64参考及图节点验证。

## 实验脚本

仅放在 ignored runs/a100_20261106/gemm_shape_dispatch/diagnostics/，没有生产性能 hook。保存修改前/后的 .so 用于同进程 rt._library handle切换。forced_shared 使用此前保存的原始 shared 库；三个库用同一输入和输出预分配方式。

CuPy Python cuBLAS包装层拒绝 graph capture，因此测量脚本使用 CuPy 创建的同一handle、同一已加载 cuBLAS 动态库的标准 cublasSgemm_v2 ABI，并设置默认FP32 math。行主序 A@B 通过列主序 C^T=B^T@A^T实现。没有改CUDA数学或降低精度。失败的初次尝试保留在 raw/gemm_paired.log。

## diff与既有工作

raw/dispatch_only.patch 是这次生产修改。git_diff.patch 是包含既有未提交修改和新增测试的完整diff：其中原 register kernel、README、其他旧测试修改不是本轮新增。raw/before.patch、before_status.txt 和 before_hashes.json 保存开始状态。

没有 Commit、Push 或 PR。
''')
(O/'CORRECTNESS.md').write_text(f'''# 正确性与数值门限

## 基线

先恢复并核验保存的旧 dispatch/library，最终 dtype guard源码不改：56项 GPU 测试通过（45原CUDA + 11 dtype/完整DonkeyNet/整数输入），0 skipped；22原主机 + 9 dtype主机通过。

第一次合并测试曾 48 passed / 8 failed：native CUDA库先装载 CUDA12.4 runtime，之后导入PyTorch遇到 cudaGetDriverEntryPointByVersion 未定义符号。它是装载顺序错误，未进入这些测试的数值逻辑。随后测试进程先 import torch，采用一致runtime，56项通过。原失败日志和重试完整argv保留，不隐藏失败。

准备阶段一次编译在收到失败结果前已启动；随后旧源码和旧库恢复，成功基线确认后才应用并重新构建本轮dispatch。以 raw/baseline_retry_commands.json 后的成功构建/测试作为验收证据。

## 新版本

- 102 项实际 GPU 测试通过，无 skipped（原45 + dtype11 + 新dispatch46）。
- 22项原主机 + 9项 dtype 主机通过。
- 42组共享GEMM边界测试记录真实 CUDA Graph kernel；基本/blocked另4项直接检查未被重定向。
- 47组shape×2tiles×4实现=376行性能/数值记录；每组12次配对；全部old/new/forced_shared输出逐位相同。
- 所有实现相对FP64参考的L2误差最大 {max(float(r['fp64_relative_l2']) for r in g):.9g}，仅为补充误差描述。
- 7轮、两实现训练最终loss和全部权重逐位相同；参数和梯度实测FP32。FP32生产源码完整SHA不变。

## 原严格门限：仍有失败，未降低标准

NumPy FP32参考、atol=rtol=1e-4 原门限原样记录。下列22行未通过；old/new/forced_shared失败集合相同，不是新dispatch引入。2048/4096时cuBLAS也失败，不能据此删除原门限。

{failtable}

此前轮次的严格门限记录保留在 raw/previous_*。不同固定seed可改变接近零元素的相对误差门限是否触发，所以不能通过更换seed掩盖失败。本次seed=9000+shape_id事先固定，对四实现完全相同。逐位相同和相对L2指标不能替代原严格gate通过；报告不声称“所有GEMM严格数值检查通过”。

## 实际选择证据

DISPATCH_DECISIONS.csv 对应每种实现/shape/tile的图节点kernel、独立CUPTI kernel名字、实际grid/block/CTA/register/shared；native图选择与独立CUPTI完全匹配。非默认stream、预分配out和连续调用在测试中执行。未仅用Python计算的predicate冒充实际dispatch。
''')
(O/'REPORT.md').write_text(f'''# 通用 Shape-Aware GEMM Dispatch：A100 验证

## 三个问题的回答

1. **目标小矩形恢复到 shared 水平**：tile16 无 profiler CUDA-event中位数，(240,64,432) 81.616→21.088 μs（3.870×），(72,64,576) 67.744→17.248 μs（3.928×）。独立直接调用CUPTI分别95.392→25.2645、124.976→32.320 μs，和强制shared的25.248/32.3045 μs一致到小于0.1%的差异。
2. **大方阵收益保留**：1024³/2048³/4096³在两个tile配置下全部仍用register，与旧dispatch接近，最大这次中位延迟变化约0.87%；相对forced shared仍有1.75–2.47×加速。
3. **端到端仅微小收益**：7轮交替同进程配对，wall均值0.834571→0.831816 s，仅减少0.330%。前向GPU省下约194 μs/step，但不能直接加到/减出wall。不能用约4×局部加速宣传显著训练加速。

**128 CTA不是已验证最优的通用阈值**。额外矩形测试发现120/127 CTA反例，延迟增加10.77–26.39%。实现严格保留用户指定第一版，交由ChatGPT审查，不自行重调启发式。

## 代码和可靠基线

main HEAD edb6fbe3bfaa9eed461addd29e6943b22e28216e，工作区开始已有前轮FP32修复、register kernel和测试/文档未提交修改。旧CUDA库在 lib_before.so、旧CUDA源码在 raw/kernels_before.cu，SHA随清单保存。

补齐最终整数dtype保护GPU回归后才将新策略作为验收候选。基线56GPU/31host通过；新版102GPU/31host通过。hytorch/functions.py和runtime.py字节不变。详见CORRECTNESS.md及SOURCE_CHANGES.md。

生产变化只有hy_gemm的shape判断：m/n/k>=64且ceil(M/64)*ceil(N/64)>=128才用register。其他variant、ABI、stream、kernel数学、精度门限全部不变，没有设备属性查询、额外同步、TF32、fast-math或网络专用shape特例。

## 实验环境与方法

物理GPU1 A100-PCIE-40GB，108SM，逻辑CUDA0；GPU0 VLLM服务保持。每次阶段启动前确认GPU1空闲，没有终止别人的进程或锁定时钟。北京时间约11:43–11:49同一实验会话，因两个诊断环境错误有短暂停顿，进程间均重新查空闲；最终GPU1无本轮进程/显存占用。完整UTC时间在命令JSON。

CUDA nvcc12.4.99，sm_80；驱动580.95.05；CuPy14.2.0、NumPy2.2.6、PyTorch2.9.1+cu128。具体构建argv、CPU线程数、环境、GPU快照均随包保存。未锁频，因此不将不同窗口绝对延迟直接比较。

四实现同进程切换已加载库/标准cuBLAS handle，同一随机输入和预分配输出。47组M/N/K：24组来自已有实际FP32 DonkeyNet instrumentation（同模型/配置未改，原完整记录已复制），另23组为方阵与通用边界矩形；两个tile，4实现，共376行。实际训练trace核对8个前向native调用，其中只conv4/conv5改变kernel。

seed=9000+shape_id；5次普通warmup并额外预上传/回放graph；12轮正反交替。事件包围CUDA Graph回放：大工作量图含1次调用，中等8次，小工作量32次，除以次数。排除H2D/D2H/输出验证和逐kernel的Python提交间隙。该结果代表重复同矩阵的热缓存吞吐；不把它冒充真实训练缓存状态或孤立单次launch延迟。

CuPy的cuBLAS wrapper不支持当前stream capture，实际cuBLAS组用同一库/handle的标准SGEMM接口，默认FP32 math，NVIDIA_TF32_OVERRIDE=0。它是cuBLAS参考，不是声称调用了cp.dot。每组实际kernel从graph DOT及另外2次直接调用的Nsight记录核对。主性能统计12次事件；2次profile仅辅助kernel/配置核验，不能冒充12次profile统计。

## 小矩形结果

{small}

Graph事件和直接CUPTI绝对时间不同源于缓存/提交/采样条件，分别比较同口径的旧/新/forced shared，不交叉计算加速。microbench中conv4/5旧register分别4/2CTAs，新shared为60/20CTAs（tile16）；两者输出逐位一致。

## 大型方阵

{big}

1024³、2048³、4096³的register候选CTAs分别256、1024、4096，全部超过阈值，实际kernel为gemm_register<16/32>。大矩阵既有收益保留，未重写数学实现。

## 泛化检查发现的反例

{reg}

这些是测量事实。单一CTA门限忽略K长度、长宽比、每CTA工作量及tile，可能在120/127等阈值附近过于保守。这里的解释是性能假设，不是硬件计数器证明；本轮没有测occupancy/带宽/stall。报告完整披露反例，不能宣称“所有小矩形都更快”。

其他形状和波动（mean/std/min/max、12次原值）全部在GEMM_COMPARISON.csv。DISPATCH_DECISIONS.csv列实际名字/CTA/grid/block，new退回shared与forced_shared性能一致并不证明128是最佳界限。

## 端到端配对训练

固定seed42、相同初始权重/数据hash、256合成样本、batch4、3epochs、192steps、SGD .001、无dropout/momentum/weight decay。每模式10warmup，7轮交替且每次恢复初始状态；模型、FP32修复、SGD、数据、计时方式完全一样，唯一区别是CUDA library的dispatch。

wall包含CPU DataLoader、pageable H2D、前后向、更新和最终同步，无逐步日志/loss读取/显式同步；两模式同样phase NVTX。loss与权重读取在计时外。

{train_table}

7个配对均新版略快，平均省{stats['paired_wall_delta_mean_s']*1000:.3f} ms/192steps；配对差值样本SD={stats['paired_wall_delta_sd_s']*1000:.3f} ms。以t分布6自由度计算的描述性95%区间为[{stats['paired_wall_delta_95_t_interval_s'][0]*1000:.3f}, {stats['paired_wall_delta_95_t_interval_s'][1]*1000:.3f}] ms。区间仅针对本窗口7次测量，不保证不同GPU负载/时钟/模型下持续收益。工程意义上很小，不应宣传成显著端到端提速。

所有14次训练的loss完全相同，最终权重hash均为2cf9c1a0745128aa7809d046befd5839b3727209d86f26d23d7f1c9e5791f094；相对旧版最大绝对权重差0。

## 时间线追溯（独立10-step trace，单位μs/step）

{pt}

Native GEMM是Forward中的子集，不能再与Forward相加；API CPU elapsed和GPU duration重叠，不能相加。TRAINING_COMPARISON.csv把各模式一份trace汇总附到7轮wall记录，每行明确注明不是7次profile。

真实训练conv4旧register113.226→shared30.093 μs；conv5 149.457→38.685 μs，合计省193.905 μs，解释了前向整体194.343 μs减少。其他6个native GEMM仍shared且时间接近；backward仍走CuPy/cuBLAS，880.524→880.705 μs基本不变，FP64转换kernel仍为0。kernel总数均147/step，主要是选择更快kernel而非减少launch。

GPU kernel+memcpy活动时间并集占profile墙钟：旧34.54%、新31.06%。这是trace时间覆盖比例，**不是occupancy或硬件利用率计数器**。GPU工作缩短后活动间隙比例增加，和wall收益较小一致；CPU逐算子提交、数据准备、异步重叠使GPU工作节省大部分未变成wall节省。CPU未开栈采样，不能把全部间隙精确归因于Python。

## 正确性限制与后续决定

所有old/new/forced_shared输出逐位相同，FP64误差指标无新增回退；但原严格NumPy FP32 allclose gate仍有22行失败（长K与2048/4096，包含cuBLAS的方阵失败）。门限和失败记录均保留，详情CORRECTNESS.md。不能以相对L2代替严格门限通过。

128 CTA启发式成功修复指定两形状并保留大方阵收益，但已经存在通用矩形反例，端到端仅0.33%改善。是否接受当前阈值、进一步设计通用dispatch，或转向col2im/dWeight，留给ChatGPT审查。本轮不进行这些后续优化，不提交/推送/创建PR。
''')
# Source and prior evidence snapshots; omit binaries.
files=[]
for entry in json.loads((PREV/'SOURCE_FILES.json').read_text()):
 rel=Path(entry['path']);src=ROOT/rel.relative_to('source')
 if not src.is_file():src=PREV/rel
 dst=O/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst);files.append(dst)
for rel in ['tests/test_gemm_dispatch.py']:
 dst=O/'source'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,dst);files.append(dst)
(O/'SOURCE_FILES.json').write_text(json.dumps([dict(path=str(p.relative_to(O)),bytes=p.stat().st_size,sha256=sha(p)) for p in files],indent=2))
for name in ['gemm_baseline_summary.json','gemm_paired_comparison.csv']:
 shutil.copyfile(ROOT/'runs/a100_20261106/operator_handoff/raw'/name,R/('previous_'+name))
shutil.copyfile(PREV/'GEMM_REAL_SHAPES.csv',R/'previous_fp32_gemm_real_shapes.csv')
shutil.copyfile(PREV/'raw/workload/operators.json',R/'previous_observed_workload.json')
subprocess.run(['git','diff','--check'],check=True)
print('Reports written; source files',len(files))
