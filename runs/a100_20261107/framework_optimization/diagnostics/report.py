from pathlib import Path
import csv,json,statistics,math,collections,hashlib,difflib,shutil,datetime
O=Path(__file__).resolve().parents[1];R=O.parents[2]
def read(n):return list(csv.DictReader((O/n).open()))
def write(n,rows):
 with (O/n).open('w') as f:w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def table(headers,rows):return '| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+''.join('| '+' | '.join(str(v) for v in row)+' |\n' for row in rows)
s=json.loads((O/'SUMMARY.json').read_text());p=s['paired'];ops=read('OPERATOR_COMPARISON.csv');summary=[]
for cat in dict.fromkeys(x['category'] for x in ops):
 vals=[]
 for case in dict.fromkeys(x['case'] for x in ops if x['category']==cat):
  d={x['implementation']:x for x in ops if x['case']==case};h=d.get('HyTorch',d.get('shared'));t=d['PyTorch'];vals.append((float(h['kernel_us'])/float(t['kernel_us']),float(h['sync_us_median'])/float(t['sync_us_median'])))
 summary.append(dict(category=cat,cases=len(vals),kernel_ratio_min=min(x[0] for x in vals),kernel_ratio_median=statistics.median(x[0] for x in vals),kernel_ratio_max=max(x[0] for x in vals),fullcall_ratio_median=statistics.median(x[1] for x in vals)))
write('OPERATOR_SUMMARY.csv',summary)
op_table=table(['类别','shape数','GPU时间比最小–最大','GPU比中位数','完整同步调用比中位数'],[[x['category'],x['cases'],f"{x['kernel_ratio_min']:.2f}–{x['kernel_ratio_max']:.2f}",f"{x['kernel_ratio_median']:.2f}",f"{x['fullcall_ratio_median']:.2f}"] for x in summary])
train_table=table(['配置','N','mean(s)','median(s)','样本SD(s)','samples/s'],[[m,s[m]['n'],f"{s[m]['mean_s']:.6f}",f"{s[m]['median_s']:.6f}",f"{s[m]['sd_s']:.6f}",f"{s[m]['samples_per_s']:.2f}"] for m in ['B0','F1','torch']])
sup_table=table(['模型','batch','B0(s)','F1(s)','wall减少','胜出'],[[x['model'],x['batch'],f"{x['B0_s']:.6f}",f"{x['F1_s']:.6f}",f"{x['reduction_pct']:.2f}%",f"{x['wins']}/{x['n']}"] for x in s['supplement']])
# Actual recorded shape/frequency evidence.
w=json.loads((O/'WORKLOAD_COMPLETE.json').read_text());freq=collections.Counter(x['function'] for x in w['function_calls']);write('WORKLOAD_FREQUENCIES.csv',[dict(function=k,calls_per_step=v) for k,v in freq.items()])
gemm=[]
for x in w['native_gemm_calls']:
 if x.get('path')=='cupy_tensordot':
  aa,bb=x['a']['shape'],x['b']['shape'];ax,bx=x['axes'];gemm.append(dict(path='dInput_CuPy_tensordot',M=math.prod(v for i,v in enumerate(aa) if i!=ax),N=math.prod(v for i,v in enumerate(bb) if i!=bx),K=aa[ax],calls_per_step=1))
 else:gemm.append(dict(path=x.get('path','native_forward'),M=x['M'],N=x['N'],K=x['K'],calls_per_step=1))
mms=[x for x in w['function_calls'] if x['function']=='MatMul']
for x in mms[3:]:
 aa,bb=[v['shape'] for v in x['inputs']];gemm.append(dict(path='Linear_backward_CuPy_dot',M=aa[0],N=bb[1],K=aa[1],calls_per_step=1))
write('GEMM_WORKLOAD.csv',gemm)
# CPU profile is separate from formal timing.
cpuprofiles=[]
for mode,path in [('B0','stage_a/cpu_B0.csv'),('F1','raw/cpu_F1.csv')]:
 for row in read(path):
  if row['function'] in ['_shape_ok','_shape_ok_uncached','prod','add_func',"<method 'sort' of 'list' objects>",'using_config','__call__']:
   cpuprofiles.append(dict(mode=mode,steps=30,**row))
write('CPU_FUNCTIONS.csv',cpuprofiles)

method='''## 实验方法与适用范围

同一物理GPU1 A100-PCIE-40GB；GPU0未使用。Python3.10.12、NumPy2.2.6、CuPy14.2.0、PyTorch2.9.1+cu128、CuPy CUDA runtime12090、cuDNN91002，动态库nvcc12.4 -O3 -std=c++14，未启用fast-math。TF32同时关闭，PyTorch正常cuDNN/autograd，benchmark=False，不使用torch.compile；SGD foreach=None/fused=None，实际trace为一个multi_tensor kernel。

算子计时：固定FP32输入(seed42，普通算子normal std0.1)、预热10次、15次平衡正反序；单独测Python提交、同步完整调用、CUDA event区间。event区间可能含CPU提交空隙，不当作纯kernel时间。纯kernel来自另一次15重复Nsight/CUPTI捕获，每调用kernel时间求和后取中位数。GEMM双方预分配输出；其它完整接口保留真实分配/packing成本。原始samples和波动均保留。SGD各实现先重置相同参数/梯度验证，再做各自调用测量，foreach与默认明确分列。

普通算子跨框架误差诊断为atol2e-5/rtol2e-4，全部零超限；不能用这个诊断替换原始单元测试门限。原测试完全未修改；历史严格GEMM失败记录保留raw/previous_gemm_*。新缓存与B0比较要求逐元素相等，训练另按既有权重1e-4/1e-3、梯度2e-5/2e-4、loss1e-4/1e-3检查全部元素/轨迹。

训练：256合成样本、batch4、3epochs192steps、seed42、相同初始化、SGD0.001、dropout0。两个框架均按step从同一DataLoader取NumPy数据并H2D，未单边预加载。GPU loss保留至完整训练最终同步，计时结束后读取全部轨迹；未把最终同步挪出区间。正式计时无profiler；配置B0/F1/Torch的六种排列各两遍，共12轮，paired t区间df11。先前F1固定居中的10轮完整保留raw/pilot_fixed_middle_order，不用于正式收益；调整理由是位置平衡，不是择优筛选。

阶段A原始数据及SHA冻结于stage_a。后补的15重复kernel采集和allocator诊断从baseline/source加载未经本轮修改的生产代码，旧数据不覆盖。阶段B正式比较B0为本轮代码开关关闭，F1开关开启，其余相同。GPU计时和profile都在本次连续窗口；没有拼接历史0.820923s/0.307929s作为分母。

Nsight CPU phase归因：同一训练进程串行phase区间按CUDA API correlationId关联kernel，包括PyTorch autograd worker线程；仅按主线程ID会漏掉其backward，已正确跨线程匹配时间区间。GPU累计时间与区间并集分别保存；CPU elapsed包含API、等待、线程调度，不代表纯Python on-CPU。不能把这些相加。没有锁频，也没有与每个trace同步的时钟记录。
'''
operator='''# HyTorch vs PyTorch 算子对比

比值统一为HyTorch时间/PyTorch时间，>1表示HyTorch更慢。GPU列为各shape的纯kernel中位时间之比；类别中位数不按训练频率加权，不能直接推算网络wall。

'''+op_table+'''
## 接近与落后

小激活和pool forward的kernel接近PyTorch，部分dBias/dInput更快；但完整Python调用仍经常慢约1.5–3倍。当前shared dispatch的真实前向GEMM约1.18–3.61倍于torch.matmul，packed dWeight的五层约1.16–2.02倍；SGD kernel约4.59倍，32个kernel对PyTorch的1个。两者均调用CuPy/cuBLAS或cuDNN的部分不能宣传为自研kernel优势。

conv1 forward明显落后；conv2/3在当前cuDNN算法和独立输入设置下GPU时间相近或更快，但fullcall不一定更快。这里不是cuDNN所有算法或所有shape的结论。dInput、dWeight、dBias分别让PyTorch只对相应输入requires_grad；FullBackward让x/W/b都需要梯度，且输出均匹配。真实模型PyTorch首层输入图像不需要梯度，HyTorch现有框架仍计算该dInput，因此孤立conv1 FullBackward中的Torch大耗时不能代入真实训练、不能据此宣称HyTorch训练backward更快。最初全部输入requires_grad的单项诊断已归档raw/diagnostic_all_inputs_require_grad，未使用其结果。

Linear表是完整F.linear与torch.nn.functional.linear接口；模型的Layer.forward实际使用F.linear_simple（MatMul+Add），数学等价但Python路径不同，该CPU比值不直接代表训练Linear层。实际训练调用和耗时已另由WORKLOAD_COMPLETE.json、TRAINING_HOTSPOTS.csv记录。Pooling为TinyNet/通用shape，DonkeyNet中调用频率为0。

GEMM表basic/blocked是当前实现；shared表示当前128CTA dispatch（可能选shared或register），register_old_dispatch是旧动态库ABI诊断，只有m/n/k>=64才选register，其它shape仍shared。具体kernel/grid/block/寄存器见OPERATOR_KERNELS.csv，旧库SHA在raw/old_gemm_library_sha.json，未替换生产库。

分配计数见OPERATOR_ALLOCATIONS.csv：CuPy MemoryHook请求与PyTorch allocator history，分别注明统计口径；不能将缓存allocator请求等同于cudaMalloc。Torch最初CPU profiler不暴露CUDA分配事件的零值被判为不可用、归档，正式表改用allocator history。最后allocator诊断与新外部任务最多重叠约6.7秒，只用于进程内请求数，不用于性能。copy/pack实际kernel名与全部memcpy记录保留于trace，不能把这些都算作H2D。

'''+method
(O/'OPERATOR_REPORT.md').write_text(operator)
# Profile actual counts and contributions.
hot=read('TRAINING_HOTSPOTS.csv');prtable=table(['trace','kernel/step','GPU累计us/step','活动并集us/step','kernel跨度内间隙us/step'],[[x['trace'],x['kernels_per_step'],f"{float(x['gpu_kernel_us_per_step']):.2f}",f"{float(x['gpu_active_union_us_per_step']):.2f}",f"{float(x['kernel_span_gaps_us_per_step']):.2f}"] for x in hot if x['phase']=='ALL'])
profile='''# 框架热点与训练时间线

'''+prtable+'''
阶段A实际训练142 vs72 kernel/step。HyTorch forward/backward/SGD为38/72/32，PyTorch33/38/1。两者均每step两次H2D，共633616bytes，计时捕获内无逐step D2H loss读取。没有找到值得删除的训练同步点；本轮不改变数据传输。

冻结trace中HyTorch GPU活动并集1.313ms/step、PyTorch1.065ms/step；kernel跨度内间隙3.585ms vs1.027ms。跨度差约2.806ms中，约0.248ms来自活动并集差、约2.558ms来自无kernel活动区间差。这是对短trace自身跨度的算术解释（约91%来自间隙），不是把profiling数字强行相加解释无profiler训练差距。间隙还可能包含拷贝、CPU等待、调度，不能全部称为纯Python。

CPU cProfile的30step：_shape_ok1140调用（38/step）、累计约13ms，np.prod约9ms；F1热缓存仍1140次_shape_ok，但不再执行该路径的np.prod。Function.__call__1980次（66/step）、backward30次，未改变；sort630次仅约0.41ms累计，不是首选。排序/对象与自动求导大改风险高，因此没有为了任务强行引入第二优化。CPU_FUNCTIONS.csv提供精确计数与累计/自身耗时。

F1不改变kernel数、数学工作或GPU分配数。B0_b/F1_b trace累计GPU时间约1.330/1.329ms/step；CPU提交间隙下降。F1_c所有未修改kernel几乎同比缩短，且无对应时钟日志，根因未确认（动态时钟为可能解释）。所有trace保留；不能将该GPU时间下降视为缓存修改了算法，也不依据它声称GPU加速。

当前约2.73倍训练差距主要体现为更高提交/调度成本、更多小kernel与SGD launch，以及部分原生GEMM性能差距；只有缓存候选的wall收益通过本轮受控消融建立了因果证据。没有完整证明每种剩余因素对2.73倍的精确贡献。
'''
(O/'FRAMEWORK_PROFILE.md').write_text(profile);(O/'PROFILE_SUMMARY.md').write_text(profile)
(O/'OPTIMIZATION_DESIGN.md').write_text('''# F1：有界 shape 校验结果缓存

唯一生产修改位于hytorch/backends/cuda/runtime.py：backend新增metadata_cache=False；_shape_ok根据开关仅对精确tuple且每个维度精确Python int使用256项LRU；miss调用原校验。np.integer、float、bool、list等走原路径，避免相等键使不同类型绕过检查。失败不缓存。

收益依据：38次/step重复纯shape校验、主要np.prod成本；整条conv/packed dWeight/gemm调用配对与正式训练验证。实现只保存不可变shape值及None成功结果，既不保存数值，也不保存GPU地址、array、stream、device。shape变化自然变成新key，容量满时逐出。shape合法性与设备无关，因此复用校验结果无需按device分key；每次实际buffer/dtype/device/alias检查完全保留。

风险控制：不改np.int64乘积的原判定、不改归约顺序、自动求导、Col2im、GEMM、SGD、workspace生命周期或任何CUDA ABI/kernel。原有shape规则本身没有扩大修复范围。默认关闭。仅对验证已证明纯函数的_shape_ok缓存，不缓存_geometry、dtype或设备结果。

候选排序：1)重复shape验证：38/step、可消除、局部低风险，本轮实现；2)Function/Variable开销：广泛但生命周期和高阶梯度风险大，不改；3)排序：小，收益上限低，不改；4)SGD融合：已有FTZ准入失败，不启用；5)数据H2D：两边2次/step相同且没有冗余D2H，不改变。

没有F2，因此最终版就是F1；不伪造组合消融。CPU tracemalloc在256/5000次唯一shape后约52KB/69.5KB，LRU始终最多256项（包括上下文与Python分配，非GPU内存）。GPU活跃峰值与分配次数不变。
''')
mem=read('MEMORY_COMPARISON.csv');memtable=table(['batch','模式','GPU活跃峰值MiB','malloc请求/768samples','clear后bytes'],[[b,m,f"{max(float(x['peak_active_bytes']) for x in mem if x['batch']==str(b) and x['mode']==m)/2**20:.3f}",next(x['malloc_calls'] for x in mem if x['batch']==str(b) and x['mode']==m),next(x['after_clear_bytes'] for x in mem if x['batch']==str(b) and x['mode']==m)] for b in [1,4,16] for m in ['B0','F1']])
correct='''# 正确性与安全验证

- 阶段A生产基线完整回归：272 passed、2 skipped；原有两个CPU DonkeyNet测试显式skip，对应GPU测试实际运行，HYTORCH_REQUIRE_CUDA=1。
- 新专项19 passed；覆盖FP32/FP64、CPU/GPU、非连续/负stride视图、不规则非方形卷积、batch动态变化、重复backward与累积、no_grad/create_graph/二阶梯度、输入原地修改后新forward、SGD hooks、stream、预分配输出、dtype拒绝、输出重叠拒绝、对象释放、cache容量/键类型/异常恢复。
- 修改后完整回归：291 passed、2个相同CPU skip；没有GPU不可用skip，没有降低门限。
- Compute Sanitizer memcheck：19 passed、0 errors；racecheck：19 passed、0 hazards。没有新增CUDA ABI或GPU内存路径，但仍测试了候选调用现有kernel的安全性。
- 15个完整调用路径B0/F1逐元素相等；36次正式训练的F1/B0全部loss轨迹、梯度和权重最大绝对误差0；所有supplement同样0。这里描述数值逐元素一致，未把assert_array_equal当作signed-zero位模式检查。
- PyTorch训练对齐全部通过，正式训练所有误差最大绝对值4.47e-8。算子跨框架诊断全部0超限；原始严格GEMM历史失败另保留，不能宣称已经修复。

原生产kernels.cu与动态库SHA完全不变，旧FP32修复和128CTA dispatch保持。旧融合SGD仍不可用且未启用；所有正式训练使用legacy SGD。

'''+memtable+'''
显存为CuPy MemoryHook活跃池峰值（非整卡峰值），不含Torch或CUDA库私有分配。reserved存在池碎片/缓存差异，全部原始数值保留。每组3轮cleargrads/gc/sync后活跃1011200bytes稳定，没有观测到逐轮增长。
'''
(O/'CORRECTNESS.md').write_text(correct)
# Incremental source diff only, excluding previous uncommitted work.
changes=[]
for rel in ['hytorch/backends/cuda/runtime.py','tests/test_framework_metadata.py']:
 old=O/'baseline/source'/rel;new=R/rel;before=old.read_text() if old.exists() else '';after=new.read_text()
 changes.append(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/'+rel if old.exists() else '/dev/null',tofile='b/'+rel)))
 dest=O/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(new,dest)
(O/'git_diff.patch').write_text(''.join(changes))
(O/'SOURCE_CHANGES.md').write_text('''# 源码改动（本轮增量）

1. hytorch/backends/cuda/runtime.py：ContextVar默认与backend增加metadata_cache=False；新增bool参数校验；_shape_ok分派到旧校验或256项LRU。旧检查主体移至_shape_ok_uncached，_shape_ok_cached只保留成功的不可变整数tuple检查结果。没有修改dtype/device/alias校验。
2. tests/test_framework_metadata.py：新增19项专项参数化测试。原有所有测试文件内容保持不变。

开启示例：

```python
with backend(matmul="shared", conv="im2col", pool="native",
             col2im="native", dweight="packed", metadata_cache=True):
    # 原训练代码，SGD(lr=0.001, backend="legacy")
    ...
```

建议保留这项小范围实现，在已验证工作负载显式开启；全局默认仍False。未实现第二候选。无CUDA源码/动态库修改，不需要重新编译；未修改自动求导、优化器、数据加载或workspace数值生命周期。

git_diff.patch为相对于本轮baseline/source的增量，baseline/working_tree.patch另保存此前全部未提交修改，勿混淆。诊断脚本位于diagnostics，性能数据不纳入生产代码。没有Commit、Push或PR。
''')
report='''# HyTorch A100 算子比较与最终框架优化

本轮已完成阶段A测量及阶段B实际源码优化。只保留一个有证据的候选F1：默认关闭的不可变shape校验缓存。没有修改CUDA计算kernel，没有启用被拒绝的融合SGD。

## 正式端到端结果

'''+train_table+f'''
F1 wall减少 **{p['wall_reduction_pct']:.2f}%**，吞吐加速 **{p['speedup']:.4f}×**，12/12配对胜出。平均节省 **{p['delta_mean_s']*1000:.2f}ms**，95%配对t置信区间 **[{p['ci95_s'][0]*1000:.2f}, {p['ci95_s'][1]*1000:.2f}]ms**。这是本轮B0分母，未引用历史窗口。

本轮B0/PyTorch={p['before_vs_torch']:.3f}×，F1/PyTorch={p['after_vs_torch']:.3f}×，仍有明显差距。只实现F1，没有F2或组合叠加收益；“最终组合版”是前几轮已验证配置加本轮F1。

## 算子对比

'''+op_table+'''
具体形状、误差、每次计时、kernel名/grid/block和allocator计数见配套CSV及OPERATOR_REPORT。类别中位数不等于训练贡献。部分kernel接近PyTorch，而fullcall仍落后，支持从框架提交路径优化。Linear接口差异、首层dInput梯度需求差异和通用shape范围在算子报告明确说明。

## 证据链与修改

真实step记录66次Function调用、38次_shape_ok、5次Conv forward/5次dWeight/5次dInput；1140次/30step重复np.prod检查是可安全消除的CPU工作。新增256项纯shape验证缓存，仍逐调用检查当前dtype/device/buffer，未保存GPU指针或stream。完整conv/packed dWeight/gemm调用15次配对支持减少提交成本，再由上述12轮无profiler训练验收。

训练kernel数仍142/step，im2col仍10/step，SGD仍32/step；并未把Python缓存包装为CUDA算法加速。PyTorch72/step且SGD为1个kernel。GPU空闲间隙与CPU phase elapsed差距比GPU计算差距更大，但未声称精确拆解全部Python在CPU上运行的时间。详细时间线见FRAMEWORK_PROFILE。

## 补充模型/批大小

'''+sup_table+'''
每项仅3对，支持这些配置内的可重复性，不代表真实数据精度或所有网络通用加速。

## 正确性、内存与保留建议

完整291 passed、2个既有CPU skip，GPU实际执行；memcheck/racecheck零错误。F1/B0的训练loss轨迹、参数和梯度逐元素数值一致。GPU活跃峰值、分配数保持不变，无workspace生命周期变化。建议保留F1并显式启用；默认仍关闭。未修改高阶求导、GEMM、Col2im、dWeight数学计算或SGD。

## 限制与异常完整披露

- 首轮脚本误用SGD.step已改为HyTorch.update，失败日志保留；未产生可用计时。
- 初始单项梯度全部requires_grad的诊断、F1固定居中的pilot训练均完整归档，正式结果未择优混用。
- 一次F1_c短trace里所有未修改kernel同比变快，缺少同期时钟数据，不能证明原因，更不能算作kernel优化收益。正式wall验收不使用profiler。
- GPU外部新任务启动晚于正式训练和所有计时trace，最后allocator history诊断存在约6.7秒可能重叠，只用于本进程计数。见raw/GPU_WINDOW.md。结束后本任务GPU进程已退出，不代表GPU1当前空闲。
- 历史严格GEMM失败记录保留；本轮未改变数值门限。allocator口径不包含所有库私有分配；不宣称显存优化收益。

'''+method+'''
## 交付导航

PPT_SUMMARY.md为六页汇报；figures/包含6张PNG及对应SVG。原始trace在traces/，stage_a/及FROZEN_SHA.json冻结原阶段A，raw/保留全部命令/失败/诊断/测试日志。REPRODUCE.md说明如何在新目录执行，不覆盖证据。
'''
(O/'REPORT.md').write_text(report)
ppt=f'''# PPT 六页汇报提纲

## 1. 算子速度不等于完整调用速度

图：figures/01_operators.png（同名SVG可编辑）。kernel层激活/pool接近PyTorch，完整调用仍明显落后；SGD32kernel对1kernel，当前实现存在FTZ约束，不能直接启用旧融合原型。类别比为样例中位数，不是网络加权贡献。

## 2. 五层卷积逐项对照

图：figures/02_conv_layers.png。展示forward/dWeight/dInput/dBias/full backward，数学操作相同。强调首层孤立input梯度与实际PyTorch训练需求不同；不把孤立层GPU优势宣传为训练优势。Linear功能接口与Layer路径亦有差异，见OPERATOR_REPORT。

## 3. 框架调度是可验证机会

图：figures/03_gpu_cpu.png。实际142 vs72 kernel/step；冻结trace GPU活动1.313 vs1.065ms，kernel跨度内间隙3.585 vs1.027ms。CPU/GPU相互重叠，不相加。38次/step纯shape验证是本轮具体可消除工作。

## 4. 生产代码的小范围修改

图：figures/04_training.png。256项LRU只缓存已验证整数shape，不缓存地址、stream、设备、数值；每次buffer安全校验保留。正式B0 {s['B0']['mean_s']:.6f}s → F1 {s['F1']['mean_s']:.6f}s，wall减少{p['wall_reduction_pct']:.2f}%。正常PyTorch仍{s['torch']['mean_s']:.6f}s。

## 5. 平衡配对验收

图：figures/05_ablation.png。六种三配置排列各2次，12/12胜出；平均节省{p['delta_mean_s']*1000:.2f}ms，95%配对区间[{p['ci95_s'][0]*1000:.2f},{p['ci95_s'][1]*1000:.2f}]ms。只有一个候选，不声称双优化叠加。完整291pass、2既有CPUskip，Sanitizer零错误；F1/B0 loss/梯度/权重数值一致。

## 6. 成本与结论边界

图：figures/06_memory.png。GPU内存峰值、分配数相同；CPU cache最多256项，约52–70KB traced分配。batch1/16/TinyNet补充均3/3胜出。推荐显式开启F1、保留默认关闭；未优化CUDA算法。保留时钟波动trace与最后分配诊断共享任务说明，不用其计时宣称收益。

所有图注明A100、batch、方法、单位、数据源。图片为本轮新窗口数据，未将历史2.67×当作本轮性能分母。
'''
(O/'PPT_SUMMARY.md').write_text(ppt)
# Verify scope, frozen evidence, and preserve source hashes.
base=json.loads((O/'baseline/SHA.json').read_text());changed=[x['path'] for x in base if hashlib.sha256((R/x['path']).read_bytes()).hexdigest()!=x['sha256']]
assert changed==['hytorch/backends/cuda/runtime.py'],changed
frozen=json.loads((O/'stage_a/FROZEN_SHA.json').read_text());assert all(hashlib.sha256((O/k).read_bytes()).hexdigest()==v for k,v in frozen['files'].items())
(O/'SHA.json').write_text(json.dumps({'scope_changed_baseline_files':changed,'new_test':'tests/test_framework_metadata.py','files':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [R/'hytorch/backends/cuda/runtime.py',R/'tests/test_framework_metadata.py',R/'hytorch/backends/cuda/kernels.cu',R/'hytorch/backends/cuda/build/libhytorch_cuda.so']},'stage_a_frozen_verified':True},indent=2))
(O/'STATUS.json').write_text(json.dumps({'status':'completed','candidate':'F1 immutable shape validation cache','default_enabled':False,'full_tests_passed':291,'existing_cpu_skips':2,'formal_rounds':12,'production_cuda_changed':False,'own_gpu_processes_exited':True,'external_task_after_formal_timings':'see raw/GPU_WINDOW.md'},indent=2))
print('Reports written; source scope and frozen SHA verified')
