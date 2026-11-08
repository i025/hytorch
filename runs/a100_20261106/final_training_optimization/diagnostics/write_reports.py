from pathlib import Path
import csv,json,statistics,hashlib,shutil,ast,difflib,importlib.metadata,platform
O=Path('runs/a100_20261106/final_training_optimization');R=Path.cwd()
def read(f):return list(csv.DictReader((O/f).open()))
def table(headers,rows):return '| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+''.join('| '+' | '.join(str(v) for v in r)+' |\n' for r in rows)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
s=json.loads((O/'raw/training_summary.json').read_text());tr=read('TRAINING.csv');mem=read('MEMORY.csv');pr=read('PROFILE_SUMMARY.csv');op=read('FULL_WRAPPERS.csv');refs=read('REFERENCES.csv');sup=read('SUPPLEMENT.csv');errs=read('TRAINING_ERRORS.csv')
actual={'B0':'D关 / SGD原路径','D':'D开 / SGD原路径','S':'D关 / SGD请求native但已回退','DS':'D开 / SGD请求native但已回退'}
table_train=table(['配置','实际执行','平均±样本SD (s)','median (s)','samples/s','B0−该组 95%配对CI(ms)','胜出'],[[m,actual[m],f"{s[m]['mean']:.6f} ± {s[m]['sd']:.6f}",f"{s[m]['median']:.6f}",f"{s[m]['samples_per_s']:.2f}",'-' if m=='B0' else f"[{s[m]['ci95_s'][0]*1000:.3f}, {s[m]['ci95_s'][1]*1000:.3f}]",'-' if m=='B0' else f"{s[m]['wins']}/12"] for m in ['B0','D','S','DS']])
summary_num=[]
for filename,label in [('NUMERICS.csv','固定+留出'),('TRAINING_DWEIGHT_NUMERICS.csv','真实训练 x/gy')]:
 rr=read(filename)
 for cmp in ['A_R','B_R','B_A']:
  zz=[x for x in rr if x['gate']=='primary' and x['comparison']==cmp];summary_num.append([label,cmp,len(zz),sum(int(x['failed'])>0 for x in zz),sum(int(x['failed']) for x in zz),f"{max(float(x['max_abs']) for x in zz):.9g}",f"{max(float(x['rel_l2']) for x in zz):.6g}"])
numtable=table(['集合','比较','case数','失败case','失败元素','max abs','max relative L2'],summary_num)
wr=table(['层','B0/D CPU提交中位数 μs','B0/D 同步完整调用中位数 μs'],[[case,' / '.join(f"{statistics.median(float(r['cpu_submit_us']) for r in op if r['case']==case and r['layout']=='contiguous' and r['mode']==m):.3f}" for m in ['B0','D']),' / '.join(f"{statistics.median(float(r['synchronized_us']) for r in op if r['case']==case and r['layout']=='contiguous' and r['mode']==m):.3f}" for m in ['B0','D'])] for case in ['conv1','conv2','conv3','conv4','conv5']])
profiletable=table(['配置','trace','总kernel/step','dWeight kernel/step','im2col/step','SGD kernel/step','dWeight GPU μs/step','全GPU μs/step'],[[r['mode'],r['repeat'],r['kernels_per_step'],r['dweight_kernels_per_step'],r['im2col_per_step'],r['update_kernels_per_step'],f"{float(r['dweight_us_per_step']):.3f}",f"{float(r['GPU_us_per_step']):.3f}"] for r in pr])
memtable=table(['batch','B0/D峰值活跃 MiB','B0/D峰值保留 MiB','3轮后活跃 MiB'],[[b,' / '.join(f"{max(float(r['peak_active_bytes']) for r in mem if int(r['batch'])==b and r['mode']==m)/2**20:.3f}" for m in ['B0','D']),' / '.join(f"{max(float(r['peak_reserved_bytes']) for r in mem if int(r['batch'])==b and r['mode']==m)/2**20:.3f}" for m in ['B0','D']),' / '.join(f"{float(next(r['after_clear_active_bytes'] for r in mem if int(r['batch'])==b and r['mode']==m and r['repeat']=='2'))/2**20:.3f}" for m in ['B0','D'])] for b in [1,4,16]])
supp=[]
for kind,batch in [('donkey',1),('donkey',16),('tiny',4)]:
 vals={m:[float(r['wall_s']) for r in sup if r['model']==kind and int(r['batch'])==batch and r['mode']==m] for m in ['B0','D']};delta=[a-b for a,b in zip(vals['B0'],vals['D'])];supp.append([kind,batch,f"{statistics.mean(vals['B0']):.6f}",f"{statistics.mean(vals['D']):.6f}",f"{100*(1-statistics.mean(vals['D'])/statistics.mean(vals['B0'])):.2f}%",f"{sum(d>0 for d in delta)}/3"])
supptable=table(['网络','batch','B0 mean s','D mean s','wall减少','胜出'],supp)
reftable=table(['框架参考','次数','mean±SD(s)','samples/s mean'],[[m,5,f"{statistics.mean(float(r['wall_s']) for r in refs if r['mode']==m):.6f} ± {statistics.stdev(float(r['wall_s']) for r in refs if r['mode']==m):.6f}",f"{statistics.mean(float(r['samples_per_s']) for r in refs if r['mode']==m):.2f}"] for m in ['cupy','torch']])
failedops=[]
for case,layout in dict.fromkeys((r['case'],r['layout']) for r in op if r['operator']=='dweight'):
 v={m:statistics.median(float(r['synchronized_us']) for r in op if r['case']==case and r['layout']==layout and r['mode']==m) for m in ['B0','D']}
 if v['D']>v['B0']:failedops.append([case,layout,v['B0'],v['D']])
(O/'REPORT.md').write_text(f'''# HyTorch 最后一轮集中优化：交付报告

实验日期：2026-11-07（Asia/Shanghai；命令日志使用UTC）。仅本窗口实测，不拼接历史百分比。

## 结论与推荐

**接受D：反向列矩阵直接布局生成；拒绝S：多张量SGD未通过subnormal逐位一致性。** 推荐本合成workload显式使用 `backend(matmul="shared",conv="im2col",pool="native",col2im="native",dweight="packed")` 和 `SGD(lr=.001,backend="legacy")`。全局默认仍dweight=cupy、col2im=cupy、SGD=legacy。

最终冻结源码下D训练平均{s['D']['mean']:.6f}s，B0为{s['B0']['mean']:.6f}s，wall减少{s['D']['wall_reduction_pct']:.3f}%，12轮中{s['D']['wins']}轮更快。配对节省{s['D']['delta_mean_s']*1000:.3f}ms，95% t区间[{s['D']['ci95_s'][0]*1000:.3f},{s['D']['ci95_s'][1]*1000:.3f}]ms。没有超过PyTorch，也不以此为目标。

S原型已实现真实多张量CUDA更新，诊断中16个参数从32个kernel降到1个，但不能进入生产推荐：当前CuPy14.2默认FTZ，显式不启用FTZ的乘法/减法在subnormal上不同。未开启FTZ、未放宽门限、未加入额外扫描/同步来掩盖问题。SGD(backend="native")安全回退原顺序更新；私有 `_sgd_multi_candidate` 和原型ABI只保留复核用途。

因此表中的S/DS是“请求开关但S被拒绝后的实测”，**不是两项有效优化的完整2×2因子消融**；真正fused SGD生产训练/组合收益为NOT_RUN（数值准入失败）。不会将S的随机差异解释成kernel融合收益，也不把D/S百分比相加。

## 基线与实现贡献

baseline保存进入任务时所有源码/测试、Git状态/diff、动态库和build参数/SHA。确认FP32常量与整数保护、GEMM register tiling及128CTA、Native col2im及128-entry metadata/POD均存在；没有forward im2col reuse。

D修改仅作用Conv2DGradW的GPU FP32前向计算：backward新生成(T,D)列矩阵，T=(n,oh,ow)、D=(ic,kh,kw)，G=gy.transpose(1,0,2,3).reshape(OC,T)，CuPy/cuBLAS执行G@col，结果reshape OIHW。已有hy_im2col复用代码，不是复用前向数值。x必要时ascontiguousarray，gy必要packing、分配和ctypes成本均纳入测量。检查x/gy均为当前GPU FP32、NCHW、匹配几何和int32 ABI支持范围；CPU、FP64或不支持情况回到原路径，无静默gy降精度。无数值缓存、无跨step指针复用。

发现原有Conv2DGradW.backward错误地使用self.outputs里的weakref而不是传入上游梯度。baseline CPU复现TypeError（raw/preexisting_higher_gradient.log）；修复为使用gys。相同修复作用所有配置，经CPU/GPU独立PyTorch高阶公式验证。原有高阶col2im测试照常执行。

S原型：Host24-byte描述符(p,g,n)，每调用私有数组；C入口分组32个，1288-byte launch POD包含uint64 block区间，kernel每个CTA定位tensor并grid-stride处理。一次Python调用，16参数1kernel，超过32按通用分组，无flat数组或device指针表分配。每次读取最新p/g/lr/stream，检查整个集合write/write与write/read overlap，任何依赖就保留旧顺序。hooks仍收集→按顺序执行→再次过滤→update；自定义SGD子类回退，不绕过update_one。编译报告24寄存器、0 stack、0 local、0 shared，无明显spill。

原型使用__fmul_rn和__fsub_rn，CUDA未启用fast-math或FTZ。NVIDIA文档定义为各自round-to-nearest-even：[CUDA12.4.1数学intrinsics](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-math-api/group__CUDA__MATH__INTRINSIC__SINGLE.html)。当前CuPy源码cuda/compiler.py却固定追加-ftz=true（raw/cupy_ftz_source.txt）。独立sgd_edge.json显示subnormal被旧路径flush；prototype测试lr=-0.0的有限位模式3/7不一致，不能把它改成近似比较。禁用S是验收结论，不是宣称实现逐位一致。

## 冻结数值协议与结果

ACCEPTANCE.md在候选运行前写入并保留SHA。原测试输入、参考和断言全部未改。新D门限为abs(B−R)<=2e-5+2e-4abs(R)，只针对声明数据集合。B/A差异非阻断，A/R及旧严格门限完整输出。90组旧集合=30独立shape/seed×3等值layout；78组留出=26独立shape/seed×3layout（2026/31415+新几何）；真实训练step0/95/191×5层=15组。

R用同一FP32输入提升后的独立CPU im2col与FP64矩阵乘法，不使用候选col作参考。最大B/R误差坐标另用CPU long-double标量归约复核，平台精度见raw/longdouble.json。FP64不进入被计时生产运算。

{numtable}

B/R全部通过，不能因此说B每个元素更精确；A与B归约顺序不同。逐位比例仅在同dtype FP32数组上用uint32位视图计算，跨FP32/FP64不伪造位相同比例；另列exact_value_fraction。训练最终逐参数逐元素检查沿用scripts/report_cuda.py的atol1e-4/rtol1e-3；单步/最终梯度为已有Donkey dtype门限2e-5/2e-4。完整loss轨迹全部有限并通过对齐，不只比较hash。

最终训练最大权重误差{max(float(r['max_abs']) for r in errs if r['kind']=='weight'):.9g}，最大梯度误差{max(float(r['max_abs']) for r in errs if r['kind']=='gradient'):.9g}；所有逐参数失败计数0，见TRAINING_ERRORS.csv。单步、补充范围及框架参考误差各有独立CSV。原大GEMM严格1e-4门限的长K/2048/4096失败仍保留历史证据，不称已解决。

## 正确性与安全检查

最终全套测试：272 passed、2 skipped；两项skip是原CPU DonkeyNet测试明确改由GPU覆盖，所有实际GPU项目执行。按集合划分CPU35、GPU237。新增专项35 passed；原型有一次预期数值拒绝记录（1failed/31passed，raw/special_initial.log），保留未删除。memcheck/racecheck对原型及新增调用各28 passed、6 deselected（CPU或已单独记录的special-rounding案例），0 errors/0 hazards。Sanitizer成功不替代数值准入。

D覆盖正/负/非单位stride的x/gy、非方形/非对称参数、非默认stream、FP64回退、重复backward、forward后原地改输入时与原路径同样读取当前内容、no_grad与二阶导数。SGD原型覆盖0/1/16/32/33/65tensor、空tensor、重分配梯度、lr变化、hooks/冻结/子类、重复存储、overlap与跨参数梯度别名、CPU/FP64/不支持布局回退。并非生产S已通过特殊浮点测试；这一失败正是禁用原因。

## 正式训练：同进程12轮平衡顺序

固定256合成样本、batch4、3epochs/192steps、seed42、dropout0、lr.001；所有四组forward/GEMM/col2im相同。每组预热10步后恢复权重/梯度/RNG；raw/order.json预先保存Williams平衡顺序。wall包含DataLoader/H2D/forward/backward/update和末尾sync；计时中不读取loss，保留GPU标量引用并在计时后统一读取。四组相同NVTX标注及日志行为，无profiler附加。没有删除慢轮。

{table_train}

PAIRED.csv逐轮差值、LOSS_TRAJECTORIES.csv完整轨迹、TRAINING_ERRORS.csv逐参数误差均在包中。95%区间来自12个配对差值、df11 t=2.200985，而非两组独立SD；样本相关性与单机单窗口限制仍存在。

初轮4.36%是补充int32几何范围检查前的结果，完整封存pre_range_guard，不用于最终推荐。因新检查改变CPU路径，使用相同预定顺序重跑受影响wrapper/训练/补充/profile；这里报告最终重测，未从两轮择优。另有换行清理的AST一致性记录；原始数值normal集合的数学路径不变。

## 算子成本与实测归因

FULL_WRAPPERS.csv记录完整Conv2DGradW Function构造、输入包装、native检查、packing、分配、cuBLAS和返回Variable，采用与一阶backward相同的no_grad。每case15次交替、预热10次，13几何×3layout，包含全部39case。以下为真实五层连续输入中位数：

{wr}

同步完整调用D退化case数={len(failedops)}/39，全部数据保留；小样本不代表任意几何保证。OPERATORS.csv是较内层计算路径和CUDA event envelope诊断，不能替代完整wrapper；event envelope包含可能的主机提交空隙。真正kernel时间来自CUPTI，OPERATOR_PROFILE.csv逐kernel列出。SGD prototype完整Optimizer.update计时含参数收集/hooks/filter/alias/descriptor，但明确REJECTED，不用其加速比宣称训练成功。

{profiletable}

两次交替短trace均确认D使dWeight copy kernel从10降至5，总dWeight kernel24→19，全步147→142。backward im2col仍5次，加forward5次总10次，GEMM及split-K reduction数保持9；cuBLAS选择由nt变为nn，具体内核名见PROFILE.csv。D省掉列矩阵重排copy，gy packing仍存在，非连续输入还需x packing。未改变GEMM原生kernel、forward、col2im或dInput算法。

SGD原型独立trace16参数32→1kernel；实际训练所有四组SGD仍32。S的完整调用可能显著更快，但特殊数值不兼容已阻断生产路径。dWeight CUPTI约从B0的两次{float(pr[0]['dweight_us_per_step']):.1f}及{float(next(r['dweight_us_per_step'] for r in pr if r['mode']=='B0' and r['repeat']=='1')):.1f}μs/step降至D对应值，具体数值见表，不能把全部波动归因D。未改forward/dInput阶段也有时钟/调度波动；2个短trace只用于计数和解释，不能替代12轮wall。

PROFILE.csv同时列CPU NVTX elapsed、CUDA API elapsed、GPU时长，可能互相重叠，禁止相加。CPU elapsed不是精确Python主动执行时间。没有锁频，GPU0服务未触碰；主机共享环境的噪声没有从样本中剔除。

## 显存与适用范围

{memtable}

CuPy MemoryHook在每次malloc_postprocess读取pool.used_bytes/total_bytes，得到该allocator的峰值；不是低频nvidia-smi采样，也不涵盖cuBLAS/CUDA/Torch私有分配。每配置batch各3轮完整训练，cleargrads/gc/sync后活跃分配稳定；保留量不是泄漏。batch4每步少5次pool请求（完整训练减少960次），batch16同样；batch1本测量分配次数和峰值并未降低，必须保留这一边界。没有使用4D-1的理论11.565MiB充当本轮测量。

{supptable}

补充各3轮，只用于范围检查，不当作与12轮主实验同等置信证据。TinyNet少量样本可能受主机噪声影响；所有补充逐参数误差通过。推荐限本轮声明网络/几何与FP32语义，不宣称全形状/真实驾驶任务精度。

## 同窗口框架参考

{reftable}

PyTorch2.9.1+cu128、cuDNN enabled、benchmark默认False；matmul和cuDNN TF32关闭，SGD foreach=None/fused=None，额外实际分支检查观察到_multi_tensor_sgd，未关闭其正常foreach优化。PyTorch和CuPy各5轮是独立参考组，不与D的12轮形成严格配对显著性结论；它们在边界检查修正前同一GPU窗口测量，修正不影响其CuPy/torch代码路径。库算法、cuDNN卷积/自动求导/optimizer实现不同。PyTorch显著更快，不制造反超结论。HyTorch四组有相同轻量NVTX标注，Torch参考无这些Python包装，属于框架诊断差异，不能把全部差异归为单个kernel。

## 交付与未运行项

source最终源码、baseline原源码/库、git_diff.patch本轮增量、SHA.json、ENVIRONMENT.json、命令日志、原始测试/计时/错误数据、Nsight-rep/SQLite及解析脚本全部提供。CLAIMS.md映射每条PPT结论，PPT_SUMMARY.md六页结构，figures含SVG和PNG（零起点/明确误差条）。

NOT_RUN：真正启用S的正式训练/组合（准入失败）；跨窗口累计早期基线收益（未恢复重测）；真实DonkeyCar/Udacity任务精度和泛化；全形状/全数值范围保证；实际跨物理GPU混合设备集成（未碰GPU0）。不使用历史不同窗口百分比填这些空白。

已释放GPU，不提交、推送、创建PR，不进入下一轮优化。
''')
(O/'SOURCE_CHANGES.md').write_text('''# 本轮源码变更

- runtime.py：backend新增dweight=cupy/packed，默认cupy；packed_dweight按当前x/gy新生成列矩阵，FP32/device/NCHW/shape/int32范围检查及fallback；可选ABI hy_sgd_multi；私有SGD描述符/别名检查原型。公开sgd_multi返回False，因为FTZ精度门限拒绝。
- functions_conv.py：仅Conv2DGradW.forward增加packed选择与回退；backward修复为使用输入gys，上游梯度公式同时适用于B0/D/S/DS。
- optimizers.py：仅SGD新增backend=legacy/native，默认legacy。精确SGD类型才尝试批量路径；收集/hooks/再次过滤顺序保留，当前native回退原路径。其他optimizer不改。
- kernels.cu：只在原文件后追加SgdTensor/SgdBatch/hy_sgd_multi/sgd_multi_kernel；原有CUDA源内容逐字节为前缀，GEMM/col2im等未改。原ABI版本与入口不变。
- tests/test_final_optimization.py：35项新增专项；包含真实higher derivative、layout、fallback和SGD语义。SGD prototype最初失败测试副本保留raw/prototype_tests.py，生产fallback最终通过不掩盖原型拒绝。

没有forward/backward数值workspace缓存、requires_grad、version counter、完整训练CUDA Graph、混合精度、TF32、fast-math或FTZ开关。dWeight GEMM依然来自CuPy/cuBLAS，不是新手写GEMM。SGD原型是新手写多张量kernel，但未通过全部数值准入。

git_diff.patch只含本轮变更，baseline/git_diff.patch包含进入任务时已存在的修改。raw/newline_cleanup.json记录CRLF清理的AST一致性；几何范围保护后已重测受影响性能，旧结果封存pre_range_guard。
''')
claims=[('C1','D使本固定训练wall减少%.3f%%，%d/12胜出'%(s['D']['wall_reduction_pct'],s['D']['wins']),'实测','TRAINING.csv; PAIRED.csv; raw/training_summary.json','限A100/该合成配置，非全workload'),('C2','D每步少5个列布局copy，147→142kernel；im2col仍10','实测','PROFILE.csv; PROFILE_SUMMARY.csv; traces/train_*.sqlite','不称前向im2col复用，不称消除backward展开'),('C3','D通过168固定/留出+15真实梯度FP64准入及完整训练对齐','实测','NUMERICS.csv; TRAINING_DWEIGHT_NUMERICS.csv; TRAINING_ERRORS.csv','不保证每个元素更准确或全数值范围'),('C4','batch4池活跃峰值下降，连续3轮未持续增长','实测','MEMORY.csv','仅CuPy allocator，不是整设备所有库精确峰值'),('C5','SGD原型16tensor从32kernel降至1，但因FTZ/subnormal不一致拒绝','实测且未通过验收','OPERATOR_PROFILE.csv; raw/special_initial.log; raw/cupy_ftz_source.txt','不能用于生产训练加速声明'),('C6','D的wall收益与减少packing/提交工作一致','推断，机制由计数支持','FULL_WRAPPERS.csv; PROFILE.csv; TRAINING.csv','不能把CPU/GPU重叠时间相加定量分解'),('C7','真实驾驶任务精度、全形状泛化、早期累计收益','未验证/NOT_RUN','REPORT.md','不得用历史报告补成新实测')]
(O/'CLAIMS.md').write_text('# 可用于汇报的结论与证据\n\n'+table(['ID','结论','性质','证据','边界'],claims))
(O/'PPT_SUMMARY.md').write_text(f'''# 六页PPT材料（所有图从本轮CSV生成）

## 1. 问题：单算子改善不等于完整训练收益
- 已有FP32/GEMM/Col2im优化作为当前可靠基线，不累计历史百分比。
- 本轮同时评估dWeight布局与多张量SGD，候选必须同时满足数值与端到端条件。
- 用图：figures/data_path.svg；历史性能仅作为背景，不与当前柱图相加。

## 2. 分析：布局与启动成本
- trace显示原dWeight每step5次im2col、10次copy、9次GEMM/归约。
- 原SGD16参数32kernel/step；直接做批量更新有潜力但仍须数值验收。
- 用图：figures/dweight_kernels.svg、figures/operator_wrappers.svg；证据PROFILE.csv/FULL_WRAPPERS.csv。

## 3. 实现：accepted D / rejected S
- D：fresh native(T,D)→gy packing→CuPy/cuBLAS→OIHW，不缓存forward数值。
- S原型：32描述符一组、当前指针/stream、别名回退、显式两次舍入，24regs/0local。
- S因CuPy默认FTZ与禁止FTZ的原型在subnormal上不一致而禁用；图中需红色注明“原型未通过”。
- 用图：figures/data_path.svg。明确库贡献边界。

## 4. 正确性
- 90旧+78留出+15真实梯度，B/R全部通过；B/A差异完整披露。
- 原有高阶weakref错误已单独修复，所有配置共同使用。
- 最终272passed+2个原CPU主动skip；Sanitizer各28passed/0error。S原型失败记录不能隐藏。
- 用图：figures/numerics.svg；表引用CORRECTNESS/REPORT及原始日志。

## 5. 性能
- B0 {s['B0']['mean']:.6f}s → D {s['D']['mean']:.6f}s，{s['D']['wall_reduction_pct']:.3f}%减少。
- 配对节省{s['D']['delta_mean_s']*1000:.3f}ms，95%CI[{s['D']['ci95_s'][0]*1000:.3f},{s['D']['ci95_s'][1]*1000:.3f}]ms。
- 四个请求配置均展示，但S*=fallback、DS*=D+fallback，不能称真正两项优化组合收益。
- 用图：training.svg、paired.svg、frameworks.svg。主图零起点，SD与CI标明；同窗口PyTorch仍更快。

## 6. 边界
- batch1/16及TinyNet各3轮范围检查，不夸大为普遍结论。
- 内存仅allocator峰值；batch1没有同等内存下降，连续轮数有限。
- 历史严格GEMM失败仍在；真实驾驶数据准确率NOT_RUN；S生产加速NOT_RUN。
- 用图：memory.svg；推荐只启用D（其余backend按报告显式设置），SGD保持legacy。

可直接引用C1–C5，但C5必须同时附“数值未通过，未启用”。所有SVG可编辑、PNG可直接放PPT，绘图脚本diagnostics/plots.py。
''')
(O/'CORRECTNESS.md').write_text('''# 正确性验收汇总

- 最终完整回归272 passed、2 skipped；CPU35、GPU237，skip是原CPU DonkeyNet两项显式由GPU覆盖。raw/full_tests_guard.log列明原因。
- 最终新增专项35 passed（raw/guard_tests.log）。
- 新原生调用/SGD原型memcheck、racecheck各28 passed、6 deselected、0错误/0hazards。原型special rounding不因sanitizer通过而获得数值准入。
- 初次SGD原型1 failed、31 passed：subnormal逐位差异，保留raw/special_initial.log。生产sgd_multi现返回False，禁用候选并继续D。
- D168组固定/留出+15组真实训练B/FP64参考均通过；A/R、B/A和其他严格门限全部在NUMERICS.csv；失败坐标未删除。
- 单步、完整192步、batch1/16、TinyNet、CuPy和Torch参考的逐参数及loss轨迹对齐全部通过。参考不同框架不是逐位相等要求。
- 新旧D相对误差按冻结混合门限；位相等与数值相等分列，跨精度位相等不适用。
- 原Conv2DGradW二阶导数错误单独CPU复现并修复；默认/packed两组在CPU/GPU与PyTorch独立公式比较通过。
- 历史大GEMM严格失败未解决；原有测试源码、门限SHA逐项保持不变。
- int32几何保护后补跑35专项+完整回归，性能完整重测。原正常数值集合算术与kernel未变。
''')
print('reports written')
