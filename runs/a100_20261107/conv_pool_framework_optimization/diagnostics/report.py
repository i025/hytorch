from pathlib import Path
import csv,json,statistics as st,hashlib,difflib,shutil,subprocess
O=Path(__file__).resolve().parents[1];R=O.parents[2]
def read(n):return list(csv.DictReader((O/n).open()))
def md(name,text):(O/name).write_text(text.strip()+'\n')
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(str(v) for v in row)+' |' for row in rows])
s=json.loads((O/'SUMMARY.json').read_text());ls=read('LAYER_COMPARISON.csv');idx={(r['case'],r['implementation']):r for r in ls};profiles=read('PROFILE_SUMMARY.csv');mem=read('MEMORY_COMPARISON.csv');pair=s['paired']
def v(case,mode,metric):return float(idx[case,mode][metric])
def localtable(names):
 return table(['shape / case','GPU B0/O1/Torch (μs)','同步完成 B0/O1/Torch (μs)','CPU提交 B0/O1 (μs)','B0/O1 kernel数'],[[n,' / '.join(f"{v(n,m,'gpu_kernel_us_median'):.3f}" for m in ['B0','O1','torch']),' / '.join(f"{v(n,m,'sync_us_median'):.3f}" for m in ['B0','O1','torch']),' / '.join(f"{v(n,m,'submit_us_median'):.3f}" for m in ['B0','O1']),' / '.join(str(int(v(n,m,'kernels_per_call'))) for m in ['B0','O1'])] for n in names])
train=table(['配置','mean(s)','median(s)','SD(s)','平均 samples/s'],[[m,f"{s[m]['mean_s']:.6f}",f"{s[m]['median_s']:.6f}",f"{s[m]['sd_s']:.6f}",f"{s[m]['samples_per_s']:.2f}"] for m in ['B0','O1','torch']])
pt=table(['配置','阶段','GPU μs/step','CPU range μs/step','CUDA API μs/step','kernels/step'],[[r['mode'],r['phase'],f"{float(r['gpu_kernel_us_per_step']):.2f}",f"{float(r['cpu_range_elapsed_us_per_step']):.2f}" if r.get('cpu_range_elapsed_us_per_step') else '—',f"{float(r['cuda_api_elapsed_us_per_step']):.2f}",r['kernels_per_step']] for r in profiles if r['phase'] in ['forward','backward','update','ALL']])
mt=[]
for b in ['1','4','16']:
 for m in ['B0','O1','torch']:
  r=[r for r in mem if r['batch']==b and r['mode']==m]
  mt.append([b,m,round(st.mean(float(x['peak_active_bytes']) for x in r)/2**20,3),int(st.mean(float(x['malloc_calls']) for x in r)),round(st.mean(float(x['after_clear_bytes']) for x in r)/2**20,3)])
mt=table(['batch','配置','峰值活跃 MiB','分配请求/完整测量','结束活跃 MiB'],mt)
lay=localtable([f'conv{i}/backward' for i in range(1,6)]+['irregular/backward','asymmetric/backward'])
pool=localtable([n+'/forward' for n in ['tiny_pool','irregular_pool','overlap_pool','wide_pool','tiny_actual_pool']])
notes='''GPU 时间来自 Nsight/CUPTI 的实际 kernel activity，15 次/shape/配置；完整同步调用和 CPU 提交来自无 profiler 的独立计时，18 次平衡排列。二者不是同一调用，不能相减解释为“等待时间”，也不能将 CPU、API、GPU 时间相加。submit 包含 Python/C 执行、CUDA API 与可能的队列等待，不等同于纯 Python 主动执行。cProfile 单独采集，self/cumulative elapsed 也不是 CPU 指令周期。'''
md('REPORT.md',f'''
# Conv2D Backward / Pooling 最终框架优化报告

## 结论

保留一个默认关闭的 O1：`Config.compact_function` 控制 `Function.__call__` 的单输出包装快路径。仅新增 11 行生产代码，不更改数学运算、CUDA kernel、GEMM dispatch、Col2im、dWeight 或 SGD。所有正常图连接和必要检查仍执行。

在已经开启 F1 的同进程 12 轮正式训练中，B0 **{s['B0']['mean_s']:.6f}s → O1 {s['O1']['mean_s']:.6f}s**，wall time 减少 **{pair['wall_reduction_pct']:.3f}%**，速度比 **{pair['speedup']:.4f}×**，12/12 轮胜出。每对节省均值 **{pair['delta_mean_s']*1000:.3f}ms**，Student-t 95% CI **[{pair['ci95_s'][0]*1000:.3f}, {pair['ci95_s'][1]*1000:.3f}]ms**。loss、全部参数梯度和最终权重逐位一致。

HyTorch/PyTorch 训练耗时比由 **{pair['before_vs_torch']:.4f}× → {pair['after_vs_torch']:.4f}×**。绝对差距从 {(s['B0']['mean_s']-s['torch']['mean_s'])*1000:.3f}ms 降到 {(s['O1']['mean_s']-s['torch']['mean_s'])*1000:.3f}ms，差距减少 {100*pair['delta_mean_s']/(s['B0']['mean_s']-s['torch']['mean_s']):.2f}%。这不代表已经接近 PyTorch 端到端水平。

## 实验范围、资源与基线

- A100-PCIE-40GB，物理 GPU1，UUID `GPU-0bc09ee6-967a-b53d-bf80-38379585cf08`，`CUDA_VISIBLE_DEVICES=1`。每个任务启动前查询资源，无他人进程才执行；不终止进程、不改 GPU 时钟。
- 主实验窗口：2026-11-07 16:09–16:22 UTC。实际 TinyNet Pooling 补测：2026-11-08 07:39 UTC。会话跨日，因此**不是所有数据来自单一连续窗口**；补测明确独立列出，没有用它与主窗口绝对时间相减。主训练 12 轮均在一个进程与连续窗口内完成。
- 补测也重新检查 GPU1 空闲。最终 `raw/RESOURCE_RELEASED.json` 为 GPU1 0 MiB、0% 且无计算进程。资源检查是瞬时检查，不是排他预留。
- `baseline/` 保存本轮前 Git状态/HEAD/完整已有diff、83份源码/构建文件及SHA，包括实际动态库；既有 FP32 修复、128 CTA GEMM、Native Col2im POD/cache、packed dWeight 均保留。
- B0 显式 `metadata_cache=True`，不把 F1 换名再实现。O1 唯一变量为新 Function 开关。Legacy SGD，默认后端设置不变。
- Torch {json.loads((O/'ENVIRONMENT.json').read_text())['torch']}，CuPy 14.2.0，TF32 matmul/cuDNN均关闭，cuDNN正常启用且benchmark=False，torch.compile关闭，SGD foreach=None/fused=None（保留框架默认优化）。不启用fast-math。

## 实际瓶颈及证据

### Conv2D Full Backward

路径是 `Variable.backward → Conv2d.backward → Deconv2d / Conv2DGradW / Sum`。dInput仍用 CuPy tensordot、rollaxis、Native Col2im；dWeight仍生成 fresh im2col，并执行 gy transpose/reshape packing 和 CuPy GEMM；dBias仍单独reduce。完整图还包含输入/输出 Variable 包装、弱引用、拓扑排序和梯度累加。

30步真实训练 cProfile 记录 `Function.__call__` **1980次（66/step）**，优化前后次数不变；`np.isscalar` 从 **2836 → 856次**，减少1980次，保留必要的标量转换。Function self elapsed 8.259→7.601ms，isscalar self elapsed 2.335→0.807ms；这些是独立 profiler 数据，不作为正式加速率。额外输出tuple/list/listcomp/单元素循环在快路径中消除，Variable和图边没有消除。

conv4 代表性分层 CPU提交（B0，无profiler中位数）：预验证直调 {v('conv4/direct','B0','submit_us_median'):.2f}μs；带安全检查原始计算 {v('conv4/raw','B0','submit_us_median'):.2f}μs；Function反向子调用 {v('conv4/function','B0','submit_us_median'):.2f}μs；完整autograd {v('conv4/backward','B0','submit_us_median'):.2f}μs。各层均计算x/W/b三类梯度，GPU数学路径不变。差值是诊断上限的线索，不能宣称全部可安全删除。

conv4 Full Backward HyTorch实际8个kernel、7个CuPy分配请求、{int(v('conv4/backward','B0','requested_bytes')):,} bytes请求；Torch4个kernel、3个分配请求、{int(v('conv4/backward','torch','requested_bytes')):,} bytes。GPU kernel相近不代表工作分解/packing/内存行为相同。诊断cProfile里 reshape 的elapsed还可能含GPU背压，不能全算作Python解释器成本。

### Pool Forward / Backward

DonkeyNet没有Pooling，不把Pooling局部收益算入它的训练收益。TinyNet真实池化位于3→4通道卷积和ReLU之后，输入是 **(4,4,16,16)**，kernel2、stride2、pad0；`WORKLOAD_TINY_POOL.json`为运行时证据。历史探针名字`tiny_pool`使用(4,3,16,16)，本报告把它列为**通用3通道形状**，不冒充真实TinyNet。

真实4通道Pool：预验证且预分配的Native直调 CPU提交 {v('tiny_actual_pool/direct','B0','submit_us_median'):.2f}μs；正常runtime含校验/两次分配 {v('tiny_actual_pool/raw','B0','submit_us_median'):.2f}μs；完整Function {v('tiny_actual_pool/forward','B0','submit_us_median'):.2f}μs；Torch {v('tiny_actual_pool/forward','torch','submit_us_median'):.2f}μs。GPU kernel均约5μs。直调跳过检查并预分配，只是性能下限，不能直接拿它替换安全生产接口。

逐项微计时见 `CPU_OVERHEAD.csv`：backend selection、_arrays、_geometry（已启用F1）、stream查询、指针封装、输出与indexes分配。参数校验、两次数组分配、ctypes和Function包装共同构成完整调用，不能把全部差距都归为ctypes。Pooling backward还需要scatter/索引/col2im，真实4通道HyTorch4 kernels、Torch2 kernels，O1不改变这些工作。

## 修改选择与安全性

O1 对非tuple单输出直接构建一个Variable；已知ndarray跳过np.isscalar，其他标量/None仍走原as_array；同样设置generation、creator、inputs和weakref outputs。tuple返回（含一个元素与多输出）原路径不变。no_grad原有inputs/outputs保留语义不变。

未修改 `Variable.backward` 的遍历、排序、梯度累加顺序，也未绕过Function.forward。所有dtype/device/buffer/alias检查原封不动。无任何GPU地址、stream、数值workspace缓存；Config沿用原项目using_config的作用域规则。

接受O1作为可选开关，默认False。**不实现O2**：虽然Pooling binding有额外成本，但其不在DonkeyNet中，省略安全检查的诊断下限不足以证明安全POD重构可获得同等收益；新增ABI/workspace复用会增加风险。既有Fused SGD拒绝策略不变。本轮没有测试第二个生产候选，也不把“未实施”描述为“已实验证明无效”。

## 局部性能

{notes}

### 完整卷积反向（x/W/b梯度集合相同）

{lay}

五层真实卷积跨shape中位数：CPU提交降低 **{s['local']['conv_real']['submit_us_median']['reduction_pct_median']:.2f}%**，同步完成降低 **{s['local']['conv_real']['sync_us_median']['reduction_pct_median']:.2f}%**；完整调用H/P中位数 **{s['local']['conv_real']['sync_us_median']['B0_over_torch_median']:.3f}× → {s['local']['conv_real']['sync_us_median']['O1_over_torch_median']:.3f}×**；GPU时间比约0.908×，几乎不变。

局部置信区间见 `LOCAL_PAIRED.csv`。conv2、conv4及irregular完整同步调用的CI包含零，不能宣称每个shape都稳定加速。直接调用路径不使用O1，但也出现测量波动，进一步说明必须使用正式训练配对结果。

conv1的PyTorch全梯度局部测量包括输入梯度，cuDNN在此形状较慢。真实训练PyTorch首层输入不requires_grad；HyTorch现有引擎仍计算它。本轮不改变梯度需求，因此不能把上述各层局部时间直接相加预测训练时间。

### Pooling前向

{pool}

主窗口4个通用shape同步调用中位数降低 **{s['local']['pool_main_window']['sync_us_median']['reduction_pct_median']:.2f}%**；真实TinyNet补测为 **{s['local']['pool_actual_supplement']['sync_us_median']['reduction_pct_median']:.2f}%**，但它的均值配对节省95%CI包含零（−0.347～3.102μs），仅作观察，不能称为稳定前向加速。其CPU提交中位数降低 {s['local']['pool_actual_supplement']['submit_us_median']['reduction_pct_median']:.2f}%。5个shape和全部反向结果都保留在CSV，没有只展示有利数据。

## 正式端到端验收

256合成样本、batch4、3epochs/192steps、seed42、dropout0、SGD lr0.001。12轮，三配置6种排列各执行两遍；同一进程重置相同权重、数据和随机数。包含DataLoader、H2D、forward、backward、SGD及最终同步；初始化/预热和D2H结果检查在时钟外。没有逐步额外同步或profiler。报告吞吐为每轮768/wall的均值。

{train}

`PAIRED.csv`保存每轮差值，`LOSS.csv`保留全部step的loss。95% CI基于12个配对差值、df=11，非独立组标准差比较。新旧最终loss均为0.00025566318072378635，Torch为0.0002556628314778209；全部误差见TRAINING_ERRORS。没有将上一轮F1的百分比与本轮相加。

补充（同主窗口，各3对，仅少量验证）：

{table(['模型','batch','B0(s)','O1(s)','wall降低','胜出'],[[x['model'],x['batch'],f"{x['B0_s']:.6f}",f"{x['O1_s']:.6f}",f"{x['reduction_pct']:.2f}%",f"{x['wins']}/{x['n']}"] for x in s['supplement']])}

batch1为768steps，batch16为48steps，均处理768个样本；TinyNet为192steps。这些不混入12轮主验收CI。

## 时间线与因果范围

以下采用主窗口b/c各10step captures均值；stage_a trace单独保留但不用于该均值。

{pt}

O1与B0的kernel数均142/step、H2D均2次且633,616 bytes/step；GPU累计时间约1.32ms/step，基本相同。72 backward kernels/step未变。两次新旧capture中完整backward CPU range从约2.78ms降至2.62ms；与正式wall收益方向一致。此处profiler会扰动耗时，不能替代正式训练结果。

GPU kernel时间线空隙（span减kernel并集）B0 {next(float(r['kernel_span_gaps_us_per_step']) for r in profiles if r['mode']=='B0' and r['phase']=='ALL'):.2f}μs/step、O1 {next(float(r['kernel_span_gaps_us_per_step']) for r in profiles if r['mode']=='O1' and r['phase']=='ALL'):.2f}μs/step。这里包含H2D/同步/CPU提交等，是**没有本进程kernel覆盖的时间**，不是设备所有引擎的严格空闲率。Torch kernel存在并发，sum与union也分别记录，不能相加。

CUDA API具体名称、次数、耗时以及launch/sync分项在PROFILE_BEFORE_AFTER与LAYER_COMPARISON内；Python profiling与CUDA API elapsed独立报告。没有通过wall−kernel推导纯Python主动执行时间。

## 内存

{mt}

B0/O1各batch分配请求数、峰值活跃显存及结束活跃显存一致，连续3次没有增长。仅Python容器减少，GPU数组分配和生命周期不变。表内GPU活跃显存是各框架allocator统计，不是nvidia-smi整进程占用；Torch含其库workspace。测量覆盖重置参数、训练及结果采集，不能直接把分配次数/192当训练每步次数。单算子分配次数/请求字节独立在LAYER_COMPARISON中。请求字节是累计分配量，不是唯一临时显存峰值。

## 正确性与限制

- 新增28项专项测试全通过；完整套件319 passed、2 skipped（明确为CPU完整DonkeyNet，GPU版本实际运行）。开关开启后既有291项也全部通过，2个相同CPU skip不计passed。
- memcheck：0 errors；racecheck：0 hazards/0 errors/0 warnings。没有新CUDA算法，仍对经过新包装路径的GPU操作做验证。
- B0/O1所有局部输出和主/补充训练loss、梯度、权重逐位一致；与Torch按原门限对齐，训练最大绝对误差 {s['TRAINING_ERRORS.csv']['max_abs']:.3g}。不修改任何既有门限；历史GEMM严格门限记录原样保留。
- 本次只验证指定硬件/工作负载窗口；局部微秒级差值不全部显著，主训练12轮CI为正。默认仍关闭，用户可显式启用。
- 主实验与补测跨日是执行窗口限制，真实Pool形状单独报告；不假称全程单窗口。

## PPT最值得展示的三条结论

1. **只增11行生产代码，训练wall减少4.20%，12/12胜出，梯度/权重/loss逐位一致。**
2. **142 kernels/step完全不变，GPU累计时间近似不变；消除单输出冗余包装能减少主机提交间隙。**
3. **训练H/P由2.434×缩小到2.332×，但仍有明显差距；局部kernel接近不等于完整框架接近。**

`figures/`含5套PNG+SVG，图注明计时范围、GPU、batch和样本数；实际Pool补测单独标注。本轮完成后停止，没有继续优化其他算子。
''')
md('CALL_PATH.md','''
# 实际调用路径与计时边界

## Conv2D

`core.Variable.backward` → 读取weak output.grad、按generation排序 → `Conv2d.backward` →

- dInput：`Deconv2d.__call__ → forward → CuPy tensordot(W,gy) → rollaxis → functions_conv.col2im_array → runtime.col2im → dtype/current-device/strides/shape/alias检查 → bounded immutable _col2im_metadata/POD → _call → fresh current stream → hy_col2im_fast → col2im_strided`。
- dWeight：`Conv2DGradW.__call__ → runtime.packed_dweight → dtype/device/geometry检查（F1 shape cache开启）→ contiguous input → fresh col → hy_im2col → gy.transpose/reshape packing → CuPy/cuBLAS GEMM → reshape`。
- dBias：`Variable.sum → Sum.__call__ → CuPy sum`。
- 包装各梯度Variable → 原有gx+old_grad累加 → 按原序调度creator → 按retain_grad清理输出梯度。

生产路径没有直接调用诊断接口。所有检查、分配、packing、GEMM、launch及stream查询都与B0相同。新增快路径只在Function.forward返回之后改变临时Python容器。

## Pooling

`functions.pooling → functions_conv.Pooling.__init__ → Function.__call__ → Pooling.forward → selected('pool',x) → runtime.max_pool2d → _arrays → _geometry → empty(y,float32) + empty(indexes,int64) → _ptr / _call / ctypes → current stream → hy_pool → pool_kernel → 保存indexes → output Variable`。

Backward：`Pooling2DGrad` 保存输入shape/dtype及indexes → zeros(gcol) → arange/索引偏移 → scatter gy → reshape/swapaxes → col2im。高阶使用Pooling2DWithIndexes，不改变max winner状态。

原生池化padding为零。局部对比输入用严格正值，确保Torch(-inf padding)选中相同winner；专项测试另外检查负值/零padding/边界和重复反向的新旧一致性。TinyNet输入在池化前经ReLU。

## 五层测量

| 层级 | 本轮实现与范围 |
|---|---|
| L1 | Nsight CUDA/NVTX/CUPTI；只加总实际kernel execution，15重复；另列kernel并集和数量 |
| L2 | conv direct：预先验证静态metadata，直接调用已有im2col/col2im C ABI，仍包括fresh临时数组和packing；pool direct另外预分配输出。只在diagnostics脚本中暴露。Torch conv direct使用aten.convolution_backward同样请求x/W/b |
| L2 checked | 原有底层runtime/Function.forward，仍执行安全检查，不创建Variable |
| L3 | conv.backward内构造并执行Deconv2d/GradW/Sum三个Function及包装结果，no_grad语义同普通backward；pool新建完整Pooling Function并保存索引。没有同一抽象边界的Torch项留空，不虚构逐层对应 |
| L4 | HyTorch真实Variable.backward遍历+梯度赋值/累加；Torch autograd.grad保留图，同样求x/W/b。重复使用图避免把forward算入backward，清梯度在计时内 |
| L5 | 无profiler完整训练，从DataLoader开始，包含H2D、forward/backward/SGD和最终sync；预热/初始化/结果D2H不在计时内 |

L3 dInput/dWeight/dBias独立Function诊断允许创建图，L3 backward_Functions则模拟默认backward的no_grad状态，两者不是可相加的组件。正式训练没有额外逐层同步。局部CPU提交与同步完成分别采样，两条独立样本不能相减。低开销component微计时为15批×1000调用，cProfile为另一次运行。

工作负载证据：WORKLOAD_COMPLETE.json（DonkeyNet逐Function shape/dtype/strides及GEMM）；WORKLOAD_TINY_POOL.json（TinyNet真实Pooling为4通道）。CSV中的tiny_pool是历史3通道通用探针，真实项为tiny_actual_pool。
''')
md('CORRECTNESS.md',f'''
# 正确性验收

| 阶段 | 实际结果 | 原始日志 |
|---|---|---|
| 修改前完整host+GPU | 291 passed, 2 skipped | raw/tests.log |
| 新增专项 | 28 passed | raw/special.log |
| 修改后完整host+GPU | 319 passed, 2 skipped | raw/tests_after.log |
| O1启用后的全部既有测试 | 291 passed, 2 skipped | raw/tests_on.log |
| Compute Sanitizer memcheck | 28 passed, 0 errors | raw/memcheck.log |
| Compute Sanitizer racecheck | 28 passed, 0 hazards/errors/warnings | raw/racecheck.log |

两个skip均为test_backward_dtype里CPU完整DonkeyNet，GPU版实际执行；skipped未计为通过。O1专项在每项内分别切换B0/O1；tests_on对原有整套测试全局开启O1，单独排除依赖默认False的专项文件（它已单独全部运行）。

新增测试覆盖CPU/GPU×FP32/FP64，batch1/3/2/1连续变化，NCHW/转置/负/零stride，非方形conv/stride/padding，梯度累积/重复backward/create_graph/二阶，输入在建图后原地修改、no_grad、更新后参数；pool负输入/边界padding/重叠窗口/高阶；非默认stream、弱引用生命周期、原alias/dtype拒绝；自定义Function的ndarray/0d/scalar/None/tuple/list异常返回契约和配置异常恢复。没有改动既有测试门限。

测试bitwise以dtype/shape/tobytes比较，包含符号零等；数学路径相同，误差应为0。局部18轮数据的B0/O1、主训练12×192步的loss/16参数及其梯度、补充训练全部逐位相等。Torch局部和训练所有已有atol/rtol门限均通过；训练最大绝对差{ s['TRAINING_ERRORS.csv']['max_abs']:.9g}。完整误差和逐步loss均保留CSV。

原生kernel/ABI未改变；GPU压力与安全测试仍确实在A100执行。之前GEMM严格门限失败记录复制到raw/previous_gemm_*，只是历史证据，不改门限、不把旧失败改记为本轮通过。

默认Config为class级配置，沿用原项目using_config机制；本轮没有扩展为线程局部配置，也没有新增跨线程共享GPU缓存。
''')
# Source snapshots and incremental patch: existing earlier work is excluded.
final=O/'source';shutil.copytree(O/'baseline/source',final,dirs_exist_ok=True)
for name in ['hytorch/core.py','tests/test_compact_function.py']:
 p=final/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(R/name,p)
patch=[]
for name in ['hytorch/core.py','tests/test_compact_function.py']:
 old=O/'baseline/source'/name;new=R/name
 before=old.read_bytes().splitlines(keepends=True) if old.exists() else []
 patch.extend(difflib.diff_bytes(difflib.unified_diff,before,new.read_bytes().splitlines(keepends=True),fromfile=('a/'+name if old.exists() else '/dev/null').encode(),tofile=('b/'+name).encode()))
(O/'git_diff.patch').write_bytes(b''.join(patch))
checks=[]
for row in json.loads((O/'baseline/SHA.json').read_text()):
 name=row['path'];cur=R/name
 checks.append(dict(path=name,baseline_sha256=row['sha256'],final_sha256=hashlib.sha256(cur.read_bytes()).hexdigest(),unchanged=hashlib.sha256(cur.read_bytes()).hexdigest()==row['sha256']))
(O/'SOURCE_INTEGRITY.json').write_text(json.dumps(checks,indent=2))
assert [x['path'] for x in checks if not x['unchanged']]==['hytorch/core.py']
md('SOURCE_CHANGES.md','''
# 本阶段源码变化

## 生产：hytorch/core.py（新增11行）

- Config：增加compact_function=False。
- Function.__call__：仅非tuple单输出，减少临时tuple、outputs list、listcomp和单元素循环；已知ndarray直接传给Variable，跳过np.isscalar。
- 保留Variable验证、generation、set_creator、inputs强引用、outputs弱引用；无梯度模式依旧保存原有Function状态。
- 未改Variable.backward、计算算法、autograd排序/累加顺序、原返回类型、异常行为、dtype/设备/alias检查。

开关：

```python
from hytorch.core import using_config
with using_config('compact_function', True):
    # forward / backward / SGD，正常用法
    ...
```

默认关闭，独立于runtime.backend(metadata_cache=True)。本轮B0/O1都开启已有F1，新开关不是F1的新名字。

## 专项测试：tests/test_compact_function.py

28个参数化测试，覆盖CPU/GPU、标量/多输出、自定义Function、卷积/池化多阶导数、视图、突变、stream、生命周期及接口拒绝；精确比较数据字节。

## 诊断与交付脚本

- save_baseline.py：冻结之前的dirty工作区、源码和动态库。
- common.py / training.py / run_tasks.py：统一配置、资源检查、12轮训练、补充/内存/profiler任务。
- layers_main18.py：主窗口7卷积+4通用pool，保持该次固定seed生成顺序。
- layers.py / tiny_workload.py：真实TinyNet 4通道pool补测与运行时shape证据。前者的预验证直调只供诊断，不是生产API。
- analyze_profiles.py / aggregate.py / summarize.py：从原始计时/CUPTI计算表格及配对CI。
- report.py / figures.py：报告与PPT图；package.py：清单与打包。

完整本阶段生产/测试diff见git_diff.patch；之前轮次的全部已存在修改位于baseline/working_tree.patch，不混入本轮增量。source/提供最终必要源码，baseline/source/提供原版。SOURCE_INTEGRITY.json对83个基线文件逐个SHA比较，仅core.py改变，kernel、runtime、functions_conv、optimizers、动态库均不变。没有Commit/Push/PR。
''')
# Preserve history explicitly without modifying older outputs.
for p in (R/'runs/a100_20261107/framework_optimization/raw').glob('previous_gemm_*'):shutil.copy2(p,O/'raw'/p.name)
md('REPRODUCE.md','''
# 复现实验

固定环境记录在ENVIRONMENT.json、baseline/ENVIRONMENT.json与raw/resource_*.json。使用现有Python3.10环境：`/home/xieminhui_pubuse/hytorch-env/bin/python`。不要在原交付目录覆写数据；run_tasks遇到已有日志会拒绝运行。

## 创建隔离副本（无Git操作）

```bash
# 在原仓库根执行；TASK_REPRO必须是尚不存在的新目录。
TASK_SRC=/home/xieminhui_pubuse/hytorch/runs/a100_20261107/conv_pool_framework_optimization
TASK_REPRO=/home/xieminhui_pubuse/hytorch_conv_pool_reproduction
mkdir "$TASK_REPRO"
cp -a "$TASK_SRC/source/." "$TASK_REPRO/"
mkdir -p "$TASK_REPRO/runs/a100_20261107/conv_pool_framework_optimization"
cp -a "$TASK_SRC/diagnostics" "$TASK_REPRO/runs/a100_20261107/conv_pool_framework_optimization/"
mkdir -p "$TASK_REPRO/runs/a100_20261107/conv_pool_framework_optimization/"{raw,stage_a,traces,figures}
cd "$TASK_REPRO"
```

source包含本轮所需hytorch、tests、benchmarks、donkeycar_hy及原CUDA动态库。再次构建不是本次Python-only变更所需。原始baseline冻结命令为diagnostics/save_baseline.py；它要求真实Git工作区且目录不存在，隔离副本直接使用交付的baseline快照，不在无Git副本重跑。

## GPU任务

runner固定物理GPU1并在每个任务前检查其是否已有计算进程；占用时退出，不终止别人任务。CUDA_PATH=/usr/local/cuda、CUDA_VISIBLE_DEVICES=1、NVIDIA_TF32_OVERRIDE=0、PYTHONHASHSEED=0、CPU线程4由runner设置；每项运行命令、UTC与exitcode在raw/commands.jsonl。先查询nvidia-smi并遵守服务器授权。

```bash
PY=/home/xieminhui_pubuse/hytorch-env/bin/python
D=runs/a100_20261107/conv_pool_framework_optimization/diagnostics
$PY $D/run_tasks.py baseline cpu special tests tests_on memcheck racecheck
$PY $D/run_tasks.py layers:--cpu layers:--repeats:18:--tag:balanced18
$PY $D/run_tasks.py layertrace:B0 layertrace:O1 layertrace:torch
$PY $D/run_tasks.py balanced cpu:O1
$PY $D/run_tasks.py trace:B0:b trace:O1:b trace:torch:b trace:B0:c trace:O1:c trace:torch:c
$PY $D/run_tasks.py supplement memory memory_torch workload tiny_workload
$PY $D/run_tasks.py layers:--repeats:18:--tag:actual18:--case-prefix:tiny_actual_pool
$PY $D/run_tasks.py layertrace:B0:actual layertrace:O1:actual layertrace:torch:actual
$PY $D/analyze_profiles.py
$PY $D/summarize.py
$PY $D/aggregate.py
```

主局部计时balanced18使用layers_main18.py，其输入生成顺序与原始主窗口一致。layers.py额外构建4通道真实pool；`layertrace`主shape应使用与主窗口一致的layers_main18.py，runner已为非actual追踪选择它。原15次探索数据保存在raw/LAYER_TIMINGS.csv；正式图表只用18次平衡样本。trace每shape15次，三种框架单独capture；所有trace仅作解释，不参与wall验收。

原生产路径在baseline/source/core.py；正式B0是同一最终源码关闭新开关，O1开启，二者其他配置相同。隔离副本测试通过后如需生成报告，把交付baseline/复制到对应输出目录；report.py中的断言要求仅core.py区别于基线。报告整理脚本不需要GPU：

```bash
$PY $D/report.py
$PY $D/figures.py
$PY $D/package.py
```

报告时间窗口和历史事实针对原始运行；重新运行后请更新人工说明，不能把它们当新实验元数据。代码/测试都不要求提交Git。
''')
# Save final status without creating commits.
if (R/'.git').exists():(O/'raw/GIT_STATUS_FINAL.txt').write_bytes(subprocess.check_output(['git','status','--short'],cwd=R))
print('Reports and source snapshot written')
