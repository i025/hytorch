from pathlib import Path
import csv,json,hashlib,subprocess,shutil,difflib,statistics,ast
ROOT=Path.cwd();O=ROOT/'runs/a100_20261106/native_col2im';R=O/'raw';PRE=ROOT/'runs/a100_20261106/gemm_shape_dispatch'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def table(h,rows):return '| '+' | '.join(h)+' |\n|'+'|'.join(['---']*len(h))+'|\n'+'\n'.join('| '+' | '.join(str(x) for x in r)+' |' for r in rows)+'\n'
b=list(csv.DictReader((O/'COL2IM_BENCHMARK.csv').open()));t=list(csv.DictReader((O/'TRAINING_COMPARISON.csv').open()));stats=json.loads((R/'training_statistics.json').read_text());p=json.loads((R/'profile_summary.json').read_text());hot=json.loads((R/'baseline_hotspots.json').read_text());cats=json.loads((R/'baseline_backward_categories.json').read_text())
shape_table=table(['层','之前最新col2im μs/step','col逻辑shape','元素strides','SH,SW','PH,PW'],[[r['layer'],f"{r['us_per_step']:.4f}",str(r['input'][0]['shape']),str([s//4 for s in r['input'][0]['strides_bytes']]),str(r['input'][1:3]),str(r['input'][3:5])] for r in sorted(hot['layers'],key=lambda r:r['layer'])])
bs=table(['层，真实非连续布局','CuPy event μs','Native event μs','加速','CuPy直接CUPTI μs','Native直接CUPTI μs'],[[r['name'],f"{float(next(x['median_us'] for x in b if x['case_id']==r['case_id'] and x['implementation']=='cupy')):.3f}",f"{float(r['median_us']):.3f}",f"{float(r['speedup_vs_cupy']):.3f}×",f"{float(next(x['profile_kernel_us'] for x in b if x['case_id']==r['case_id'] and x['implementation']=='cupy')):.3f}",f"{float(r['profile_kernel_us']):.3f}"] for r in b if r['implementation']=='native' and r['layout']=='training' and r['name'].startswith('conv')])
reg=[r for r in b if r['implementation']=='native' and float(r['speedup_vs_cupy'])<1]
regtable=table(['case','layout','CuPy μs','Native μs','Native延迟增加'],[[r['name'],r['layout'],f"{float(next(x['median_us'] for x in b if x['case_id']==r['case_id'] and x['implementation']=='cupy')):.3f}",f"{float(r['median_us']):.3f}",f"{(1/float(r['speedup_vs_cupy'])-1)*100:.2f}%"] for r in reg])
train_table=table(['实现','wall s，mean ± sample SD','samples/s','最终loss'],[[m,f"{stats[m]['wall_mean']:.6f} ± {stats[m]['wall_sd']:.6f}",f"{stats[m]['samples_per_s']:.2f}",f"{stats[m]['loss']:.12g}"] for m in ['cupy','native']])
pairs=table(['配对轮次','CuPy wall s','Native wall s','CuPy−Native ms（负为退化）'],[[r['repeat'],f"{float(next(x['wall_s'] for x in t if x['repeat']==r['repeat'] and x['mode']=='cupy')):.6f}",f"{float(r['wall_s']):.6f}",f"{float(r['paired_cupy_minus_native_s'])*1000:.3f}"] for r in t if r['mode']=='native'])
pt=table(['backend','GPU kernel总计','col2im','backward GPU','forward GPU','SGD GPU','backward CPU elapsed','launch API CPU elapsed'],[[m,f"{p[m]['kernel_us_per_step']:.3f}",f"{p[m]['col2im_us_per_step']:.3f}",f"{p[m]['phase_gpu_us']['backward']:.3f}",f"{p[m]['phase_gpu_us']['forward']:.3f}",f"{p[m]['phase_gpu_us']['update']:.3f}",f"{p[m]['phase_cpu_elapsed_us']['backward']:.3f}",f"{sum(v['us_per_step'] for k,v in p[m]['cuda_api'].items() if 'Launch' in k):.3f}"] for m in ['cupy','native']])
layertable=table(['层','本轮CuPy训练col2im μs','本轮Native训练col2im μs'],[[l,f"{p['cupy']['col2im_layers_us'][l]:.3f}",f"{p['native']['col2im_layers_us'][l]:.3f}"] for l in ['conv1','conv2','conv3','conv4','conv5']])
# Strict production-scope checks: old CUDA file is an exact prefix; old CuPy kernel untouched.
oldcu=(O/'baseline/hytorch/backends/cuda/kernels.cu').read_bytes();newcu=(ROOT/'hytorch/backends/cuda/kernels.cu').read_bytes();assert newcu.startswith(oldcu)
def fn(src,name):
 tree=ast.parse(src)
 n=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
 return ast.dump(n,include_attributes=False)
oldfc=(O/'baseline/hytorch/functions_conv.py').read_text();newfc=(ROOT/'hytorch/functions_conv.py').read_text()
for name in ['_col2im_gpu','_im2col_gpu','conv2d','deconv2d']:
 assert fn(oldfc,name)==fn(newfc,name)
oldtree=ast.parse(oldfc);newtree=ast.parse(newfc)
for n in oldtree.body:
 if isinstance(n,ast.ClassDef):
  actual=next(x for x in newtree.body if isinstance(x,ast.ClassDef) and x.name==n.name);assert ast.dump(n,include_attributes=False)==ast.dump(actual,include_attributes=False)
h=json.loads((R/'before_hashes.json').read_text());assert sha(ROOT/'hytorch/functions.py')==h['hytorch/functions.py']
changes=['hytorch/backends/cuda/kernels.cu','hytorch/backends/cuda/runtime.py','hytorch/functions_conv.py'];incremental=''
for rel in changes:
 incremental+=''.join(difflib.unified_diff((O/'baseline'/rel).read_text().splitlines(True),(ROOT/rel).read_text().splitlines(True),fromfile='baseline/'+rel,tofile='current/'+rel))
full=subprocess.check_output(['git','diff'])
for rel in ['tests/test_col2im_cuda.py','tests/test_col2im_host.py']:
 q=subprocess.run(['git','diff','--no-index','--','/dev/null',rel],capture_output=True);assert q.returncode==1;incremental+=q.stdout.decode();full+=q.stdout
(R/'this_task.patch').write_text(incremental);(O/'git_diff.patch').write_bytes(full);(R/'git_status.txt').write_bytes(subprocess.check_output(['git','status','--short']))
(O/'SOURCE_CHANGES.md').write_text('''# 生产修改与接口

仅修改3个生产文件，新增2个测试文件。现有GEMM128 CTA选择和所有旧CUDA kernel原样保留；旧kernels.cu是新文件的完整字节前缀。FP32修复functions.py的SHA完全不变。functions_conv中所有Function类、原_im2col_gpu/_col2im_gpu、Conv2d/Deconv2d数学实现AST均不变。

## kernels.cu:196 / :234

新增 `col2im_strided` 和可选ABI-1扩展 `hy_col2im`。每输出像素一个writer；无atomic、无shared memory；256线程/block，grid最多65535块，再用64位grid-stride覆盖输出。

对于 ty=y+PH：
```
lo=max(0, ty-(OH-1)*SH)
hi=min(KH-1, ty)
first=lo+(ty-lo)%SH  # 仅非空合法区间计算；SH=1省略mod
```
循环kh=first,first+SH,...<=hi，对应oh逐次减1。kw同理。起点做一次除法，内层没有反复整除、取模或无效贡献判断；SH/SW为1时起点直接减法。按原kh、kw升序累加FP32，地址为 `n*s0+c*s1+kh*s2+kw*s3+oh*s4+ow*s5`。

六个真实strides按元素单位作为int64值参数传入，支持负stride、零stride、非单位stride和训练rollaxis视图，不将col当连续内存。输出连续NCHW，不修改输入。图像索引/offset用64位，binding检查尺寸、总元素与可达offset范围，grid-stride末次递增也防signed overflow。C ABI正常调用须遵守binding检查的元数据约束。

## runtime.py:31 / :202

`backend(..., col2im="cupy")` 增加可选末尾参数，原位置参数不变；可选值cupy/native，默认cupy。增加hy_col2im ctypes签名：2个pointer + 18个int64 + 当前stream pointer。旧ABI-1库无新symbol仍可加载，选择native时回退CuPy。

`col2im(col,img_shape,stride,pad,out=None)` 明确检查FP32、6D、几何一致、当前设备、stride单位、宽索引界限、输出shape/dtype/连续性/别名。不调用_arrays，避免其ascontiguousarray复制输入。输入geometry不合法抛ValueError；非FP32直接调用底层native helper抛TypeError。生产dispatch对非FP32自动fallback，对超出支持的索引metadata/旧库缺扩展以NotImplementedError回退。没有默默降为FP32。

该Python校验/ctypes层有主机开销；本轮保留实现，不因训练退化引入第二套优化。

## functions_conv.py:453

只在col2im_array的GPU路径加selected("col2im",col)分支；native成功返回新输出，否则沿原_col2im_gpu。CPU路径与自动求导Col2im/Im2col类不变；显式FP64继续原CuPy路径。fallback计数记录在dispatch_stats。

示例：
```python
from hytorch.backends.cuda import backend
with backend(matmul="shared", conv="im2col", pool="native", col2im="native"):
    # Existing forward/backward code.
    loss.backward()
```
默认 `col2im="cupy"` 保留，未默认启用native。

## 测试与实验

新增tests/test_col2im_cuda.py（98 GPU cases）和tests/test_col2im_host.py（1 host case）。性能/分析脚本仅在ignored runs/native_col2im/diagnostics中；benchmark的预分配CuPy kernel从**原函数AST表达式**创建，逐case与生产函数输出核对，没有修改生产CuPy数学实现。

raw/this_task.patch是本轮增量；git_diff.patch含此前已有tracked改动及本轮新增测试，不应把既有FP32/GEMM实现归为本轮新增。baseline/保存完整必要基线源码，lib_baseline.so在服务器保留；未Commit/Push/PR。
''')
(O/'CORRECTNESS.md').write_text('''# 正确性

- 完整GPU回归：200 passed，11 deselected，无skipped。此前102项全部保留，新增98项。
- 主机：23项原主机/新backend契约 + 9项dtype回归 = 32 passed。
- Compute Sanitizer memcheck：67 passed，ERROR SUMMARY: 0 errors。
- Compute Sanitizer racecheck：67 passed，0 hazards（0 errors, 0 warnings）。

新增测试含KH/KW=1/3/5、stride1/2/3、padding0/1/2的27组合以及3个非对称组合；连续、训练转置、负stride三布局共90项。另测试零stride广播/步长切片、预分配out、非默认stream、连续3次调用、无贡献输出位置、输出别名/几何拒绝、旧ABI库fallback。

每组直接与原CuPy结果**逐位相等**，与独立CPU scatter参考逐位相等；输入前后逐位不变。CUDA Graph实际kernel节点检查native路径确实运行col2im_strided。CPU scatter是按kh/kw升序、逐oh/ow散射，不复用native合法范围推导。

FP32/FP64、to_matrix true/false另与PyTorch Fold的一阶和二阶梯度对照（rtol2e-6,atol2e-7）。显式FP64时无native调用，输出及一阶/二阶梯度保持FP64。自动求导类不变。

36个benchmark case（12种几何×3布局），每种15轮事件配对，全部与原生产CuPy函数逐位一致、输入不变；预分配CuPy基准的AST kernel也逐项与生产函数验证。7轮训练的全部最终参数梯度、权重和loss逐位一致，最大绝对差0；参数及梯度保持FP32。

既有GEMM的严格门限和此前失败记录未修改；本轮没有重新进行大型GEMM门限实验，也不把col2im测试通过描述成旧GEMM门限问题已解决。旧GEMM源码字节前缀保留，完整旧GPU回归执行。

约束：合法非空6D col、连续非重叠输出、当前设备；非法geometry和别名明确报错。超出支持索引metadata时生产路径fallback。未测试不受binding约束的外部任意C ABI调用。
''')
(O/'REPORT.md').write_text(f'''# 阶段4C：原生CUDA Col2im实现与A100验证

## 结论

指定算法已实现，默认仍为CuPy。原生kernel在真实五层布局的无profiler单算子事件测量中加速1.68–3.32×，但**7轮完整训练全部比CuPy慢**，均值0.831259→0.836731 s，退化约0.658%。不能宣称端到端加速，不默认启用native，不自行引入第二套算法。

200项GPU回归、32项host、memcheck/racecheck全部通过；同一pair最终loss、全部参数梯度和权重逐位相同。前轮FP32修复、GEMM128 CTA、前向和dWeight实现均保留。

## 修改前热点确认（不是旧FP64的215μs）

先读最新shape-aware dispatch的train_new trace：col2im **232.6826 μs/step**，占1556.4135 μs GPU kernel总计的14.95%，占backward880.7051 μs的26.42%，每步5次、每层1次。其占dInput340.5230 μs的68.33%；同期dWeight340.8753、dBias57.2996、激活反向73.7861、Linear反向39.7824 μs。故仍属主要热点，满足继续实现的前提。

五层FP32输入的布局如下，KH/KW即shape第3/4轴；元素strides由已实测byte strides除以4，均非连续：

{shape_table}

图像输出H/W分别conv1=66/200、conv2=31/98、conv3=14/47、conv4=5/22、conv5=3/20。最新trace的backward序列与此前FP32详细metadata逐项匹配名字/grid/block，共1470 kernels；标签来自匹配，duration取最新trace。raw/baseline_hotspots.json和baseline_backward_categories.json保存原始来源与校验方法。

## 实现与范围

新增col2im_strided/hy_col2im；runtime新增backend(col2im="cupy"|"native")和strided binding；functions_conv只在col2im_array加选择入口。默认cupy，显式FP64和不支持metadata走原CuPy，无静默降精度。

每个NCHW输出独立计算；先推导kh/kw合法上下界，再跳到stride同余类起点并按stride递增。oh/ow起点只算一次，内层逐次减1；stride1省略模和除法。按原kh/kw升序FP32累加，无atomic。传入6个真实signed int64元素strides，不pack输入；输出连续。输出、offset、步进防溢出由binding/宽索引保护。

原CUDA文件完整字节前缀不变，证明GEMM及其他既有kernel未修改。原CuPy _col2im_gpu、dWeight/前向/autograd类AST不变，FP32 functions.py SHA不变。完整代码、适用范围和精度行为见SOURCE_CHANGES.md。

## 正确性与Sanitizer

| 项目 | 实际结果 |
|---|---|
| 完整GPU回归 | 200 passed，无skipped |
| 主机回归 | 32 passed |
| memcheck | 67 passed，0 errors |
| racecheck | 67 passed，0 hazards/errors/warnings |
| 原CuPy与独立CPU scatter | 新增形状测试逐位相等 |
| PyTorch Fold及高阶梯度 | FP32/FP64、matrix/6D输入对齐 |
| 36个microbenchmark case | 输入不变，结果逐位相等 |
| 7轮训练 | loss、最终全部参数梯度和权重逐位相等 |

测试覆盖零/负/非单位strides、非方形核、非对称stride/pad、无贡献输出、非默认stream、连续调用及预分配out。完整日志和统计在CORRECTNESS.md及raw/。

## 单算子：同进程配对

固定seed501+case_id；相同col、输出预分配，10warmup，15次正反交替。CUDA event包围32次同算子CUDA Graph回放后除以32，排除H2D/D2H和逐算子Python提交间隙。原CuPy的ElementwiseKernel表达式由原函数AST逐字读取，仅诊断调用复用预分配out；每case再与原生产函数验证逐位一致。

{bs}

同一几何还测连续和负stride布局；另有重叠、不重叠、stride3有空洞、非对称pad、不规则尺寸、不同N/C等共36case，CSV共72行。独立直接调用CUPTI各3次用于确认kernel时间/配置；不是15轮事件统计的替代。Graph热缓存事件与直接调用/profile/训练不是相同缓存与提交条件，绝不跨列计算加速。

实际native kernel为col2im_strided，block256、38 registers/thread、0 shared bytes；旧CuPy col2im为block128、32 registers/thread、0 shared。每case实际grid见CSV observed_signature。没有测occupancy、带宽或stall计数器。

### 完整披露退化case

{regtable}

共31/36 case中位延迟改善，5/36退化。holes的KH=KW=1、stride3，多数输出无贡献，native较旧版本慢约20.7–26.9%；tiny几何两个布局约2.8%退化。宽索引/一般化边界计算和launch粒度可能在轻量shape上占优劣，但这是源码支持的假设，未用硬件计数器确认。本轮不增加shape特例或第二套优化。

## 端到端：唯一变量col2im backend

保持相同FP32修复、同一编译库、GEMM128 CTA和所有其他backends：shared/im2col/native。固定256合成样本、batch4、3epochs/192steps、seed42、SGD .001、dropout0；每模式10warmup，7轮交替，每轮恢复相同权重。

wall包括CPU DataLoader、pageable H2D、forward/backward/SGD和最终同步；无逐步日志、loss read或sync，参数/梯度快照在计时外。两组phase NVTX相同。

{train_table}

{pairs}

均值差CuPy−Native为{stats['paired_delta_mean_s']*1000:.3f} ms/192steps，配对差样本SD={stats['paired_delta_sd_s']*1000:.3f} ms，描述性t区间（df6，95%）为[{stats['paired_delta_95_t_interval_s'][0]*1000:.3f},{stats['paired_delta_95_t_interval_s'][1]*1000:.3f}] ms；7对均为负。不是只比较两组独立SD。该区间只描述当前窗口，不能保证其他设备/负载下同样比例。

14次最终权重hash一致，loss=0.0002556632098276168；梯度最大绝对差0，权重最大绝对差0。未运行真实数据集准确率实验。

## 同窗口训练profiling（μs/step）

{pt}

{layertable}

本轮CuPy trace测得col2im **217.6035 μs/step**，native为85.8467 μs，约减少60.55%；这是本窗口新FP32实测，不是历史FP64的215 μs。之前最新232.6826与本轮217.6035属于不同采样窗口，不用于直接计算本轮加速。

真实训练总kernel数均147/step，col2im均5/step；H2D/memcpy均20次/10steps、6,336,160 bytes，未增加拷贝。final stream synchronize均1次/trace，没有新增逐步同步。

### 为什么算子快但训练更慢

已测事实：backward主机NVTX elapsed从2949.548增至3056.474 μs/step；launch API elapsed从599.861增至628.768 μs/step。CuPy的5个driver launch被5个native runtime launch替代，总kernel数不变。GPU kernel+memcpy活动时间并集占trace窗口从28.91%降至27.12%，说明GPU计算缩短没有直接压缩完整提交窗口。

源码可见新增runtime会检查geometry/strides/设备/out别名并构造18个int64 ctypes参数；这些Python和binding成本与异步提交空隙可能抵消GPU节省。未开启Python/CPU栈采样，因此不能把全部CPU elapsed差定量归因于某一个检查。NVTX CPU elapsed包含子调用/可能等待，不是纯Python主动执行时间；API和GPU时间重叠，不可相加，也不能把profile差值直接换算成未profile wall差。

此外，未改代码的前向GPU也从520.066变为557.704 μs，dWeight从317.774变为340.740 μs，约7%的同向采样差异表明这两份单次trace有条件漂移（频率/缓存/采样影响等未分别测定）。因此不能把总GPU变化45.296 μs全归因于col2im算法，更不能忽略七轮未profile的实际退化。微基准15轮交替是kernel效果更稳健的证据，训练七轮配对是端到端验收依据。

## 环境、完整性与交付

物理GPU1 A100-PCIE-40GB，GPU0 VLLM未触碰。使用前确认GPU1空闲，同一集中窗口完成全部GPU工作后无本轮进程/显存占用。未锁频、未改权限、未终止他人任务。nvcc12.4.99 sm_80、-O3/-lineinfo/-std=c++14，无fast-math；运行环境/每条命令/UTC起止时间在raw/window_commands.json，用户时间按北京时间换算。

原baseline库、完整必要baseline源码和working diff保存。二进制trace/库留服务器，上传包仅含报告、CSV、文本原始日志、源码和脚本；具体路径/大小/SHA见EXTERNAL_ARTIFACTS.json。原GEMM严格门限的历史失败仍保留，没有调整或宣称解决。

结论：实现与独立算子加速成立，但本轮端到端验收无收益且轻微退化，默认CuPy保持。后续是否研究binding/调度成本或其他算子，由ChatGPT审阅决定，本轮停止。
''')
# Complete necessary baseline and current source snapshots.
source=[];baseline=[]
for e in json.loads((PRE/'SOURCE_FILES.json').read_text()):
 rel=Path(e['path']).relative_to('source');current=ROOT/rel
 if not current.is_file():current=PRE/'source'/rel
 dst=O/'source'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(current,dst);source.append(dst)
 bd=O/'baseline'/rel
 if not bd.exists():bd.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(PRE/'source'/rel,bd)
 baseline.append(bd)
for rel in ['tests/test_col2im_cuda.py','tests/test_col2im_host.py']:
 dst=O/'source'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,dst);source.append(dst)
(O/'SOURCE_FILES.json').write_text(json.dumps([dict(path=str(x.relative_to(O)),bytes=x.stat().st_size,sha256=sha(x)) for x in source+baseline],indent=2))
for name in ['previous_gemm_baseline_summary.json','previous_gemm_paired_comparison.csv']:
 shutil.copyfile(PRE/'raw'/name,R/name)
shutil.copyfile(PRE/'CORRECTNESS.md',R/'previous_gemm_correctness.md')
(R/'scope_checks.json').write_text(json.dumps(dict(old_cuda_source_exact_prefix=True,gemm128_unchanged=True,old_cupy_col2im_unchanged=True,all_autograd_conv_classes_unchanged=True,fp32_functions_unchanged=True),indent=2))
subprocess.run(['git','diff','--check'],check=True)
print('Reports and source snapshots written',len(source),len(baseline))
