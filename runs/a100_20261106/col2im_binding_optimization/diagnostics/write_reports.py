from pathlib import Path
import csv,json,statistics
O=Path('runs/a100_20261106/col2im_binding_optimization');stats=json.loads((O/'raw/training_statistics.json').read_text());profile=json.loads((O/'raw/profile_summary.json').read_text());bench=list(csv.DictReader((O/'COL2IM_COMPARISON.csv').open()));by={(r['case_id'],r['mode']):r for r in bench};cpu=list(csv.DictReader((O/'CPU_OVERHEAD.csv').open()))
def table(headers,rows):return '| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+''.join('| '+' | '.join(str(v) for v in r)+' |\n' for r in rows)
def val(case,component):return float(next(r['median_us'] for r in cpu if r['case']==case and r['component']==component))
training=table(['实现','wall mean ± sample SD (s)','samples/s mean','最终 loss'],[[m,f"{s['wall_mean']:.9f} ± {s['wall_sd']:.9f}",f"{s['samples_per_s']:.3f}",f"{s['loss']:.16g}"] for m,s in stats.items() if m in ['cupy','old','fast']])
pairs=table(['轮次','CuPy − fast (ms)','old − fast (ms)'],[[i+1,f"{stats['cupy_minus_fast']['pairs_s'][i]*1000:.3f}",f"{stats['old_minus_fast']['pairs_s'][i]*1000:.3f}"] for i in range(10)])
ci='\n'.join(f"- {b} − fast：均值 {stats[b+'_minus_fast']['mean_s']*1000:.3f} ms；描述性配对 t 95% CI [{stats[b+'_minus_fast']['ci95_t_s'][0]*1000:.3f}, {stats[b+'_minus_fast']['ci95_t_s'][1]*1000:.3f}] ms。" for b in ['cupy','old'])
cputable=table(['case','旧 Native 完整 wrapper','新 Native 完整 wrapper','预验证旧 ABI','预验证 POD','metadata hit','uncached metadata'],[[f'conv{i}',*[f'{val(f"conv{i}",k):.3f}' for k in ['old','fast','direct18','direct4','metadata_hit','metadata_validate_uncached']]] for i in range(1,6)])
shapetable=table(['case / training layout','CuPy GPU μs','旧 Native GPU μs','新 Native GPU μs','CuPy/旧/新 单次提交 μs','CuPy/旧/新 连续5次提交 μs'],[[by[str(i),'cupy']['name'],*[f"{float(by[str(i),m]['gpu_us_median']):.3f}" for m in ['cupy','old','fast']],'/'.join(f"{float(by[str(i),m]['single_submit_us_median']):.3f}" for m in ['cupy','old','fast']),'/'.join(f"{float(by[str(i),m]['five_submit_us_median']):.3f}" for m in ['cupy','old','fast'])] for i in [0,3,6,9,12]])
prtable=table(['指标 μs/step（除比例）','CuPy','old','fast'],[[label,*[f'{fn(profile[m]):.3f}' for m in ['cupy','old','fast']]] for label,fn in [('全部 GPU kernel 累计',lambda s:s['kernel_us_per_step']),('col2im GPU 累计',lambda s:s['col2im_us_per_step']),('backward GPU 累计',lambda s:s['phase_gpu_us']['backward']),('forward GPU 累计',lambda s:s['phase_gpu_us']['forward']),('CPU backward elapsed',lambda s:s['phase_cpu_elapsed_us']['backward']),('CUDA launch API CPU elapsed',lambda s:sum(v['us_per_step'] for k,v in s['cuda_api'].items() if 'Launch' in k)),('GPU 活动区间占 trace wall (%)',lambda s:s['activity_union_fraction']*100)]])
reg=[]
for i in range(36):
 a=float(by[str(i),'cupy']['gpu_us_median']);b=float(by[str(i),'fast']['gpu_us_median'])
 if b>a:reg.append([by[str(i),'cupy']['name'],by[str(i),'cupy']['layout'],f'{a:.3f}',f'{b:.3f}',f'{(b/a-1)*100:.2f}%'])
report=f'''# 阶段 4C-2：Col2im binding 优化

## 结论与范围

已保留有配对证据支持的两项修改：128-entry LRU 不可变 metadata 缓存和可选 POD C ABI。新旧 Native 执行完全相同的 col2im_strided kernel，未改变 GEMM 128 CTA dispatch、dWeight、forward 或其他算子；FP32 修复原样保留。默认 col2im 仍为 cupy，没有提交、推送或创建 PR。

本窗口无 profiler 训练：新 Native 比旧 Native 平均减少 13.027 ms（1.547%，10/10 轮更快）；比 CuPy 减少 5.510 ms（0.660%，9/10 轮更快）。这是小幅但本轮配对证据支持的端到端改善，不是 CUDA kernel 算法加速，也不代表其他机器或 workload 普遍获得相同比例收益。保留默认值，将是否切换默认实现交给后续审查。

## 先诊断，再修改

初始未修改源码诊断 raw/diagnosis.csv：完整 wrapper 16.52–17.07 μs，预验证直接旧 ABI 6.06–6.22 μs；差约 10.5 μs/调用。每次192步、每步5次，共960次调用，对应约10 ms 的简化路径上限，足以覆盖上轮约5.47 ms退化，因此允许进入最小 binding 实验。诊断路径仅在 diagnostics 下，未成为生产接口。

raw/pilot.csv 是同进程交替配对最小候选：cache-only 约13.4–14.0 μs，相比旧 wrapper约17 μs节省3.3–3.6 μs；再加 POD 降到约11.2–11.6 μs。独立旧ABI直接调用6.18–6.37 μs，而POD直接调用4.03–4.16 μs，约2.2 μs差值支持精简18个 int64 参数传递。两项方案均有实测收益才保留。

最终独立低开销诊断（μs/调用，31批×20次，非 cProfile）：

{cputable}

当前 stream 读取约0.26–0.27 μs；backend selection约1.94–1.99 μs；独立 dtype/device/alias 组合约0.68–0.71 μs；8个整数归一化约1.4 μs。它们不是严格可加的路径分解：微基准循环与辅助函数有固定开销。完整数据在 CPU_OVERHEAD.csv；其 scope 区分 unprofiled CPU submit、CUPTI API elapsed 和 cProfile instrumented self elapsed。

cProfile 单独运行每层每实现300次，未混入正式训练。Python col2im、生成器、metadata归一化与调用封装是可归因的主机工作；cProfile self时间仍可含C扩展、ctypes及抢占等待，并非纯Python字节码on-CPU时间。API CPU elapsed来自另一独立Nsight trace，不能直接从不同实验的self time相减来得到精确Python主动执行时间。

CPU提交计时包含同步调用返回前的Python/ctypes/CUDA主机路径，但不等待GPU完成；同步后完整执行时间另列。CUDA event测GPU，端到端wall计所有主机与GPU关键路径。上述时域有重叠，禁止相加。低开销 timer 在每批前同步，排除前一批队列拥塞；其结果仍依赖批大小和调用链。单算子CSV的单次/五次提交与20次批量CPU诊断不是同一条件。

## 完整调用链与实现

`col2im_array`（未修改）先处理几何及可能的reshape/transpose，再 `selected` 按backend/dtype选择。Native每次检查CuPy FP32、6D输入、4D输出、整数几何和当前device；以真实 `img_shape,col.shape,col.strides,stride,pad` 值查LRU；未命中时执行原几何、整数stride、索引溢出检查。每次仍验证输出dtype/shape/连续性/device及输入输出alias，读取最新input/output地址和当前stream，再由 `_call` 经过ctypes启动可选 `hy_col2im_fast`，错误仍经 `hy_cuda_error` 传播。

LRU容量128，不缓存GPU地址、数组、device或stream。值为整数tuple和不可变bytes。整数归一化在查缓存前进行，避免float与int相等导致无效geometry命中。新ABI参数是input/output/params/stream；params用 `=18q`、native-endian 18个有符号64位整数、144字节，无padding。C端 `int64_t v[18]` 并静态断言size/alignment，先将可能未对齐的CPU bytes memcpy到对齐局部POD，再转发原hy_col2im；仅CPU内存拷贝，无GPU传输。bytes在同步host启动函数返回前保持存活，launch参数已被CUDA读取；不保留params指针。

字段顺序：N,C,H,W,KH,KW,OH,OW,SH,SW,PH,PW,s0,s1,s2,s3,s4,s5，strides为元素单位。原hy_col2im及ABI版本1保留，旧库有hy_col2im但没有fast符号时仍用旧接口，连col2im也没有时沿用CuPy fallback。显式FP64继续走原CuPy路径。

## 正确性

主机33 passed；GPU204 passed、12 deselected（CPU项目）、0 skipped。完整新增与原col2im测试在memcheck和racecheck下各103 passed，0错误、0hazards。其他原GPU/GEMM/dtype测试也执行，未更改门限。原大型GEMM严格门限失败属于前轮已知结果，未因本轮修改消失；本轮没有将那组诊断失败改成通过。

36个微基准case三种实现输出逐位相等；10轮完整训练30次，loss、最终梯度、权重逐位相等（最大误差0）。正/负/零/非单位strides、非连续view、非默认stream、metadata变化、地址变化、alias、输出dtype/布局、FP64 fallback、二阶梯度均覆盖。当前device不匹配通过mock current-device ID验证分支，错误输出device用测试专用CuPy ndarray subclass覆盖；未占用GPU0来创建真实跨设备buffer。详见CORRECTNESS.md。

## 单算子

原36 case（12几何×training/contiguous/negative），固定seed501+caseid，预分配输出，20次预热，15轮三方循环轮换/反转顺序。GPU event围绕32次CUDA Graph调用后除32，避免Python调用间隙；CPU single、five和同步完整执行在graph外单独测。使用统一非默认stream。原CuPy预分配版本从原 `_col2im_gpu` AST提取原ElementwiseKernel表达式；生产CuPy带分配/创建包装的完整调用另在CPU诊断与正式训练中测试。

{shapetable}

old/fast使用同一动态库和完全相同kernel。各case GPU中位数的微小差异是测量波动，本轮没有kernel加速机制；没有case出现fast相对old超过5%的GPU中位数回退。实际名称col2im_strided、block256、寄存器38、shared0；真实五层grid619/1140/371/83/60，旧新一致。CuPy block128、寄存器32。每case全部配置见COL2IM_COMPARISON.csv observed_signature，独立trace每实现3次。

新Native相对CuPy仍存在以下GPU退化，绑定优化不消除它们：

{table(['case','layout','CuPy μs','fast μs','退化'],reg)}

36case中31个GPU更快、5个更慢。所有108行及原始1620个配对测量记录均保留，没有选择性展示；完整执行时间与CPU提交p10/p90见CSV。

## 正式训练（无 profiler）

同进程、同一动态库、seed42、固定初始化、256样本、batch4、3epochs/192steps、dropout0、SGD0.001、同FP32与GEMM128CTA配置。每实现预热10步，测量前恢复相同权重；10轮三方轮换且反向平衡顺序（10不是6的倍数，位置计数差最多1）。训练包含DataLoader、H2D、forward/backward/SGD与末尾同步；无每步loss读取或同步。所有实现同样启用轻量phase NVTX标注，但正式运行未附加profiler。最终snapshot在计时外。

{training}

{pairs}

{ci}

CI使用10个逐对差值、df9、t=2.262157；不是两组独立SD比较，也不是跨机器总体置信保证。第2轮fast比CuPy慢1.809 ms，明确保留。相对旧Native平均节省13.027 ms，比隔离调用实验简单外推的约5–6 ms更大，说明不能把CPU微基准线性外推当作训练关键路径的定量分解。实验只改变binding，因此支持这组binding修改改善该workload；不能据此声称上一轮所有退化均被单一Python检查解释。

## 独立 Nsight 时间线

每实现单独10步trace，不是10轮profile重复。数据如下：

{prtable}

三组每步均147个kernel，col2im5次；10步各20次memcpy、6336160字节和1次最终stream同步；未添加GPU传输、同步或FP64cast。old/fast CUDA launch次数完全相同：每10步450次cudaLaunchKernel和1020次cuLaunchKernel。PROFILE_BEFORE_AFTER.csv用scope区分kernel/phase/category/API，重叠汇总不可重复相加。

短trace中fast的CPU backward elapsed比old高约70.7 μs/step，GPU old/fast也存在约7%的整体差异，包括未修改forward，故该单次profile不能证明host速度因果量或kernel改善。可能涉及profiling扰动、时钟/缓存或调度状态，但未控制锁频，无法判定。GPU活动比例约26–28%，这是时间区间占比，不是SM occupancy，更不能将全部空白归为Python。API launch elapsed也没有稳定随binding优化下降；改善主要证据是隔离wrapper对照与无profiler配对wall。

## 环境、故障记录与停止点

GPU1 A100-PCIE-40GB（108 SM），GPU0其他用户任务未触碰。诊断始于约08:59 UTC；正式命令清单窗口09:04:37–09:07:50 UTC，全阶段沿用GPU1。编译/编辑间GPU上下文自然退出，没有保留空转占用；每次继续前重新检查，无他人GPU1任务。最后GPU1为0MiB、0%利用率、无计算进程。

初次bench因legacy默认stream不能Graph capture而停止，raw/bench.log保留；只修正诊断脚本为统一非默认stream。一次恢复检查读到瞬时1MiB而拒绝启动，随后无compute process且0MiB/0%复查通过，继续剩余实验；没有覆盖之前通过的测试和原实验目录。

Python3.10.12/CuPy14.2/Torch2.9.1+cu128，CUDA12.4.99，driver580.95.05；nvcc -O3 C++14 sm80/compute80，无fast-math，NVIDIA_TF32_OVERRIDE=0。完整环境/build/命令保存在raw。未调整数值门限，未修改其他算子，默认cupy保留。已经停止，不进入dWeight/裁剪或第二套kernel优化。
'''
(O/'REPORT.md').write_text(report)
(O/'CORRECTNESS.md').write_text('''# 正确性与安全性

| 验证 | 结果 | 原始记录 |
|---|---|---|
| 全部原主机测试及metadata新增测试 | 24 passed、4 deselected GPU项目 | raw/host.log |
| host dtype | 9 passed、13 deselected GPU项目 | raw/dtype_host.log |
| GPU完整回归 | 204 passed、12 deselected CPU项目、0 skipped | raw/gpu_tests.log |
| Compute Sanitizer memcheck | 103 passed，ERROR SUMMARY 0 | raw/memcheck.log |
| Compute Sanitizer racecheck | 103 passed，0 hazards、0 errors、0 warnings | raw/racecheck.log |
| 新binding首次验证 | 5 passed | raw/binding_first.log |
| 36个单算子case三方 | 最大误差0，逐位一致 | COL2IM_COMPARISON.csv |
| 10轮×3完整训练 | loss/梯度/权重逐位一致，最大误差0 | TRAINING_COMPARISON.csv |

GPU强制HYTORCH_REQUIRE_CUDA=1，不能把unavailable转换为skip。现有test_col2im_cuda用独立CPU scatter与PyTorch Fold进行参考验证；重叠累加保持kh/kw顺序，与CuPy逐位相等。PyTorch及高阶梯度原门限rtol2e-6/atol2e-7未改变。

新增test_col2im_binding检查：144-byte POD整数布局、LRU容量128、重复metadata、shape变化、负/零strides变化、输入输出地址变化、非默认stream、热缓存后alias/dtype/布局/device/非法整数参数仍检查、旧ABI可选符号回退、空params返回错误。

设备错误检查采用mock当前device ID和测试专用错误device ndarray subclass，不碰GPU0其他用户任务；这不是实际多GPU buffer集成测试。所有原正/负/零/非单位stride、非方形kernel、非对称stride/pad、无贡献输出及显式FP64 fallback继续运行。cache value只有tuple与bytes，没有GPU地址、数组、stream或device。

原严格GEMM长K/2048/4096失败记录位于前轮gemm_shape_dispatch与fp32_fix_and_profile，未改变门限或声称解决；本轮仅执行既有GPU回归，未重复其大型诊断基准。原始失败证据另附raw/previous_GEMM_strict文件（如可用）。

失败记录：raw/bench.log为diagnostic Graph在legacy默认stream捕获失败；修正脚本后raw/bench_corrected.log通过。没有生产正确性失败或Sanitizer错误。检查raw/scope_checks.json确认旧CUDA源码为新文件逐字节前缀、functions_conv与FP32 functions.py未变。
''')
(O/'SOURCE_CHANGES.md').write_text('''# 本阶段源码改动

本阶段git_diff.patch仅包含以下两处生产文件和一个新增测试；baseline/git_diff.patch保留进入阶段时工作区已有修改，禁止将后者误认为本轮修改。

1. hytorch/backends/cuda/runtime.py
   - load_library新增可选hy_col2im_fast签名[p,p,c_char_p,p]，原ABI和旧库兼容不变。
   - 新_col2im_metadata @lru_cache(maxsize=128)：以实际值key验证不变几何/strides，返回tuple+144-byte不可变CPU bytes。未命中时执行原验证；每次先operator.index，防float/int缓存key相等。
   - col2im保留逐次dtype/device/shape维数/output/alias检查；最新GPU地址、当前stream每次读取；命中缓存不保留指针。有fast符号时调用POD，缺少时旧hy_col2im回退。
2. hytorch/backends/cuda/kernels.cu
   - 只在原文件末尾追加Col2imParams及hy_col2im_fast。18×int64_t，static_assert大小144、自然对齐；memcpy从CPU bytes到aligned local，再调用原hy_col2im。参数生命周期仅覆盖同步host调用。
   - 原文件全部内容逐字节不变。col2im_strided及所有GEMM/forward/dWeight算法、block/grid、FP32语义、ABI版本1未变。
3. tests/test_col2im_binding.py：5项新增测试，见CORRECTNESS.md。

functions_conv.py未修改，FP32 functions.py未修改。默认仍cupy；无DonkeyNet shape特例。未引入pybind11或全运行时重构。诊断unchecked direct18/direct4只位于runs/.../diagnostics，生产无法选择；cache-only候选快照放在raw/cache_only_runtime.py，仅用于同进程消融基准。

POD字段顺序 n,c,h,w,kh,kw,oh,ow,sh,sw,ph,pw,s0..s5，native-endian、有符号64bit、无字段间padding；Python bytes无对齐承诺，因此C先memcpy。保留原hy_col2im参数检查、状态码、stream约定；raw C ABI客户端仍须提供有效buffers及144-byte参数区，Python负责完整几何与索引检查，契约没有降低。

附source当前必要源码和baseline对应原始源码。保存旧/新动态库及原Nsight二进制在服务器，路径和SHA见EXTERNAL_ARTIFACTS.json；压缩包不含大型二进制或权重/数据集。
''')
print('reports written')
