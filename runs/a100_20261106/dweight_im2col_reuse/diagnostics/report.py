from pathlib import Path
import csv,json,hashlib,shutil,subprocess,platform,importlib.metadata
R=Path.cwd();O=R/'runs/a100_20261106/dweight_im2col_reuse'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with p.open('w') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def table(headers,rows):return '| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+''.join('| '+' | '.join(str(x) for x in r)+' |\n' for r in rows)
r=list(csv.DictReader((O/'raw/math_gate.csv').open()));names=list(dict.fromkeys(x['case'] for x in r))
summary=table(['形状','最大 |B−A|','最大相对误差 B/A','A 对 FP64 最大绝对误差','B 对 FP64 最大绝对误差','B/A 超过 Donkey 门限的组数'],[[n,*[f"{max(float(x[k]) for x in r if x['case']==n):.8g}" for k in ['max_abs_B_A','max_rel_B_A','max_abs_A_FP64','max_abs_B_FP64']],str(sum(int(x['donkey_B_A_failed'])>0 for x in r if x['case']==n))+'/9'] for n in names])
gates=table(['门限来源','atol / rtol','B vs A 失败组数','A vs FP64 失败组数','B vs FP64 失败组数'],[[name,tol,*[sum(int(x[f'{gate}_{cmp}_failed'])>0 for x in r) for cmp in ['B_A','A_FP64','B_FP64']]] for gate,name,tol in [('conv_host','test_cuda_host.py Conv backward','1e-5 / 1e-4'),('donkey','test_backward_dtype.py DonkeyNet','2e-5 / 2e-4'),('small_dtype','小型 Conv+Linear dtype 测试（辅助）','2e-6 / 2e-5')]])
write(O/'DWEIGHT_BENCHMARK.csv',[dict(**x,status='numerical_gate_only_no_performance_measurement',im2col_gpu_us='',packing_gpu_us='',gemm_gpu_us='',dweight_gpu_us='',cpu_submit_us='',cuda_api_us='') for x in r])
write(O/'TRAINING_COMPARISON.csv',[dict(mode=m,status='NOT_RUN',reason='numerical_compatibility_gate_failed_before_production_reuse_implementation',steps='',wall_s='',samples_per_s='',loss='',gradient_error='',weight_error='') for m in ['reuse_off','reuse_on']])
write(O/'PROFILE_BEFORE_AFTER.csv',[dict(mode=m,status='NOT_RUN',reason='stopped_at_numerical_gate',forward_gpu_us='',dweight_gpu_us='',dinput_gpu_us='',kernel_calls_per_step='',im2col_calls_per_step='',cuda_api_us='',cpu_scheduling_us='') for m in ['reuse_off','reuse_on']])
mem=[]
for n in names:
 x=next(z for z in r if z['case']==n);mem.append(dict(case=n,measurement='actual_isolated_col_nbytes',batch_shape=x['shape'],T=x['T'],D=x['D'],col_bytes=x['col_bytes'],training_peak_gpu_bytes='',training_growth_bytes='',note='size of actual allocated diagnostic matrix; not retained-workspace or training-peak measurement'))
base=sum(int(next(z for z in r if z['case']==n)['col_bytes']) for n in ['conv1','conv2','conv3','conv4','conv5'])
for batch in [1,4,8,16]:mem.append(dict(case='DonkeyNet_5_layer_sum',measurement='analytic_payload_only_NOT_measured_peak',batch_shape=f'batch={batch}',col_bytes=base*batch//4,training_peak_gpu_bytes='',training_growth_bytes='',note='4*sum(T*D), excludes allocator/other tensors/lifetime overlap; no implemented reuse'))
write(O/'MEMORY_COMPARISON.csv',mem)
(O/'REPORT.md').write_text(f'''# 阶段4D-1：前向im2col复用的数值准入实验

## 执行结果：在生产修改前停止

按用户第二阶段停止条件，本轮完成源码核对、独立数学/布局/精度实验和原始基线验证后停止。没有增加reuse开关，没有修改生产或测试源码，也没有修改CUDA动态库、GEMM、Col2im、dInput或dWeight原路径。已有4C-2缓存/POD与FP32/GEMM128CTA修改完整保留。

**停止依据是保守的旧实现数值兼容性准入，而非数学公式错误或已证明新实现更不准确。** 将现有DonkeyNet梯度比较数值门限atol=2e-5、rtol=2e-4用于B对A的替换一致性检查，90组中12组超限（12个元素，4个shape/seed组合在3种输入layout下重复）。新B对FP64在这一门限下全部通过；旧A对FP64有9组失败。因此不能把新旧不一致直接解释为新实现精度下降。若审查认为应以高精度参考而不是旧结果作为唯一准入依据，需要明确这一验收解释后再实施，不能在本轮自行放宽或替换验收含义。

原有测试门限本来用于其指定样本与PyTorch比较，并不是仓库声明的全形状误差定理。本轮将同样数值用于unit-normal通用数据的新旧兼容性诊断，原测试本身仍全部通过；没有声称原DonkeyNet单元测试失败。小型dtype更严格门限仅作辅助展示，不用于单独宣判算法错误。

## 当前源码核对

- functions_conv.py Conv2d.forward选择native_conv，runtime.conv2d的im2col路径先 `_arrays` 打包非连续输入，再生成(T,D)连续矩阵、hy_im2col、forward gemm、hy_conv_pack。返回仍只有y，col未传给Conv2d实例。
- Conv2d.backward无条件调用Conv2DGradW(self)(x,gy)，GPU Conv2DGradW.forward重新im2col_array(to_matrix=False)，再tensordot(gy,col,((0,2,3),(0,4,5)))。与指定设计前提一致。
- hy_im2col令row=i/(IC*KH*KW)、q=i%(IC*KH*KW)；row解码n/oh/ow，q解码ic/kh/kw，确实是T=(n,oh,ow)、D=(ic,kh,kw)。
- 核对Config.enable_backprop/create_graph、Variable.backward、Conv2DGradW.backward及现有dtype/高阶梯度测试。当前Variable.data直接暴露可变ndarray，没有通用原地修改version counter，未来复用需要明确保护/失效机制；仅保存指针不能证明内容未变。本轮未实现或验证新的生命周期处理。

## 独立实验

A = 当前GPU im2col_array(...,to_matrix=False) + 原CuPy tensordot。
B = 现有hy_im2col产生真实forward(T,D)列矩阵，G=gy.transpose(1,0,2,3).reshape(OC,T)，B=(G@X_col).reshape(OC,IC,KH,KW)。使用CuPy/cuBLAS矩阵乘法；没有改native GEMM。

数据：5个DonkeyNet真实形状+5个通用形状（非方形kernel、不规则尾块、重叠、padding、stride>1、holes、不同batch/channels）；每个3种输入布局（连续/负stride/步长2slice）×3个seed(42,123,501)，共90组。x、gy均固定seed的N(0,1)转FP32，覆盖通用数值范围，不代表192步训练中每层的实际梯度尺度。每个输入layout表达相同逻辑值，3种layout重复并非3个独立随机样本。统一非默认CUDA stream，NVIDIA_TF32_OVERRIDE=0，没有fast-math。

90/90原native列矩阵与原im2col六维tensor重排结果逐位相等，证明布局公式与非连续输入packing正确。dWeight只有54/90组逐位相等；其余36组改变FP32归约结果。逐项误差判据为abs(B−A) <= atol+rtol*abs(A)。相对误差定义abs(diff)/max(abs(reference),1e-30)，近零位置会放大，因此必须结合绝对误差与混合门限。

{summary}

{gates}

“B对FP64通过Donkey门限”不意味着B每个元素都更准确：例如conv1最大绝对参考误差B=1.4158e-4、A=1.2613e-4，最大误差位置与混合门限失败位置不同。报告完整保留两边参考误差，不选择性展示。

## 可复现失败坐标

conv1 / seed42 / contiguous，dW[20,1,0,2]：

- A=-0.11882781982421875
- B=-0.11876296997070312
- 独立CPU long-double参考=-0.11878202656945513
- |B−A|=6.4849853515625e-5，原Donkey数值门限此处约4.3765564e-5，超限1.4818倍。
- A参考误差4.5793255e-5，B参考误差1.9056599e-5。该例新结果更接近参考，但不满足按旧结果判定的兼容性检查。

conv2 / seed123 / contiguous，dW[14,3,1,2]：A=-0.008676528930664062，B=-0.008701324462890625，CPU参考=-0.008687375804047267。两者对FP64均在Donkey门限内，但B−A约2.47955e-5超过旧结果兼容门限，说明两种FP32舍入不能简单视为同一结果。

完整失败点见raw/failure_examples.json。CPU独立im2col + np.longdouble归约复核7个记录（含重复comparison），与GPU FP64参考最大差约1.04e-13，排除了复用同一GPU布局导致的参考错误。不是由TF32或降精度导致；实际cuBLAS内部归约树差异是合理解释，本轮没有采集其底层算法ID，不将此推断表述为已观测kernel实现细节。

## 验证结果

主机24+9=33 passed；原GPU204 passed、12 CPU项目deselected、0 skipped。日志在raw/baseline_*.log，强制HYTORCH_REQUIRE_CUDA=1。memcheck和racecheck分别执行完整90组独立实验，错误0、race hazards0。数值兼容性失败作为数据输出，不是内存安全错误；Sanitizer退出码0也不代表数值门限通过。

这些是未修改基线与独立算术实验的验证，不能当作已实现reuse的高阶梯度、重复backward、no_grad、输入原地修改或跨step生命周期验证。后者没有候选实现，未执行。

## 性能与显存：未执行项明确留空

因数值准入未通过，未实现生产reuse，未运行10轮端到端或新旧Nsight性能比较，也未宣称消除backward im2col。DWEIGHT_BENCHMARK.csv包含90组数值准入数据，所有GPU/CPU时延字段留空；TRAINING_COMPARISON.csv、PROFILE_BEFORE_AFTER.csv以NOT_RUN和原因记录，不能当作零耗时或零加速。

MEMORY_COMPARISON.csv的单case col_bytes来自真实分配矩阵的nbytes；理论若五层都保留列矩阵，batch4 payload合计{base:,} bytes（{base/2**20:.3f} MiB），batch同比线性增长。batch1/4/8/16行均明确标为公式推算，不是训练峰值显存实测。未测额外存活时间、allocator峰值或跨step增长，不据此判断内存代价是否可接受。

本轮停止点无需运行未验证候选的192步性能；也没有引入保持旧累加布局的第二方案、implicit GEMM、梯度裁剪或其他CUDA kernel。前轮任何0.83s训练时间不作为本轮性能结果复用。

## 资源与交付

使用物理GPU1 A100-PCIE-40GB，先确认0MiB/0%且无计算进程。GPU0其他任务未触碰。同一阶段使用窗口内完成数学实验与基线验证，最终GPU1无计算任务。源码和动态库SHA与baseline完全一致。Git仍保留进入本轮时的未提交修改，没有Commit、Push或PR。

baseline保存原始源码、已有git diff、状态和动态库；source保存复现依赖；git_diff.patch是空文件，表示本阶段没有生产或测试源码变更。诊断脚本、原始90组数据、失败坐标、独立参考、验证日志、环境和命令均打包；动态库留服务器并给出SHA，不包含大型数据集/权重。完成后停止，交给ChatGPT审查数值验收含义。
''')
(O/'CORRECTNESS.md').write_text('''# 正确性状态

- 原主机回归：33 passed（24+9）。
- 原GPU回归：204 passed、12 deselected CPU项目、0 skipped。
- 列矩阵layout：90/90逐位相等，包含负stride和非单位slice。
- dWeight B/A：54/90逐位相等，36组非逐位一致。
- 原Donkey数值门限atol2e-5/rtol2e-4用于新旧一致性：12/90组失败（12个元素、4个独立shape/seed，三种layout重复）。
- 同一门限对FP64：旧A9/90失败，新B0/90失败。不能称为新算法普遍精度恶化。
- Conv host门限atol1e-5/rtol1e-4用于B/A：21/90失败。更严格小型dtype门限仅辅助记录。
- 独立CPU long-double参考7个失败记录与GPU FP64参考最大差1.04e-13。
- Compute Sanitizer memcheck：90组完成，0 errors。
- Compute Sanitizer racecheck：90组完成，0 hazards、0 errors、0 warnings。

Sanitizer与重复诊断的数值结果见raw/memcheck.csv、raw/racecheck.csv；它们的成功退出不代表数值准入通过。未增加生产reuse，不存在“新路径完整训练通过”结论。未执行新reuse高阶梯度、原地修改、生命周期、跨step stale数据和训练对齐测试。

原测试源码与门限未修改；大形状N(0,1)数据的新旧比较是独立实验，原单元测试仍通过。门限原本的参考对象为其指定PyTorch测试，本轮将数值阈值保守用于替换一致性，而未擅自改为只看FP64参考。
''')
(O/'SOURCE_CHANGES.md').write_text('''# 本阶段修改说明

没有生产或测试源码修改，没有新增reuse开关，没有重编译或替换CUDA库。git_diff.patch为空，baseline/git_diff.patch是进入阶段时已有修改，二者不能混淆。

核对并保存：functions_conv.py Conv2d/Conv2DGradW；runtime.py conv2d/_arrays/_geometry；kernels.cu hy_im2col/im2col_kernel；core.py Function/Variable.backward/Config；functions.py FP32修复。

只新增runs/a100_20261106/dweight_im2col_reuse下诊断脚本和实验交付：
- math_gate.py：原tensordot、真实native列矩阵、新矩阵布局、FP64参考与未放宽门限比较。
- check_reference.py：CPU独立展开与long-double复核失败点。
- validate_baseline.py：完整既有host/GPU基线和独立实验Sanitizer。
- report.py、package.py：报告、清单与压缩包。

未保存或跨step复用任何生产workspace。只在独立实验局部变量持有col，不能把其生命周期当作新生产方案。raw/scope_checks.json含文件与动态库SHA一致性证明。
''')
checks={}
for f in ['hytorch/functions_conv.py','hytorch/backends/cuda/runtime.py','hytorch/backends/cuda/kernels.cu','hytorch/core.py','hytorch/functions.py']:
 checks[f]=dict(before=sha(O/'baseline'/f),after=sha(R/f));assert checks[f]['before']==checks[f]['after']
checks['library']=dict(before=sha(O/'baseline/libhytorch_cuda.so'),after=sha(R/'hytorch/backends/cuda/build/libhytorch_cuda.so'));assert checks['library']['before']==checks['library']['after']
(O/'raw/scope_checks.json').write_text(json.dumps(checks,indent=2));(O/'git_diff.patch').write_text('')
(O/'raw/environment.json').write_text(json.dumps(dict(python=platform.python_version(),platform=platform.platform(),packages={k:importlib.metadata.version(k) for k in ['numpy','cupy-cuda12x','torch','pytest']},tf32='NVIDIA_TF32_OVERRIDE=0',data='N(0,1), FP32, seeds42/123/501',gpu='physical1 A100-PCIE-40GB'),indent=2))
shutil.copyfile(R/'hytorch/backends/cuda/build/build.json',O/'raw/baseline_build.json')
for pattern in ['hytorch/**/*.py','hytorch/**/*.cu','tests/*.py','scripts/build_cuda.py','benchmarks/bench_cuda_training.py','donkeycar_hy/*.py']:
 for p in R.glob(pattern):
  if 'build' in p.parts:continue
  dst=O/'source'/p.relative_to(R);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst)
(O/'EXTERNAL_ARTIFACTS.json').write_text(json.dumps([dict(path=str(O/'baseline/libhytorch_cuda.so'),sha256=sha(O/'baseline/libhytorch_cuda.so'))],indent=2))
print('report written; all production and library hashes unchanged')
