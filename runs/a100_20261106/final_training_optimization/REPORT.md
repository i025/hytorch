# HyTorch 最后一轮集中优化：交付报告

实验日期：2026-11-07（Asia/Shanghai；命令日志使用UTC）。仅本窗口实测，不拼接历史百分比。

## 结论与推荐

**接受D：反向列矩阵直接布局生成；拒绝S：多张量SGD未通过subnormal逐位一致性。** 推荐本合成workload显式使用 `backend(matmul="shared",conv="im2col",pool="native",col2im="native",dweight="packed")` 和 `SGD(lr=.001,backend="legacy")`。全局默认仍dweight=cupy、col2im=cupy、SGD=legacy。

最终冻结源码下D训练平均0.820923s，B0为0.851160s，wall减少3.552%，12轮中12轮更快。配对节省30.237ms，95% t区间[27.827,32.647]ms。没有超过PyTorch，也不以此为目标。

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

| 集合 | 比较 | case数 | 失败case | 失败元素 | max abs | max relative L2 |
|---|---|---|---|---|---|---|
| 固定+留出 | A_R | 168 | 12 | 12 | 0.000126126922 | 3.42207e-07 |
| 固定+留出 | B_R | 168 | 0 | 0 | 0.000141582449 | 3.42207e-07 |
| 固定+留出 | B_A | 168 | 12 | 12 | 0.000152587891 | 3.35477e-07 |
| 真实训练 x/gy | A_R | 15 | 0 | 0 | 3.20851604e-10 | 1.9298e-07 |
| 真实训练 x/gy | B_R | 15 | 0 | 0 | 2.94283333e-10 | 2.08443e-07 |
| 真实训练 x/gy | B_A | 15 | 0 | 0 | 3.49245965e-10 | 2.83659e-07 |


B/R全部通过，不能因此说B每个元素更精确；A与B归约顺序不同。逐位比例仅在同dtype FP32数组上用uint32位视图计算，跨FP32/FP64不伪造位相同比例；另列exact_value_fraction。训练最终逐参数逐元素检查沿用scripts/report_cuda.py的atol1e-4/rtol1e-3；单步/最终梯度为已有Donkey dtype门限2e-5/2e-4。完整loss轨迹全部有限并通过对齐，不只比较hash。

最终训练最大权重误差2.98023224e-08，最大梯度误差3.7252903e-09；所有逐参数失败计数0，见TRAINING_ERRORS.csv。单步、补充范围及框架参考误差各有独立CSV。原大GEMM严格1e-4门限的长K/2048/4096失败仍保留历史证据，不称已解决。

## 正确性与安全检查

最终全套测试：272 passed、2 skipped；两项skip是原CPU DonkeyNet测试明确改由GPU覆盖，所有实际GPU项目执行。按集合划分CPU35、GPU237。新增专项35 passed；原型有一次预期数值拒绝记录（1failed/31passed，raw/special_initial.log），保留未删除。memcheck/racecheck对原型及新增调用各28 passed、6 deselected（CPU或已单独记录的special-rounding案例），0 errors/0 hazards。Sanitizer成功不替代数值准入。

D覆盖正/负/非单位stride的x/gy、非方形/非对称参数、非默认stream、FP64回退、重复backward、forward后原地改输入时与原路径同样读取当前内容、no_grad与二阶导数。SGD原型覆盖0/1/16/32/33/65tensor、空tensor、重分配梯度、lr变化、hooks/冻结/子类、重复存储、overlap与跨参数梯度别名、CPU/FP64/不支持布局回退。并非生产S已通过特殊浮点测试；这一失败正是禁用原因。

## 正式训练：同进程12轮平衡顺序

固定256合成样本、batch4、3epochs/192steps、seed42、dropout0、lr.001；所有四组forward/GEMM/col2im相同。每组预热10步后恢复权重/梯度/RNG；raw/order.json预先保存Williams平衡顺序。wall包含DataLoader/H2D/forward/backward/update和末尾sync；计时中不读取loss，保留GPU标量引用并在计时后统一读取。四组相同NVTX标注及日志行为，无profiler附加。没有删除慢轮。

| 配置 | 实际执行 | 平均±样本SD (s) | median (s) | samples/s | B0−该组 95%配对CI(ms) | 胜出 |
|---|---|---|---|---|---|---|
| B0 | D关 / SGD原路径 | 0.851160 ± 0.003312 | 0.851365 | 902.31 | - | - |
| D | D开 / SGD原路径 | 0.820923 ± 0.002368 | 0.820723 | 935.54 | [27.827, 32.647] | 12/12 |
| S | D关 / SGD请求native但已回退 | 0.851843 ± 0.002983 | 0.850785 | 901.58 | [-2.552, 1.185] | 6/12 |
| DS | D开 / SGD请求native但已回退 | 0.821635 ± 0.001617 | 0.821931 | 934.72 | [27.364, 31.685] | 12/12 |


PAIRED.csv逐轮差值、LOSS_TRAJECTORIES.csv完整轨迹、TRAINING_ERRORS.csv逐参数误差均在包中。95%区间来自12个配对差值、df11 t=2.200985，而非两组独立SD；样本相关性与单机单窗口限制仍存在。

初轮4.36%是补充int32几何范围检查前的结果，完整封存pre_range_guard，不用于最终推荐。因新检查改变CPU路径，使用相同预定顺序重跑受影响wrapper/训练/补充/profile；这里报告最终重测，未从两轮择优。另有换行清理的AST一致性记录；原始数值normal集合的数学路径不变。

## 算子成本与实测归因

FULL_WRAPPERS.csv记录完整Conv2DGradW Function构造、输入包装、native检查、packing、分配、cuBLAS和返回Variable，采用与一阶backward相同的no_grad。每case15次交替、预热10次，13几何×3layout，包含全部39case。以下为真实五层连续输入中位数：

| 层 | B0/D CPU提交中位数 μs | B0/D 同步完整调用中位数 μs |
|---|---|---|
| conv1 | 138.408 / 110.727 | 153.483 / 120.809 |
| conv2 | 142.052 / 111.882 | 187.956 / 149.148 |
| conv3 | 141.988 / 111.904 | 150.731 / 114.974 |
| conv4 | 138.961 / 109.916 | 143.308 / 109.913 |
| conv5 | 132.341 / 105.123 | 134.726 / 102.000 |


同步完整调用D退化case数=0/39，全部数据保留；小样本不代表任意几何保证。OPERATORS.csv是较内层计算路径和CUDA event envelope诊断，不能替代完整wrapper；event envelope包含可能的主机提交空隙。真正kernel时间来自CUPTI，OPERATOR_PROFILE.csv逐kernel列出。SGD prototype完整Optimizer.update计时含参数收集/hooks/filter/alias/descriptor，但明确REJECTED，不用其加速比宣称训练成功。

| 配置 | trace | 总kernel/step | dWeight kernel/step | im2col/step | SGD kernel/step | dWeight GPU μs/step | 全GPU μs/step |
|---|---|---|---|---|---|---|---|
| B0 | 0 | 147.0 | 24.0 | 10.0 | 32.0 | 286.138 | 1187.308 |
| D | 0 | 142.0 | 19.0 | 10.0 | 32.0 | 262.452 | 1327.254 |
| S | 0 | 147.0 | 24.0 | 10.0 | 32.0 | 295.939 | 1227.769 |
| DS | 0 | 142.0 | 19.0 | 10.0 | 32.0 | 262.026 | 1326.509 |
| B0 | 1 | 147.0 | 24.0 | 10.0 | 32.0 | 340.391 | 1412.252 |
| D | 1 | 142.0 | 19.0 | 10.0 | 32.0 | 227.645 | 1152.312 |
| S | 1 | 147.0 | 24.0 | 10.0 | 32.0 | 296.465 | 1227.349 |
| DS | 1 | 142.0 | 19.0 | 10.0 | 32.0 | 262.177 | 1326.729 |


两次交替短trace均确认D使dWeight copy kernel从10降至5，总dWeight kernel24→19，全步147→142。backward im2col仍5次，加forward5次总10次，GEMM及split-K reduction数保持9；cuBLAS选择由nt变为nn，具体内核名见PROFILE.csv。D省掉列矩阵重排copy，gy packing仍存在，非连续输入还需x packing。未改变GEMM原生kernel、forward、col2im或dInput算法。

SGD原型独立trace16参数32→1kernel；实际训练所有四组SGD仍32。S的完整调用可能显著更快，但特殊数值不兼容已阻断生产路径。dWeight CUPTI约从B0的两次286.1及340.4μs/step降至D对应值，具体数值见表，不能把全部波动归因D。未改forward/dInput阶段也有时钟/调度波动；2个短trace只用于计数和解释，不能替代12轮wall。

PROFILE.csv同时列CPU NVTX elapsed、CUDA API elapsed、GPU时长，可能互相重叠，禁止相加。CPU elapsed不是精确Python主动执行时间。没有锁频，GPU0服务未触碰；主机共享环境的噪声没有从样本中剔除。

## 显存与适用范围

| batch | B0/D峰值活跃 MiB | B0/D峰值保留 MiB | 3轮后活跃 MiB |
|---|---|---|---|
| 1 | 6.268 / 6.268 | 7.745 / 7.769 | 0.964 / 0.964 |
| 4 | 24.254 / 18.229 | 32.341 / 20.292 | 0.964 / 0.964 |
| 16 | 90.865 / 66.768 | 99.534 / 75.437 | 0.964 / 0.964 |


CuPy MemoryHook在每次malloc_postprocess读取pool.used_bytes/total_bytes，得到该allocator的峰值；不是低频nvidia-smi采样，也不涵盖cuBLAS/CUDA/Torch私有分配。每配置batch各3轮完整训练，cleargrads/gc/sync后活跃分配稳定；保留量不是泄漏。batch4每步少5次pool请求（完整训练减少960次），batch16同样；batch1本测量分配次数和峰值并未降低，必须保留这一边界。没有使用4D-1的理论11.565MiB充当本轮测量。

| 网络 | batch | B0 mean s | D mean s | wall减少 | 胜出 |
|---|---|---|---|---|---|
| donkey | 1 | 3.150568 | 3.057801 | 2.94% | 3/3 |
| donkey | 16 | 0.231454 | 0.223015 | 3.65% | 3/3 |
| tiny | 4 | 0.322124 | 0.309749 | 3.84% | 3/3 |


补充各3轮，只用于范围检查，不当作与12轮主实验同等置信证据。TinyNet少量样本可能受主机噪声影响；所有补充逐参数误差通过。推荐限本轮声明网络/几何与FP32语义，不宣称全形状/真实驾驶任务精度。

## 同窗口框架参考

| 框架参考 | 次数 | mean±SD(s) | samples/s mean |
|---|---|---|---|
| cupy | 5 | 0.848673 ± 0.005684 | 904.97 |
| torch | 5 | 0.307929 ± 0.011747 | 2496.87 |


PyTorch2.9.1+cu128、cuDNN enabled、benchmark默认False；matmul和cuDNN TF32关闭，SGD foreach=None/fused=None，额外实际分支检查观察到_multi_tensor_sgd，未关闭其正常foreach优化。PyTorch和CuPy各5轮是独立参考组，不与D的12轮形成严格配对显著性结论；它们在边界检查修正前同一GPU窗口测量，修正不影响其CuPy/torch代码路径。库算法、cuDNN卷积/自动求导/optimizer实现不同。PyTorch显著更快，不制造反超结论。HyTorch四组有相同轻量NVTX标注，Torch参考无这些Python包装，属于框架诊断差异，不能把全部差异归为单个kernel。

## 交付与未运行项

source最终源码、baseline原源码/库、git_diff.patch本轮增量、SHA.json、ENVIRONMENT.json、命令日志、原始测试/计时/错误数据、Nsight-rep/SQLite及解析脚本全部提供。CLAIMS.md映射每条PPT结论，PPT_SUMMARY.md六页结构，figures含SVG和PNG（零起点/明确误差条）。

NOT_RUN：真正启用S的正式训练/组合（准入失败）；跨窗口累计早期基线收益（未恢复重测）；真实DonkeyCar/Udacity任务精度和泛化；全形状/全数值范围保证；实际跨物理GPU混合设备集成（未碰GPU0）。不使用历史不同窗口百分比填这些空白。

已释放GPU，不提交、推送、创建PR，不进入下一轮优化。
