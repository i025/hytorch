# 执行中的源码审查

基线c00a8d380d473781880ed8f14d1bc0818e10f52d。完整算子映射复用
[上阶段审查](../cuda-reduction-foundation/source-analysis.md)，本轮不重复开发现有Kernel。

| 分类 | 源码事实 | 本轮安排 |
|---|---|---|
| 已有原生并曾验收 | GEMM Forward、Conv Forward、Pool Forward、Gather dInput、Split-K dWeight、Fused SGD | 在新窗口跑完整回归，不能借历史时间替代当前测量 |
| 原型待GPU验收 | 原生Sum、原生ReLU Backward | 第一/第二优先级，保持原始容差，保存负样本 |
| 仍走CuPy | Linear/MatMul Backward显式native=False；BMM、Softmax/多数激活、原始卷积和Pooling梯度 | PyTorch/cuBLAS不等于自研CUDA C；本轮不再增加Kernel |
| 优化不一定传递至端到端 | 已有dInput剪枝/部分形状Gather、CPU预取 | 必须新窗口匹配A/I/P/R/S/J/E/E-prefetch |
| Python调度 | Variable.backward每次入队对整个funcs按generation排序 | 仅此一个新增实验原型，heap可选，保留相同generation与LIFO平局顺序 |

## CPU Profiling证据

`profile_cpu.py`：同样四逻辑CPU亲和性、BLAS1线程；两CNN各5步，分支图各50步，
完整cProfile函数表和pstats保存服务器。CPU原生GEMM/Im2col/Col2im主导CNN，不代表GPU瓶颈。

256分支图的sort自耗时0.864382s，generation key 0.669087s，总profile4.072875s，合计37.65%。
DonkeyNet排序自耗时仅64.368µs/5步；SpatialNet51.162µs/5步。不能据此声称CNN应有52%加速。
Function创建、isscalar/get_array_module、计算图操作也有开销，但本轮不同时重写它们。

## 一个通用调度原型

`using_config('backward_schedule','heap')`显式启用，默认list。
优先级(-generation,-入队序号)与旧稳定排序后pop顺序一致，seen_set仍阻止重复入队，
因此共享子图的梯度累加顺序也保持。没有修改算子数学、requires_grad、CUDA Kernel或SGD。
一阶/二阶、retain_grad、叶子、无梯度接收、异常恢复均有回归。

五次交替CPU完整step计时，CPU初始状态每次重置；DonkeyNet/SpatialNet使用4H同一真实batch，
另有TransformerEncoderBlock与合成分支图。完整数值比较188项逐元素一致。
CNN实测变化很小，暂视为未证实收益；分支图是适用范围证据，不能作为自动驾驶性能。

## 数据与Runtime边界

当前数据由有界AsyncDataLoader在CPU解码、保持indices和Batch顺序；GPU线程负责convert。
全组open_stream=True，I/A/E为同步，P/J/E-prefetch为同样2worker预取。
不是CUDA计算融合，不改数据划分、预处理或标签。
安装的CuPy12.3 `asarray(a,dtype=None,order=None)`没有新版本的blocking参数；
PyTorch当前convert的`.to(cuda)`使用默认non_blocking=False。
Pinned staging、独立传输stream和双缓冲需要明确生命周期/事件依赖及新GPU证据，本轮仅分析，不开发第二个原型。

Sum纯CPU轴规划和ctypes绑定的主机开销尚需与完整算子/核时交叉测量。减少launch不能保证端到端收益。
Softmax前向原数组sum、BatchNorm原数组mean/var不会因F.sum开关自动被替换；报告不能扩大覆盖范围。

CPU主机微测130份重复记录：plan_sum跨24形状平均7.54µs/调用（4.02–13.65µs），
backend上下文约3.48µs；纯主机ABI版本函数约0.135µs，**不是CUDA提交耗时**。
因此本轮Sum基准将backend上下文放在整个测量循环外，符合训练一次配置、多次调用的实际用法，
避免只给native路径额外计入每调用配置成本。历史准备脚本保留，不修改旧数据。

## 离线历史Trace交叉验证

本轮CPU重读4H `runs/udacity_real_analysis_01/trace-I.sqlite`（历史GPU1共享窗口），
不是本轮GPU0计算。新归因器得到2392个Kernel，全部匹配API；其中Forward624、Backward832、
Update16，另外920在所选训练phase以外（含初始化/预热等）。与历史CSV逐项核时匹配，
这份Trace不是4A更早的2664次Trace，不能混用计数。
16步的Backward累计17150.770µs（1071.923µs/step），主机NVTX累计67591.847µs；
两种口径不能相减冒充Python独占时间。
原转移表记录32次Memcpy、10137856字节，即每Batch633616字节（图像633600+标签16），
和Batch4 FP32协议相符；这是历史Trace字节核查，不是本轮H2D性能或GPU利用率。
两个CPU合成测试还验证：GPU执行晚于NVTX退出时仍按API归属正确phase，未匹配API明确报出。

## 本轮CPU Sum算术重放

两CNN真实数据三步检查156项训练比较、192项逐归约比较全部满足旧阈值。
随后每模型每路径256step，独立轨迹/检查点812项、共同参数状态的一步重放312项全部通过；
不是把CPU fallback称原生验证：实验显式注入了已有Sum的256lane/warp/分段FP32算术模型。
它仍不能证明GPU地址、同步或stream正确。GPU任务与上述CPU模型实验是分开的状态。
首次记录中的`gpu_priority_pause_s=0.0433`实为slot检查加等待总耗时，当时GPU窗口未启动，实际GPU优先暂停为0；
后续脚本字段已明确改名，保留原记录，不重写数值。没有继续追求长期逐位一致或放宽阈值。
CPU 长轨迹的对数误差图把实际为0的误差点显示在1e-12处；这只是对数轴显示下限，CSV/JSON中仍保留真实0值。

## Function / Variable 与类型判断的实测调用量

从同一份已保存的 CPU cProfile 导出逐函数调用次数和 self time，
不把互相嵌套的 cumulative time 相加。原始调用表见
[calls.csv](benchmarks/cpu-calls-01/cpu-calls-01/calls.csv)。

| CPU 工作负载 | Function.__call__/step | Variable.__init__/step | as_variable/step | np.isscalar/step | add_func调用/step |
|---|---:|---:|---:|---:|---:|
| DonkeyNet | 66 | 83 | 122 | 82 | 21 |
| SpatialNet | 41 | 53 | 75 | 54 | 12 |
| 分支32 | 260 | 326 | 487 | 454 | 128 |
| 分支256 | 2052 | 2566 | 3847 | 3590 | 1024 |

调用量包括 Forward 与 Backward 创建的临时 Variable/Function，不等于 GPU Kernel Launch 数。
add_func调用包含已在seen_set中的重复检查，不等于实际入队或执行的Function数。
例如 DonkeyNet 的 Variable.__init__ self time 为75.057µs/step，np.isscalar 为113.042µs/step，
但 Function.__call__ 的 self time 也可能包含数组释放、GC 或未单独记录的 C 调用。
这些都是带 cProfile 插桩的 CPU 记录，不能当作未插桩延迟，更不能直接当作 GPU 训练主机开销。

CPU CNN 中排序和 generation key 合计仅占 profile self time 约0.0168%/0.0117%，
而256分支图为37.6508%。这与独立拓扑补测的宽队列收益、短链无明显收益一致。
说明函数包装/类型判断值得继续量化，但本轮不以此推导未经实测的 GPU 加速，也不再开发第二个原型。
