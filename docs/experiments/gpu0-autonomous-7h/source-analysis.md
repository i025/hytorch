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
