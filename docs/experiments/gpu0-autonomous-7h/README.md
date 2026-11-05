# GPU0七小时自主研究（进行中）

基线c00a8d3，分支`experiment/gpu0-autonomous-7h`。
北京时间2026-11-06 00:18:50开始，07:18:50截止；06:48:50停止新的GPU任务并预留收尾。
本文为阶段性记录，最终状态以本轮时间线和原始数据更新为准。

物理GPU0已核对UUID `GPU-f099fdbd-adf2-6157-3462-4145bc2a8359`，A100-PCIE-40GB。
初查有其他用户PID13632，占约29.5GB；不能因utilization=0就启动。
从00:21:27起独立tmux监控约60秒一次，连续3次无compute进程、显存≤128MiB、利用率≤2%才可启动。
未发现squeue/qstat/bjobs命令，遵循用户对空闲GPU0的明确授权；没有独占租约，冲突即停，只结束本轮子进程。

## 阶段性状态

- SM80离线编译完成，现有CUDA Kernel未修改。
- CPU最终主机回归213 passed、0 failed、0 skipped；此前210项及定向集合记录保留，不重复相加。
- 新调度的CNN/Transformer CPU数值比较188项逐元素一致；默认仍list。
- 初次新增测试2项因FP64字面量参考失败，改成FP32参考后14项通过，未改容差或生产数学。
- 初次CPU benchmark因使用只创建一次的CSV辅助函数中断，修正增量写入后完成；两份失败日志保留。
- CPU五次交替完整step：Donkey 311.831→307.846ms，Spatial322.937→321.896ms，Transformer块3.4611→3.4470ms；变化不足以证明稳定训练收益。
- 已有Sum的CPU FP32算术模型在两CNN的256step真实输入轨迹/检查点812项、共同状态单步312项比较中全部满足原阈值；GPU仍未验收。
- 256分支合成图45.3951→29.8321ms（约52.17%吞吐变化），仅为CPU合成图证据。
- Sum/ReLU GPU、正式训练、Nsight、新调度GPU：当前NOT_RUN，等待真实空闲。

## 计划及数值边界

复用Sum103项GPU测试及其强消去原样输入，另保存三种输入族、两种keepdims的完整张量比较：
CuPy/PyTorch/NumPy/FP64/CPU FP32算术模型。超差仍FAIL，模型一致只能帮助归因，不能制造PASS。
仅正确配置进入微基准；再独立验收ReLU和基线GPU回归、两CNN相同状态及三步训练。

A/I/P/R/S/J/E/E-prefetch公平对照：同GPU0、4H真实数据和顺序、初始权重、FP32、Batch4、
SGD.001、seed42、Dropout/TF32关闭；10预热后重置、256step、三次独立重复轮换。
compute、io、同步phases分开，Nsight在正式吞吐后独立采集。
Sum/ReLU按实际数值资格决定S/J组成，没有正确性资格的配置不生成性能通过结论。
只准备一个新heap调度原型，不新增其他CUDA Kernel。

[源码分析](source-analysis.md)、[候选取舍](optimization-candidates.md)、
[CPU原始摘要](benchmarks/cpu-schedule-summary.json)。最终会补齐数据索引、图、窗口起止和完整报告。

新增[独立计算图拓扑补测](graph-topologies.md)：12组、24项逐元素一致，宽图收益可以复现，链式/残差无稳定收益；均为CPU结果。
新增[固定种子 Sum 条件数诊断](sum-conditioning.md)：320项FP64比较中普通输入全通过，强消去68项超差保留，不作为GPU正确性证据。

补充[CPU原始数据索引](correctness/cpu-raw-02/file-index.csv)、
[CPU长轨迹数值图](figures/cpu-numerical-01/cpu-sum-trajectory.png)、
[强消去条件数](figures/cpu-numerical-01/retained-conditioning.csv)。图和指标均不是本轮GPU数据。
