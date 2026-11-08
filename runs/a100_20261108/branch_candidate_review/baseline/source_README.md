# GPU0自主性能研究：最终报告

任务预算北京时间2026-11-06 00:18:50–07:18:50。GPU计算截止06:48:50；
最后5分钟停止监控、归档和提交。实际监控6.8583小时、412条，
从2026-11-06T00:21:27.501674+08:00到2026-11-06T07:12:57.478151+08:00，最大采样间隔60.101543秒。
没有将七小时预算写成七小时实测监控，准备时段也没有冒充持续监测。

## GPU生命周期与实验状态

物理GPU0，UUID `GPU-f099fdbd-adf2-6157-3462-4145bc2a8359`。最高连续空闲计数1；
满足连续空闲要求的采样0条，其中GPU截止前0条。
利用率为0但外部计算进程仍驻留的采样138条，未据此抢占。
窗口状态 **NOT_RUN**。所有GPU实验只允许这一个UUID，不换卡，不使用其他七张GPU。
没有独占租约，外部任务/资源冲突即停。窗口未执行时，不适用“完成同窗口实验/释放Context”，
因为本轮没有创建GPU计算Context；若已执行，窗口起止、子进程退出和release记录在汇总JSON与原始window.json。
本轮监控及GPU控制进程已按预算停止，停止记录见服务器monitor-end.json。

| 项目 | 实际状态 |
|---|---|
| Native Sum GPU测试 | NOT_RUN |
| Native ReLU Backward GPU测试 | NOT_RUN |
| 已有原生GPU回归 | NOT_RUN |
| 新heap调度GPU | NOT_RUN |
| 未插桩训练运行记录 | 0次；不足三次的组标PARTIAL，不计算正式收益 |

Sum超差样本即使能由FP32顺序解释，也保留FAIL；完整CuPy/PyTorch/FP64/CPU模型误差若执行则见原始sum-diagnosis。
只对通过配置测性能，ReLU独立验收，S/J还需真实数据共同状态检查。没有放宽数值阈值。
纯Kernel时间来自API correlationId与同线程NVTX，不把Event span、提交API时间或覆盖率称为Kernel核时/GPU利用率。

## CPU研究与唯一新原型

最终主机集合：{'status': 'PASS', 'passed': 219, 'tests': 219, 'failures': 0, 'errors': 0, 'skipped': 0}，来源`/srv/lab-disks/disk2/users/xymo/projects/hytorch/docs/experiments/gpu0-autonomous-7h/correctness/regression-final-03.xml`。
此前210/213项及定向集合各次记录均保留，不能相加重复计数。
归档状态预检：{'status': 'PASS', 'passed': 3, 'tests': 3, 'failures': 0, 'errors': 0, 'skipped': 0}；内存回调/注册生命周期预检：{'status': 'PASS', 'passed': 3, 'tests': 3, 'failures': 0, 'errors': 0, 'skipped': 0}。
这些预检已纳入最终合并集合，不重复计数，不代表GPU已执行。
heap调度在CNN/Transformer的188项比较逐元素一致，保持generation优先级和相同LIFO平局顺序；默认list。
五次交替CPU计时中，256分支合成图吞吐均值约提高52.17%，CNN只变化0.3%–1.3%，不宣称稳定CNN收益。
该批前段与回归共享CPU，微小差异不能作为收益证据。初始测试参考dtype和CSV脚本失败保留，生产数学没有因此改变。
后期合并回归有一次因命令漏设CUDA_PATH而218通过/1失败；补齐既有12.2路径后重跑，失败与重跑日志均保留。
独立拓扑补测12组、24项Loss/输入梯度比较逐元素一致：宽64/256的树形与扇出图CPU吞吐提高约10%–55%，
ready frontier仅1–2的链式/残差图变化约-1.53%到+1.22%，不能宣称稳定收益。
这说明收益取决于待调度队列宽度；图节点数本身不足以预测收益。详见[拓扑研究](graph-topologies.md)。
独立CPU tracemalloc预热后36次记录中，宽度256的树形/扇出图峰值增加约22.3–22.8KiB；不是GPU显存。
原生Sum的CPU算术模型真实图像三步训练156项、逐归约192项比较通过；仍不是GPU验收。
256step CPU Sum算术诊断：{'status': 'DIAGNOSIS_COMPLETED', 'gpu': 'NOT_RUN', 'steps_per_model_per_mode': 256, 'trajectory_failed': 0, 'replay_failed': 0, 'trajectory_comparisons': 812, 'replay_comparisons': 312, 'gpu_priority_pause_s': 0.04330804059281945, 'first_loss_failure': {'donkey': None, 'spatial': None}, 'finished_utc': '2026-11-05T17:44:00.076716+00:00', 'scope': 'CPU mathematical simulation of existing Sum; not a CUDA execution or performance benchmark'}。
其CPU长轨迹和共同状态比较不作为CUDA执行或加速证据。
另有[固定种子条件数诊断](sum-conditioning.md)：320项FP64比较，普通输入160项全通过，
强消去输入68项超差原样保留（CPU模型31项、NumPy37项）；不能据此豁免GPU数值失败。
主机Sum轴规划约4–14µs，配置上下文约3.48µs；已避免基准给native每次重复计入配置成本。
纯主机ABI版本调用约0.135µs，不等于CUDA提交耗时。

没有新增CUDA Kernel。本轮只有一个新的通用heap调度原型，保持已有GPU Kernel/优化器/数据语义。
继续依赖CuPy的MatMul梯度、BMM、Norm/Softmax、布局物化等已列候选；没有同时开发多个未经验证的算子。

## 端到端、累计收益与PyTorch差距

若窗口执行且组完整，本轮匹配的I/A、P/I、R/I、S/I、J/I、J/A、J/E、J/E-prefetch结果见
[逐次训练与统计](benchmarks/final-20261105T231350/)。所有原始值、SD和不同计时口径分开；
吞吐变化、耗时下降与加速比各列报告，不跨历史窗口拼接。
若GPU未执行，则异步预取的真实GPU训练收益、Sum/ReLU端到端收益、与PyTorch差距变化均 **NOT_RUN**。
CPU数据准备或合成图加速不能替代上述结论。Loss下降不证明自动驾驶能力。

## 负结果、未完成项和下一轮

Sum长归约强消去的已有数值负样本保留；原型不能因CPU模型一致而宣称通用精度全部通过。
GPU没有空闲或时间不足的任务保持NOT_RUN/PARTIAL，其他任务重新占用则保留已完成部分并停止。
heap对普通短链CNN缺乏稳定收益证据，不改默认；额外显存/Kernel数量不因此减少。
下一轮优先：已有Sum/ReLU未完成的真实GPU验收；再依据新Trace决定MSE/激活链融合；
若I/O占主导才研究pinned H2D双缓冲。仅靠CPU不能确定应重写哪个CUDA Kernel。

## 交付

[时间线](timeline.md)、[GPU监控CSV](gpu-monitor.csv)、[源码分析](source-analysis.md)、
[候选依据](optimization-candidates.md)、[环境](environment.md)、[全部命令](commands.md)、
[本轮汇总](benchmarks/final-20261105T231350/report.md)、[完整数据索引](data-index.md)、[全部图](figures/final-20261105T231350/)。
原始数据目录`/srv/lab-disks/disk2/users/xymo/projects/hytorch/runs/gpu0_autonomous_7h_01`；Nsight/SQLite、完整数组和动态库留服务器，适合Git的CSV/JSON/XML/日志和PNG上传。
图像均为实际记录可视化，不冒充截图。
实验分支`experiment/gpu0-autonomous-7h`。本轮停止，不开始下一阶段。
