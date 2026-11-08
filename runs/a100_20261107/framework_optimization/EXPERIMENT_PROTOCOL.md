# 待执行协议

B0 使用 shared/im2col/native pool/native col2im/packed dweight；SGD legacy。维持已有数值门限，不启用被拒绝的融合 SGD。

固定 seed42、256合成样本、batch4、3epochs/192steps、dropout0、SGD0.001。TF32禁用，PyTorch正常cuDNN/autograd，无compile。双方同样按step传输和保留GPU loss，在最终同步后读取轨迹。

阶段A：从真实step捕获所有Conv/Linear输入、输出、梯度布局、strides、参数、实际GEMM调用和次数；加入通用不规则形状。每组至少15次平衡交替、预热、固定输入。区分CUDA event区间（含可能的提交间隙）与CUPTI kernel累计时间。完整Python提交、同步完成、分配、copy、误差独立记录。卷积dInput/dWeight/dBias/full backward与等价Torch运算分别比。

CPU cProfile独立运行，不用其耗时做正式性能。Nsight短trace分别捕获两个框架，记录实际kernel/launch/API/NVTX/H2D/D2H；区分活动时间并集和kernel累计时间，CPU/GPU重叠不相加。按真实调用频率报告热点。

冻结阶段A数据与SHA后，按实测影响、可消除程度和风险选择1–2候选。每项有默认关闭开关，不改变梯度累积顺序/高阶语义，无跨step数值缓存。先专项正确性再计时；无收益的复杂改动不保留。

正式训练各有效配置至少10轮平衡交替；Torch至少5轮同窗口参考。报告逐对差、mean/median/sample SD/95%配对t区间。补充batch1/16和TinyNet。最终权重/梯度/loss使用既有门限比较。分析内存池指标时说明未包含库内分配；Nsight/CUDA API数据和无profiler正式训练分离。

完整host/GPU回归：不得将GPU skip计为pass。针对涉及ABI/内存路径执行适用memcheck/racecheck。确认资源状态后才开始GPU窗口；整个窗口定期记录共享任务，发生干扰不混成正式配对数据。
