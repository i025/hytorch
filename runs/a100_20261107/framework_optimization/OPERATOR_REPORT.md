# HyTorch vs PyTorch 算子对比

比值统一为HyTorch时间/PyTorch时间，>1表示HyTorch更慢。GPU列为各shape的纯kernel中位时间之比；类别中位数不按训练频率加权，不能直接推算网络wall。

| 类别 | shape数 | GPU时间比最小–最大 | GPU比中位数 | 完整同步调用比中位数 |
|---|---|---|---|---|
| ConvForward | 7 | 0.43–2.67 | 1.19 | 2.72 |
| dWeight | 7 | 1.16–3.84 | 1.81 | 1.57 |
| dInput | 7 | 0.08–1.20 | 0.66 | 1.38 |
| dBias | 7 | 0.40–1.74 | 0.74 | 0.61 |
| FullBackward | 7 | 0.22–1.58 | 0.91 | 2.68 |
| GEMM | 11 | 0.75–3.61 | 1.86 | 1.86 |
| Activation | 6 | 0.75–1.41 | 0.88 | 1.68 |
| Pooling | 2 | 0.92–0.95 | 0.93 | 2.94 |
| PoolBackward | 2 | 1.98–2.03 | 2.00 | 2.20 |
| Linear | 4 | 1.08–1.54 | 1.45 | 2.19 |
| SGD | 1 | 4.59–4.59 | 4.59 | 3.96 |

## 接近与落后

小激活和pool forward的kernel接近PyTorch，部分dBias/dInput更快；但完整Python调用仍经常慢约1.5–3倍。当前shared dispatch的真实前向GEMM约1.18–3.61倍于torch.matmul，packed dWeight的五层约1.16–2.02倍；SGD kernel约4.59倍，32个kernel对PyTorch的1个。两者均调用CuPy/cuBLAS或cuDNN的部分不能宣传为自研kernel优势。

conv1 forward明显落后；conv2/3在当前cuDNN算法和独立输入设置下GPU时间相近或更快，但fullcall不一定更快。这里不是cuDNN所有算法或所有shape的结论。dInput、dWeight、dBias分别让PyTorch只对相应输入requires_grad；FullBackward让x/W/b都需要梯度，且输出均匹配。真实模型PyTorch首层输入图像不需要梯度，HyTorch现有框架仍计算该dInput，因此孤立conv1 FullBackward中的Torch大耗时不能代入真实训练、不能据此宣称HyTorch训练backward更快。最初全部输入requires_grad的单项诊断已归档raw/diagnostic_all_inputs_require_grad，未使用其结果。

Linear表是完整F.linear与torch.nn.functional.linear接口；模型的Layer.forward实际使用F.linear_simple（MatMul+Add），数学等价但Python路径不同，该CPU比值不直接代表训练Linear层。实际训练调用和耗时已另由WORKLOAD_COMPLETE.json、TRAINING_HOTSPOTS.csv记录。Pooling为TinyNet/通用shape，DonkeyNet中调用频率为0。

GEMM表basic/blocked是当前实现；shared表示当前128CTA dispatch（可能选shared或register），register_old_dispatch是旧动态库ABI诊断，只有m/n/k>=64才选register，其它shape仍shared。具体kernel/grid/block/寄存器见OPERATOR_KERNELS.csv，旧库SHA在raw/old_gemm_library_sha.json，未替换生产库。

分配计数见OPERATOR_ALLOCATIONS.csv：CuPy MemoryHook请求与PyTorch allocator history，分别注明统计口径；不能将缓存allocator请求等同于cudaMalloc。Torch最初CPU profiler不暴露CUDA分配事件的零值被判为不可用、归档，正式表改用allocator history。最后allocator诊断与新外部任务最多重叠约6.7秒，只用于进程内请求数，不用于性能。copy/pack实际kernel名与全部memcpy记录保留于trace，不能把这些都算作H2D。

## 实验方法与适用范围

同一物理GPU1 A100-PCIE-40GB；GPU0未使用。Python3.10.12、NumPy2.2.6、CuPy14.2.0、PyTorch2.9.1+cu128、CuPy CUDA runtime12090、cuDNN91002，动态库nvcc12.4 -O3 -std=c++14，未启用fast-math。TF32同时关闭，PyTorch正常cuDNN/autograd，benchmark=False，不使用torch.compile；SGD foreach=None/fused=None，实际trace为一个multi_tensor kernel。

算子计时：固定FP32输入(seed42，普通算子normal std0.1)、预热10次、15次平衡正反序；单独测Python提交、同步完整调用、CUDA event区间。event区间可能含CPU提交空隙，不当作纯kernel时间。纯kernel来自另一次15重复Nsight/CUPTI捕获，每调用kernel时间求和后取中位数。GEMM双方预分配输出；其它完整接口保留真实分配/packing成本。原始samples和波动均保留。SGD各实现先重置相同参数/梯度验证，再做各自调用测量，foreach与默认明确分列。

普通算子跨框架误差诊断为atol2e-5/rtol2e-4，全部零超限；不能用这个诊断替换原始单元测试门限。原测试完全未修改；历史严格GEMM失败记录保留raw/previous_gemm_*。新缓存与B0比较要求逐元素相等，训练另按既有权重1e-4/1e-3、梯度2e-5/2e-4、loss1e-4/1e-3检查全部元素/轨迹。

训练：256合成样本、batch4、3epochs192steps、seed42、相同初始化、SGD0.001、dropout0。两个框架均按step从同一DataLoader取NumPy数据并H2D，未单边预加载。GPU loss保留至完整训练最终同步，计时结束后读取全部轨迹；未把最终同步挪出区间。正式计时无profiler；配置B0/F1/Torch的六种排列各两遍，共12轮，paired t区间df11。先前F1固定居中的10轮完整保留raw/pilot_fixed_middle_order，不用于正式收益；调整理由是位置平衡，不是择优筛选。

阶段A原始数据及SHA冻结于stage_a。后补的15重复kernel采集和allocator诊断从baseline/source加载未经本轮修改的生产代码，旧数据不覆盖。阶段B正式比较B0为本轮代码开关关闭，F1开关开启，其余相同。GPU计时和profile都在本次连续窗口；没有拼接历史0.820923s/0.307929s作为分母。

Nsight CPU phase归因：同一训练进程串行phase区间按CUDA API correlationId关联kernel，包括PyTorch autograd worker线程；仅按主线程ID会漏掉其backward，已正确跨线程匹配时间区间。GPU累计时间与区间并集分别保存；CPU elapsed包含API、等待、线程调度，不代表纯Python on-CPU。不能把这些相加。没有锁频，也没有与每个trace同步的时钟记录。
