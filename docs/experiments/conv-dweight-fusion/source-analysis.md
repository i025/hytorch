# dWeight 调用路径与支持边界

`Conv2d.backward → Conv2DGradW(self)(x,gy) → forward`。原CPU路径仍是矩阵化Im2col及CPU GEMM；原GPU路径生成 `[N,C,Kh,Kw,OH,OW]` col，再 `cupy.tensordot(gy,col,((0,2,3),(0,4,5)))`。轴置换本身可为view，随后的矩阵reshape/布局物化才触发Copy，不能将每次transpose都算成Copy。

4A补测用API correlationId确认每层dWeight的Im2col和两次CuPy Copy。4F实际父节点再次确认4–5个Kernel，包含cuBLAS及可能的矩阵部分和后处理。Kernel累计时间与主机Conv2DGradW.forward范围分别记录，不等同于训练Step份额。原col至少写出并被矩阵准备读取，物化可能再写/读同尺寸缓冲；这是逻辑流量下界，不是实测DRAM带宽。

新 `conv_dweight.cu` 仅实现一个split-K tiled implicit GEMM算法，两阶段：`conv_dweight_tiles`生成部分和，`conv_dweight_reduce`确定性归约。线性block索引展开split、Cout tile和Cin/Kh/Kw tile。所有地址依赖通用几何，无网络名、层号或固定Tensor shape。16×16 tile和128项split是固定算法参数，不是形状表。

当前ABI为4个设备指针、13个几何整数和当前Stream指针；Runtime检查连续FP32同设备NCHW、输出几何和工作区INT_MAX限制，不隐式物化非连续输入、不静默改选原路径。输出OIHW，groups1/dilation1。函数使用FP32 FMA单次舍入、显式FP32加法，不用Tensor Core、TF32、FP64、原子或cuDNN。归约分组改变FP32舍入顺序，沿用既有atol1e-5/rtol1e-4实测。

`backend(conv_dweight="fused")`显式启用，训练CLI `--conv-dweight fused`。默认original。`Conv2DGradW.forward`仅在 `Config.enable_backprop=False` 的普通反向调度native，创建高阶图则保留既有可微路径。未改Conv2DGradW.backward数学、dInput、Forward、dBias、SGD或4E选择器。

GPU生命周期先完成CPU准备、SM80编译和全命令计划，再选择一次空闲UUID；统一入口保持正常初始化的控制器context跨子任务存活，不使用忙循环或占满显存模拟租约。独立测试/Nsight子进程继承同一CUDA_VISIBLE_DEVICES。所有GPU任务完成或失败后deviceReset并记录释放时间；随后CUDA_VISIBLE_DEVICES设为空执行导出、统计、绘图、Git操作。一次窗口中断不换卡续跑。

实际首次窗口因测试切片断言与释放接口问题提前结束，未进行性能任务。两处脚本在CPU上修正后，用户明确授权一次同GPU0完整补测；第二窗口连续完成全部76项任务，`cudaDeviceReset`返回0。阶段实际发生两个窗口，完整验收只引用第二窗口，首次失败日志保留，详见README及`results/window-01-failure.json`。
