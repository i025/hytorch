# 本轮准备记录：尚未完成实验

当前没有空闲且授权的 A100，阶段 A GPU 对比未运行，阶段 B 尚未实施。此包是准备与阻塞记录，不是最终实验交付。历史 0.820923s / 0.307929s 不作为本轮基线或加速比分母。

## 已完成

保存 HEAD、Git 状态、全部 tracked 未提交 diff、untracked 清单，以及 82 个源码/测试/构建文件与 SHA256。baseline/source 包含实际 CUDA 动态库与编译记录。版本来自当前虚拟环境的包元数据；当前 cuDNN runtime 版本待空闲 GPU 窗口采集。

执行 tests/test_cuda_host.py、tests/test_col2im_host.py、tests/test_conv2d_gemm_cpu.py：19 passed。仅代表这三个文件，不代表完整 host/GPU 回归通过。

未修改生产源码、数值门限、默认配置或已有实验记录；没有提交、推送或创建 PR。

## 资源阻塞

只读查询发现 GPU1 上 PID1290426 正在运行 GRCache extract_v9.py --panel panel_90_150。连续查询显存从2419MiB增长至3463MiB，最后3445MiB，GPU利用率最高观察到38%。即使瞬间利用率为0%，也不把已占用设备当作空闲。
GPU0 为 VLLM PID1115402，占用20543MiB。未终止、暂停或影响这些进程，没有启动本任务 GPU 工作负载。

## 源码检查所得候选（未经性能验证）

- hytorch/core.py Variable.backward：每次 add_func 排序，以及逐节点上下文管理器；必须保持 generation 和同代 tie 顺序，不能跳过梯度节点。待 cProfile 与无 profiler 完整 backward 配对测量。
- hytorch/core.py Function.__call__：输入包装、输出 Variable 与 weakref 创建。no_grad 下仍记录 inputs/outputs；自定义 Function 可能依赖这些成员，不能盲删。
- hytorch/backends/cuda/runtime.py current_config 返回 dict 副本；selected、布局校验与 ctypes 参数构造需要分项计时。任何缓存不得保存指针或 stream。
- hytorch/dataloaders.py 同步路径逐样本组装 NumPy batch，再由训练循环 H2D；尚无本轮证据证明它占主要耗时。预加载不能只施于优化组。

这些候选没有按实际训练收益排序，尚不满足生产优化的证据门槛。

## 未运行

实际 workload instrumentation、15次逐算子对比、Nsight 时间线、候选优化、完整回归、Sanitizer、10轮训练消融、PyTorch5轮参考、TinyNet/batch补充、显存实验和 PPT 图表均未运行。没有生成伪造/空白性能CSV来充当交付。

待 GPU1 空闲后继续同一新窗口采集 B0 和所有候选数据，不能拼接历史窗口计算配对收益。阶段A测量冻结之后再决定阶段B的1–2项修改。
