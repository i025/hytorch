# 环境与资源约束

Ubuntu20.04.6，双EPYC7763，Python3.8.10；NumPy1.24.4、CuPy12.3.0、PyTorch2.4.1+cu121，
Toolkit/nvcc12.2.140、GCC9.4。使用项目既有`.venv`，未安装新依赖。
当前驱动只读记录见[driver.txt](environment/driver.txt)，GPU Runtime/compute capability仅在获准空闲窗口采集。

实验进程固定CPU0–3，BLAS/OMP/MKL1线程；PyTorch设置4线程。避免占用共享服务器大量CPU。
CPU模式CUDA_VISIBLE_DEVICES为空；GPU模式固定物理0的确认UUID，逻辑编号0。
没有sudo、Reset、改时钟/功耗/MIG/MPS、显存占位或GPU忙循环。

没有检测到squeue/qstat/bjobs命令。用户明确授权空闲GPU0，观察空闲不代表独占租约；
有外部compute进程、低剩余显存或OOM时停止本轮GPU子进程，不干预其他任务，不换设备。
空闲基线band≤128MiB，util≤2%，无compute进程，连续3次60秒采样。
监控本身不创建设备Context；GPU任务由独立子进程顺序完成，同一窗口，结束后统一结束所有子进程。

截止为UTC2026-11-05 23:18:50（北京时间11-06 07:18:50）；GPU截止提前30分钟。
完整配置见[schedule.json](environment/schedule.json)。
