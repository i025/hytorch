# 实际时间线（持续更新）

对外时间为北京时间UTC+8，原始日志保留UTC，不能把两者混用。

- 2026-11-06 00:18:50：开始任务；截止07:18:50，GPU计算截止06:48:50。
- 00:19左右：物理GPU0和历史UUID一致；PID13632占29.5GB，utilization虽为0仍忙。
- 00:21:27：独立tmux监控开始，PID17169，约60秒一次。此前是交互确认与脚本准备，不称连续监控。
- 00:26：SM80现有库离线编译完成，无GPU计算。
- 00:26:54：CPU两CNN/两分支图cProfile完成；选择单个heap调度原型，不新增CUDA Kernel。
- 00:34–00:37：初次新增测试2失败/12通过（FP64字面量参考问题），修正FP32参考后14通过；
  全主机回归210通过、0失败、0跳过。
- 00:38:54：五次交替CPU调度基准和188项CNN/Transformer数值比较完成，全数值逐元素一致。
  初次CSV创建模式错误导致的中断数据保留。该批前段与主机回归共享CPU，微小CNN差异不作收益声明。
- 约00:44：Sum/ReLU/heap共126项GPU测试仅collect，未初始化设备。
- 环境元数据辅助命令首次误用系统Python，缺少numpy包信息；改用既有.venv，没有安装任何依赖。

后续监控逐条见最终`gpu-monitor.csv`；准备阶段快照见`environment/gpu-monitor-preparation.csv`。
若出现空闲窗口，实际起止和每项任务命令由`runs/gpu0_autonomous_7h_01/gpu-window-01/window.json`记录。
当前GPU结果仍NOT_RUN，不能预写已完成。
