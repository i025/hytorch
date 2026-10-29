# 阶段 4D 数据索引

本轮 GPU 未执行；不存在新的 GPU 吞吐、Nsight 或 GPU 数值通过数据。[status.json](results/status.json) 明确区分完成和 NOT_RUN。

## 服务器原始数据

工作目录 `/home/xymo/projects/hytorch`，真实目录 `/srv/lab-disks/disk2/users/xymo/projects/hytorch`。

| 路径 | 内容 |
|---|---|
| /home/xymo/projects/hytorch/runs/conv_backward_fusion_build_01/ | 完整 build.log、build.json、真实 libhytorch_cuda.so；只编译/主机加载 |
| /home/xymo/projects/hytorch/runs/conv_backward_fusion_cpu_equation_01/ | 8 个 NPZ 输入/参考/抽样位置、1884 条比较 CSV、逐组误差 JSON |
| /home/xymo/projects/hytorch/runs/conv_backward_fusion_plan_01/ | 64 条待执行命令、NOT_RUN completion；UUID=UNALLOCATED |
| /home/xymo/projects/hytorch/runs/conv_backward_fusion_figures_01/ | 绘图脚本重建的 historical-design-evidence.png、cpu-equation-errors.png；初稿仍保留 Git 图目录 |

[file-index.csv](results/raw/file-index.csv) 列出前三个目录的 **15 个文件、4,986,408 字节**，其中 **6 个文本文件**复制到 Git。索引仅包含路径/大小/副本位置，没有哈希。动态库与张量 NPZ 留在服务器；另有上表单独列出的两张绘图重建 PNG。

## Git 内完整记录

- [环境原文](environment/system.txt)、[环境采集命令](environment/system.txt.command.json)。
- [GPU 占用 JSON](environment/gpu-availability-02.json)、[完整检查输出](environment/gpu-availability-02.log)。
- [提交前最后一次 GPU 检查](environment/gpu-final.log)，8 张卡均仍有计算客户端。
- [CPU 测试/编译/主机加载/方程抽样/脚本检查/归档的全部日志与 XML](tests/)，各命令记录在相邻 `.command.json`。
- [构建原始文本](results/raw/conv_backward_fusion_build_01/)。
- [CPU 逐位置原始比较和汇总](results/raw/conv_backward_fusion_cpu_equation_01/)。
- [待执行命令计划](results/raw/conv_backward_fusion_plan_01/)。
- [历史 4C 逐层核时及形状计算 CSV](results/history/historical-costs.csv)、[历史汇总](results/history/summary.json)。
- [CPU 误差图](figures/cpu-equation-errors.png)、[历史设计证据图](figures/historical-design-evidence.png)；都不是新 GPU 性能图。
- [执行/分析/绘图脚本](scripts/)、[完整命令与待执行模板](commands.md)。

所有原始输出保留。这里的历史表只读取 4C 数据，未改旧文件；当前计划不是已执行结果。

## 尚无数据的项目

本轮 microbenchmark CUDA Event/Host/Wall/分配原始 CSV、A/B/C/D/F/E 训练 CSV、GPU 数值误差、Native 实际 dispatch、Nsight `.nsys-rep`/SQLite/导出 CSV、Kernel 数量变化和 GPU 时间线均 **NOT_RUN**。七张要求的 GPU 性能图没有数据，因此未生成。后续只能在新目录运行，不覆盖这里的 CPU、编译或占用检查记录。
