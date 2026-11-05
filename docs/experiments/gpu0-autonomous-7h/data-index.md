# 最终数据索引

服务器完整根目录：`/srv/lab-disks/disk2/users/xymo/projects/hytorch/runs/gpu0_autonomous_7h_01`。

- [完整逐文件索引](correctness/final-raw-231351/file-index.csv)：全部CPU/GPU原始目录、服务器二进制路径、Git可移植副本。
- [最终分析](benchmarks/final-20261105T231350/)、[最终图](figures/final-20261105T231350/)、[实时监控完整CSV](gpu-monitor.csv)。
- [CPU初始原始索引](correctness/cpu-raw-01/file-index.csv)、[日志](logs)、[环境](environment)、[测试XML](correctness)。
- GPU若执行，原始`gpu-window-01`含完整window.json、每次命令和日志、测试数组、训练权重、逐step Loss/耗时、Nsight。
- Nsight的`.nsys-rep`/SQLite及所有NPZ/动态库只留服务器，索引不计算哈希。
- `monitor-end.json`和`publication-state.json`在服务器根目录，后者记录最终commit及推送状态。

未执行的文件不提前宣称已生成。原始失败和成功记录均保留，GPU未执行不填造性能CSV或图。
