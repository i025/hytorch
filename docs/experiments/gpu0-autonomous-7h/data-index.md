# 原始数据索引（准备阶段）

服务器根目录：`/home/xymo/projects/hytorch/runs/gpu0_autonomous_7h_01/`。

- [CPU原始记录及二进制索引](correctness/cpu-raw-01/file-index.csv)：cProfile全函数CSV、五次调度基准原始耗时/正确性、完整构建记录。
- 服务器pstats、完整输入/梯度/权重NPZ、动态库保留对应目录，不上传GitHub。
- [CPU调度摘要](benchmarks/cpu-schedule-summary.json)：含每次原始ms及SD，不只百分比。
- [210项主机XML](correctness/regression-01.xml)、[新增14项XML](correctness/schedule-host-02.xml)、初始失败XML及[完整日志](logs)。
- [调度/截止时间](environment/schedule.json)、[GPU准备入口](environment/gpu-plan.json)、[环境](environment.md)。
- 实时监控原文件`docs/experiments/gpu0-autonomous-7h/gpu-monitor.csv`在任务完成前持续增长；
  阶段性提交仅保存[准备时快照](environment/gpu-monitor-preparation.csv)，最终归档完整CSV。
- GPU窗口尚未执行；未来原始路径为`runs/gpu0_autonomous_7h_01/gpu-window-01/`，
  未生成的GPU文件不提前列为已完成。

最终索引将按实际生成文件补齐。PNG只能由真实CSV/JSON产生，不冒充终端截图。
