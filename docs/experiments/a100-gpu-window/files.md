# 全部文件、命令与服务器数据索引

项目逻辑路径 `/home/xymo/projects/hytorch`，实际路径 `/srv/lab-disks/disk2/users/xymo/projects/hytorch`。

[逐文件公开索引 files.csv](files.csv)：每个文件的仓库路径、服务器绝对路径、字节数和 GitHub 链接，覆盖本轮 stage2a、stage2b、gpu-window 的所有数据/图/源码。
[完整服务器原始文件清单](raw-server-files.json)：包含未上传的初始权重、数据数组、最终权重、预训练权重、Nsight 原始报告与全部日志的具体绝对路径。仅列路径和大小，不做哈希；权重/样本/二进制留服务器。

| 阶段 | 公开数据目录 | 服务器原始目录（项目根下） | 命令与时间证据 |
|---|---|---|---|
| 环境与八小时上限监控 | environment/ | runs/gpu_window_watch_01 至 _04 | config/result/observations、安装 *.log.command.json |
| 主机对齐 | pytorch-parity/ | 同公开目录；runs/host_after_dtype_fix_01.* | pytest 日志、XML、numeric-comparisons-01/02.jsonl |
| GEMM | ../a100-stage2a-gemm/formal-01、pilot-03、failed-cupy-api-02 | runs/gemm_stage2a_01、gemm_stage2a_pilot_02/03、gpu_window_gemm_* | monitor-*/launch.json、completion.json、benchmark.log |
| Ops | ../a100-stage2b-ops/results-01、monitor-01 | runs/ops_stage2b_01、gpu_window_ops_monitor_01 | ops.csv、完整监控启动信息 |
| Torch GEMM / 误差诊断 | analysis/torch-gemm-*、gemm-diagnosis-* | runs/torch_gemm_01、gemm_diagnosis_01、window_*_monitor_01 | monitor launch/completion 与 FP64/FMA 数值 |
| dtype 修复 | analysis/dtype-*、validation-after-dtype-fix-* | runs/dtype_before_02、dtype_after_01、validation_dtype_fix_01 | 原始失败与成功断言、GPU tests.xml |
| 初始训练（存在反向 FP64） | training-comparison/*throughput-01、*before-fix* | runs/synthetic_{tiny,donkey}_comparison_01、synthetic_{tiny,donkey}_phases_01 | commands.json、config/summary/steps/losses/hardware、每次 .log |
| 最终 FP32 训练 | training-comparison/*fixed* | runs/synthetic_{tiny,donkey}_{throughput,phases}_fixed_01 | 同上；每组初始与最终参数/输入 NPZ 留服务器 |
| Nsight/Sanitizer | profiling/ | runs/nsys_window_01.nsys-rep、nsys_dtype_fixed_01.nsys-rep 及对应 .sqlite；runs/gpu_memcheck_window_01；runs/window_ncu_* | CSV 汇总、间隔 CSV、监控日志、XML |
| 预训练 | pretrained/{alexnet,vgg11,vgg16} 及 -monitor | runs/pretrained_{alexnet,vgg11,vgg16}_01、.venv/torch-cache/hub/checkpoints | weights.json、全部 logits.csv、完整命令与日志 |

命令 journal：environment/remaining_stages_01.jsonl、dtype_retest_stages_01.jsonl。每轮实验的实际命令是 launch.json/commands.json/command.json 中的记录；重跑应换唯一输出目录。

## 关键图

- [GEMM 延迟](../a100-stage2a-gemm/figures-formal-01/gemm-latency.png)、[vs CuPy](../a100-stage2a-gemm/figures-formal-01/speedup-vs-cupy.png)、[GFLOPS](../a100-stage2a-gemm/figures-formal-01/gemm-gflops.png)、[Tile](../a100-stage2a-gemm/figures-formal-01/tile-comparison.png)、[传输/计算](../a100-stage2a-gemm/figures-formal-01/transfer-vs-compute.png)、[逐次变异](../a100-stage2a-gemm/figures-formal-01/latency-variability.png)。失败数值标红叉。
- [Conv 延迟](../a100-stage2b-ops/figures/conv-latency.png)、[Conv 加速比](../a100-stage2b-ops/figures/conv-speedup.png)、[Direct/Im2col](../a100-stage2b-ops/figures/direct-vs-im2col.png)、[Pooling](../a100-stage2b-ops/figures/pooling.png)、[Tile](../a100-stage2b-ops/figures/conv-tile-size.png)。
- [TinyNet Loss](training-comparison/tiny-throughput-fixed-details-01/loss.png)、[DonkeyNet Loss](training-comparison/donkey-throughput-fixed-details-01/loss.png)、[DonkeyNet 分段时间](training-comparison/donkey-phases-fixed-details-01/phases.png)。
- [修复后实际时间线](profiling/nsys-analysis-after-fix/timeline.png)、[主机数值误差](figures/host-parity-after-fix.png)。其他每次训练硬件/step 图均在 files.csv。

所有图片由原始日志或数值生成，并非终端截图。绘图源码：stage2a/scripts/plot_gemm.py；本目录 scripts/plot_ops.py、report_parity*.py、training_details.py、nsys_analysis.py；仓库 scripts/report_cuda.py。原始 GPU 监控留服务器，公开副本只去除 PID 等进程身份，保留整卡指标与进程数量。

完整阶段提交见 [commits.md](commits.md)。目录中 `README-progress-initial.md`、`README-before-fix.md` 明确是历史进度/修复前报告，不代表当前验收状态。
