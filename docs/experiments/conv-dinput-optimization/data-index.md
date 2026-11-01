# 4F 原始数据与图像索引

仓库服务器路径 `/home/xymo/projects/hytorch`，真实路径 `/srv/lab-disks/disk2/users/xymo/projects/hytorch`。以下 `runs/` 均相对此路径。原始stdout/stderr、失败输入、输出张量、权重、Nsight报告/SQLite全部保留服务器；Git只上传便于审查的文本、CSV、JSON、XML、日志和PNG。没有哈希、checksum或fingerprint清单。

## 可直接审查的数据

| 内容 | Git文件/目录 |
|---|---|
| 环境和完整Python包列表 | [environment/system-info.log](environment/system-info.log)及同名command.json |
| GPU窗口、UUID、占用与采样峰值 | 最终[results/hardware-02/windows.json](results/hardware-02/windows.json)、[子命令采样区间](results/hardware-02/commands.csv)及各原始guard记录；早期environment和hardware-01快照保留 |
| 67项主机回归 | [logs/host-tests-01.log](logs/host-tests-01.log)、[results/host-tests-01.xml](results/host-tests-01.xml) |
| 最终446项GPU状态 | [results/gpu-correctness-03.json](results/gpu-correctness-03.json)、[完整XML](raw-records/initial-01/conv_dinput_opt_correctness_03/results/gpu-tests.xml) |
| 全部GPU误差 | [shape-correctness](raw-records/initial-01/conv_dinput_opt_correctness_03/results/shape-correctness/)、[pruning](raw-records/initial-01/conv_dinput_opt_correctness_03/results/pruning-correctness/)、[旧Gather](raw-records/initial-01/conv_dinput_opt_correctness_03/results/legacy-correctness/)、[SGD](raw-records/initial-01/conv_dinput_opt_correctness_03/results/sgd-correctness/) |
| 新旧SM80编译与资源 | [build目录](raw-records/initial-01/conv_dinput_opt_build_01/)、[kernel-resources.csv](results/kernel-resources.csv)、[cuobjdump日志](logs/resources-01.log) |
| 预声明124形状 | [results/predeclared-shapes.json](results/predeclared-shapes.json)；历史线索[history-shapes.csv](results/history-shapes.csv) |
| 每次微基准耗时 | [results/micro-01/latencies.csv](results/micro-01/latencies.csv)；66,960行正式计时 |
| 微基准均值/SD/Median/P95 | [process-summary.csv](results/micro-01/process-summary.csv)、[summary.csv](results/micro-01/summary.csv) |
| 完整操作比较/内存/误差 | [comparisons.csv](results/micro-01/comparisons.csv)、[memory.csv](results/micro-01/memory.csv)、[errors.csv](results/micro-01/errors.csv) |
| 首轮全部训练 | [results/training-01/](results/training-01/)；含嘈杂数据，不丢弃 |
| 第二匹配窗口训练 | [results/training-02/](results/training-02/)；raw、steps、losses、errors、summary、speedups和长期progress-table |
| 为什么整体重测 | [results/repeat-training-protocol.json](results/repeat-training-protocol.json) |
| 26份完整Nsight分析 | [results/profiles-01/](results/profiles-01/)；完整三次纯Kernel重复；GPU1窗口 |
| 纯Kernel每次/每进程/各形状 | [micro-kernels.csv](results/profiles-01/micro-kernels.csv)、[pure-kernel-processes.csv](results/profiles-01/pure-kernel-processes.csv)、[pure-kernel-comparisons.csv](results/profiles-01/pure-kernel-comparisons.csv) |
| NVTX父节点/Kernel/API/内存 | [trace-phases.csv](results/profiles-01/trace-phases.csv)、[gradient-components.csv](results/profiles-01/gradient-components.csv)、[allocation-requests.csv](results/profiles-01/allocation-requests.csv)、[attribution-checks.json](results/profiles-01/attribution-checks.json) |
| 全124形状跨口径索引及每step归一值 | [results/report-01/](results/report-01/)；不把不同窗口计时相减 |
| 全部顶层命令与日志 | [commands.md](commands.md)、[logs/](logs/)；通常log旁有command.json；wait06外层143及内部成功的独立证据见[wrapper-exit-06.json](logs/wrapper-exit-06.json) |
| 最终文件与数值产物检查 | [artifact-review-01.json](results/artifact-review-01.json)；测试数量、26份Trace、三重复、文件索引、PNG和本地链接均检查通过 |

## 服务器原始记录

首批归档[逐文件清单 file-index.csv](raw-records/initial-01/file-index.csv)索引 **26,201个原始文件**，复制 **1,565个便携文本记录**。每行包含绝对服务器路径、字节数、Git对应路径和存储方式。超过1MB的原始文本按原字节内容gzip保存，可用 `gzip -dc FILE.gz` 查看，未转写或过滤stdout/stderr；大二进制仅索引，不上传。

最终成功采集另有[2,745文件/195个便携记录](raw-records/successful-traces-01/file-index.csv)，完整解析另有[367文件/367个便携记录](raw-records/trace-analysis-01/file-index.csv)。三批合计 **29,313个服务器原始文件、2,127个Git便携记录**；不含本目录直接保存的汇总、图像和顶层日志。成功采集完整stdout/stderr在[training](raw-records/successful-traces-01/conv_dinput_opt_traces_wait_06/attempt-01/results/training/)和[micro](raw-records/successful-traces-01/conv_dinput_opt_traces_wait_06/attempt-01/results/micro/)，完整事件级CSV在[trace-analysis-01](raw-records/trace-analysis-01/conv_dinput_opt_trace_analysis_01/)。

| 服务器原始目录（runs/下） | 内容和实际状态 |
|---|---|
| `conv_dinput_opt_build_01/` | build.log、build.json、libhytorch_cuda.so；成功 |
| `conv_dinput_opt_host_equations_01/` | CPU函数体编译、抽样CSV与NPZ；通过 |
| `conv_dinput_opt_correctness_wait_01/attempt-01/` | 首轮52失败/394通过、完整失败输入和输出 |
| `conv_dinput_opt_forward_diagnosis_01/attempt-01/results/` | c23/v19 Forward的CPU FP64/CuPy/原生Torch/cuDNN诊断 |
| `conv_dinput_opt_correctness_wait_02/attempt-01/` | GPU0被外部任务重新占用，中断，非通过 |
| `conv_dinput_opt_correctness_03/` | GPU2成功446项，所有原始误差/张量/guard/XML |
| `conv_dinput_opt_micro_01/results/` | GPU2、36个微基准进程，原始输入/输出、各次latencies/errors/memory |
| `conv_dinput_opt_training_01/results/` | GPU2首窗口60进程、每stepJSONL、输入/初始及最终权重 |
| `conv_dinput_opt_training_02/results/` | GPU2第二窗口60进程，和首轮独立保留 |
| `conv_dinput_opt_traces_01/training/` | GPU2训练A/F/H两个模型，共6份完整nsys-rep/SQLite |
| `conv_dinput_opt_traces_01/micro/` | GPU2全形状Trace；早期辅助分析使用22份完整配对导出；未配对第三重复不纳入 |
| `conv_dinput_opt_trace_analysis_partial_01/` | 22份导出的逐Kernel、NVTX、CUDA API、Memcpy事件CSV和解析日志 |
| `conv_dinput_opt_traces_02/` | GPU2后续采集被真实外部进程中断，保留全部中间结果 |
| `conv_dinput_opt_traces_wait_03/` | GPU6窗口被外部任务重新占用；中断 |
| `conv_dinput_opt_traces_wait_04/` | GPU0采集因guard归属判断中断；不计正式结果 |
| `conv_dinput_opt_traces_05/` | GPU0采集，随后其他用户进程进入，guard停止本次任务 |
| `conv_dinput_opt_traces_wait_06/attempt-01/results/training/` | 最终GPU1完整6份训练nsys-rep/SQLite，A/F/H×两个模型 |
| `conv_dinput_opt_traces_wait_06/attempt-01/results/micro/` | 最终GPU1完整20份微基准nsys-rep/SQLite，含新旧各三次纯Kernel重复 |
| `conv_dinput_opt_trace_analysis_01/` | 最终26份逐Kernel、API、NVTX、Memcpy事件CSV和完整解析日志 |

所有上述根目录的guard/before.json、launch.json、gpu-load.jsonl、completion.json和terminal.log也在便携归档。中断后的日志可能没有子命令完成标记，不能按存在文件就判为成功。[profiles-01/raw-location.json](results/profiles-01/raw-location.json)标明报告实际解析的三条绝对路径。先前GPU2的[22份部分分析](results/profiles-partial-01/)和[对应合并表](results/report-partial-01/)完整保留，不与GPU1混算。

## 结果图

全部是实测数据可视化，非截图；源代码[plot_results.py](scripts/plot_results.py)，每张图页脚均标注来源。最终纯Kernel图使用GPU1完整三重复，未插桩计时为GPU2三重复；不跨设备相减。

1. [旧/新纯Kernel延迟](figures/final-01/01-pure-kernel-latency.png)：124形状，Nsight三重复均值±进程SD。
2. [历史/独立集合性能分布](figures/final-01/02-cross-shape-distribution.png)：纯Kernel和完整墙钟分别画CDF。
3. [全部24独立形状完整dInput墙钟](figures/final-01/03-complete-dinput-wall.png)：四方法及旧/新比值。
4. [第二窗口五版本训练吞吐](figures/final-01/04-matched-throughput.png)：两个模型，三次实际进程点及SD。
5. [首轮嘈杂窗口完整吞吐](figures/final-01/04b-first-noisy-window.png)：不删除波动和负结果。
6. [相对无Native A的累计收益](figures/final-01/05-cumulative-progress.png)：本次第二匹配窗口。
7. [与PyTorch的差距](figures/final-01/06-pytorch-gap.png)：同一窗口吞吐百分比。
8. [编译器资源](figures/final-01/07-compiler-resources.png)：40/48寄存器、0 spill；不是实测Occupancy。
9. [Backward主机/GPU时间线](figures/final-01/08-backward-timeline.png)：各模型F/H第一测量步，以correlation父节点归因；灰为主机范围，蓝为其他Kernel，橙为旧Gather，绿为新Gather。
10. [单独同步阶段耗时](figures/final-01/09-synchronized-phases.png)：不代替普通训练吞吐。
11. [各步Loss数值对照](figures/final-01/10-loss-correctness.png)：第二窗口第1重复的完整256step。
12. [训练各必要dInput的设备Kernel时间](figures/final-01/11-training-dinput-kernels.png)：Nsight各16次父节点均值。

图片可直接在GitHub目录查看，原始数字均可回溯上述CSV/JSON及服务器原始张量。没有真实桌面截图，本轮不使用日志渲染冒充截图。

早期两重复结果生成的[另12张辅助图](figures/partial-01/)保持原样，标题中的PARTIAL不是最终采集状态；最终12张图在[final-01](figures/final-01/)。两套均保留，全文结论和上方逐图链接采用完整三重复数据。
