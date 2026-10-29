# 补测实际命令与日志

工作目录 `/home/xymo/projects/hytorch`，实际路径 `/srv/lab-disks/disk2/users/xymo/projects/hytorch`。补测基于 `053ea938af3e280aa604c0ed6ea9a66f102ba2c5`，没有切换到其他实验分支。

以下命令于 2026-10-29 执行。所有输出采用新目录，原始历史实验未覆盖；复现时必须另换唯一目录。没有哈希、checksum 或 fingerprint 校验。Git 本身的提交标识不作为实验数值验证。

## 1. 环境与 GPU 运行

```bash
git status --short --branch
git log -1 --format='%H %s'
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu --format=csv
source docs/experiments/a100-stage2a-gemm/environment/activate.sh
python docs/experiments/donkeynet-backward-profiling/scripts/run_gpu.py \
  --gpu GPU-158426a4-6684-7863-8cbe-ed6595f97683 \
  --record runs/donkeynet_4a_supplement_monitor_01 -- \
  python docs/experiments/donkeynet-backward-profiling/scripts/run_supplement.py \
    --gpu-uuid GPU-158426a4-6684-7863-8cbe-ed6595f97683 \
    --output runs/donkeynet_4a_supplement_01
```

沙箱不暴露 `/dev/nvidia*`，环境与 GPU 主命令在实际服务器执行，用户仍为 xymo，无 sudo。完整系统查询 argv/环境/起止时间在 [server-system.txt.command.json](environment/server-system.txt.command.json)，未成功读取 GPU 的初始沙箱日志也保留。

[run_supplement.py](../scripts/run_supplement.py) 顺序执行以下任务，任一子任务失败即停止，不并行占用多卡：

1. `validate_gpu_instrumentation.py --output .../validation --device 0 --warmup 10 --steps 32 --repeats 3 --memory-stats --no-fingerprints`：先验证数值，再测驻留批次开销。
2. 既有 `run_measurements.py`：18 次未插桩训练。三后端各 3 次重复、分别 throughput/phases；`--samples 256 --batch 4 --epochs 4 --warmup 10 --seed 42 --lr 0.001 --no-monitor --no-fingerprints --save-comparison-inputs`。
3. `profile_training.py --operator-nvtx --nvtx`：9 次插桩 throughput，配置与第二项一致；HyTorch 另加 `--memory-stats`。
4. `nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none`：六份 Trace；三后端分别只加 phase NVTX/增加完整算子插桩，`--samples 64 --epochs 1 --batch 4 --warmup 10 --mode throughput`。
5. 每份 `nsys export --type sqlite` 导出到新 SQLite。

每个完整 argv、时间、退出码及日志路径逐项保存在：

- [raw/gpu-run/commands.json](raw/gpu-run/commands.json)：主脚本 23 个子命令。
- [raw/gpu-run/measurements/commands.json](raw/gpu-run/measurements/commands.json)：内层 18 个训练命令。
- [raw/monitor/launch.json](raw/monitor/launch.json)：物理 GPU、逻辑映射、环境与主命令。
- [raw/monitor/terminal.log](raw/monitor/terminal.log)、[completion.json](raw/monitor/completion.json)：完整主终端与完成状态。

所有子进程均返回 0。GPU 保护期间没有外部占用中断。训练与 Trace 分开执行，NVTX 标签没有在微算子之间加入同步。

## 2. CPU 回归与离线分析

以下均无需 GPU。各执行的完整 stdout/stderr 和参数另存日志，不能用控制台摘要替代。

```bash
source docs/experiments/a100-stage2a-gemm/environment/activate.sh
python docs/experiments/a100-stage1/run_logged.py \
  --log docs/experiments/donkeynet-backward-profiling/supplement-01/logs/instrumentation-cpu-tests.log -- \
  python -m pytest docs/experiments/donkeynet-backward-profiling/scripts/test_profile_training.py -q \
    --junitxml=docs/experiments/donkeynet-backward-profiling/supplement-01/logs/instrumentation-cpu-tests.xml
python docs/experiments/donkeynet-backward-profiling/scripts/supplement_analyze.py \
  --input runs/donkeynet_4a_supplement_01 \
  --output runs/donkeynet_4a_supplement_analysis_01
```

实际离线入口参数以 [raw/trace-analysis/commands.json](raw/trace-analysis/commands.json) 及其 `*-command.json` 为准，包含六次 `analyze_trace.py`、两次逐层归因、首步/后 15 步分析、不重计热点与传输统计。解析器 CPU 合成验证两份记录也归档在 [raw](raw/)，测试覆盖异步越界、嵌套 API、跨线程、Copy/Memset 与逐层父节点；没有重跑或修改旧历史结果。

正式训练汇总、独立 SQLite 审查与绘图使用：

- [summarize_supplement_training.py](../scripts/summarize_supplement_training.py)：完整实际 argv 在 [training-summary.log.command.json](logs/training-summary.log.command.json)。
- [review_supplement_traces.py](../scripts/review_supplement_traces.py)：日志与 argv 在 [logs](logs/)，独立于原分析器重算归因。
- [plot_supplement_traces.py](../scripts/plot_supplement_traces.py)：最终命令见 [trace-plots-02.log.command.json](logs/trace-plots-02.log.command.json)。第一次绘图的共享刻度显示问题仅影响图形标签；初稿留 `runs/donkeynet_4a_supplement_plot_draft_01/`，原 CSV 未改。

图表直接读取实际 CSV/JSON，Matplotlib 缓存位于忽略的项目缓存或 `runs/`，没有修改系统字体/环境。

## 3. 归档与提交

```bash
source docs/experiments/a100-stage2a-gemm/environment/activate.sh
python docs/experiments/a100-stage1/run_logged.py \
  --log docs/experiments/donkeynet-backward-profiling/supplement-01/logs/archive.log -- \
  python docs/experiments/donkeynet-backward-profiling/scripts/archive_supplement.py \
    --raw runs/donkeynet_4a_supplement_01 \
    --monitor runs/donkeynet_4a_supplement_monitor_01 \
    --analysis runs/donkeynet_4a_supplement_analysis_01 \
    --output docs/experiments/donkeynet-backward-profiling/supplement-01
```

归档只将适合 Git 的文本逐字节复制并比较，不做哈希；Nsight/SQLite/NPZ 保留服务器，记录完整路径与字节数。最终交付文件索引用路径与字节数生成。提交前检查链接、JSON、脚本语法、变化范围、未上传二进制/权重/凭据，保留原始 CSV 换行和日志内容。

```bash
python docs/experiments/donkeynet-backward-profiling/scripts/finalize_supplement.py
git add docs/experiments/donkeynet-backward-profiling
git commit -m "Complete DonkeyNet GPU profiling supplement and layer attribution"
git push origin experiment/donkeynet-backward-profiling
```

普通 push，不 force、不合并 main。实际提交 ID 由 Git 历史和最终交付给出，避免文档引用自己的循环提交标识。推送后停止，不开始 4B。
