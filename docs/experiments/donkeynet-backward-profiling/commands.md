# 阶段 4A 命令与执行状态

2026-10-29 的 GPU 补测实际命令见 [supplement-01/commands.md](supplement-01/commands.md)。下文保留上一提交的历史命令及当时状态；本次新日志独立存放。

工作目录 `/home/xymo/projects/hytorch`（实际 `/srv/lab-disks/disk2/users/xymo/projects/hytorch`）。下面区分已经执行的离线工作、被保护程序拒绝的 GPU 命令及未执行的后续采集计划。所有输出目录均为唯一目录；复现时换新名称，不覆盖原始文件。不计算哈希、checksum 或 fingerprint。

## 分支与环境：已执行

```bash
git fetch origin experiment/a100-gpu-complete
git switch -c experiment/donkeynet-backward-profiling origin/experiment/a100-gpu-complete
source docs/experiments/a100-stage2a-gemm/environment/activate.sh
```

基线 `aad4d0562fd968083a727099e478e7db64d35dd0`。环境脚本只激活现有虚拟环境、配置 Toolkit 和项目缓存，初始 `CUDA_VISIBLE_DEVICES=''`；GPU UUID 由保护包装器单独设置。原生产动态库保留，未重编译。

完整 OS/CPU、包版本、GCC、nvcc、Nsight、驱动和权限查询见 [system.txt.command.json](environment/system.txt.command.json)；八卡占用查询见 [gpu-busy.txt.command.json](environment/gpu-busy.txt.command.json)。两者使用已有 `docs/experiments/a100-stage1/run_logged.py` 保存 stdout/stderr、环境、起止时间和返回码。

GPU 保护脚本从已提交阶段 3B 的监控工具复用，仅复用单卡空闲检查与进程归属保护，不引入其 Kernel：

```bash
git show experiment/gemm-block32-tree:docs/experiments/gemm-block32-tree/scripts/run_gpu.py > docs/experiments/donkeynet-backward-profiling/scripts/run_gpu.py
```

## 新 GPU 测量：启动被拒绝，NOT_RUN

```bash
source docs/experiments/a100-stage2a-gemm/environment/activate.sh
python docs/experiments/donkeynet-backward-profiling/scripts/run_gpu.py \
  --gpu GPU-d40fdd6f-33be-8ccc-fc58-f4daf39a4b82 \
  --record runs/donkeynet_4a_measurement_monitor_01 -- \
  python docs/experiments/donkeynet-backward-profiling/scripts/run_measurements.py \
    --gpu-uuid GPU-d40fdd6f-33be-8ccc-fc58-f4daf39a4b82 \
    --output runs/donkeynet_4a_measurements_01
```

返回 1，输出 `BLOCKED: selected GPU is not idle; no GPU work started`。12:04:56–57 UTC 两次快照均有外部计算进程和 21960 MiB 占用。子脚本从未执行，所以没有训练输出目录或子进程 launch.json。实际退出信息在 [result.json](logs/measurement-attempt-01/result.json)，明确记录其来源为执行器结果；原 `before.json` 在服务器 `runs/` 中保留。

后续八卡快照均占用；本轮没有改用另一张忙碌设备、没有启动新 GPU Trace、没有长时间等待。

准备好的 `run_measurements.py` 将执行三个后端各 3 次独立重复，分开 throughput/phases 共 18 个子进程。每次 `--samples 256 --batch 4 --epochs 4 --warmup 10 --seed 42 --lr 0.001 --workers 0 --no-monitor --no-fingerprints --save-comparison-inputs`。CuPy 为全部 cupy；Combined 为 `shared/im2col/cupy/tile16`；Torch 使用相同显式输入/初始化映射。计划每次 256 Step，实际执行次数 **0**。

## CPU 插桩验证：已执行

第一次为 5 项测试，扩充 CLI 与 Torch 上下文合约后第二次为 10 项；两份日志都保留。完整命令分别在 `logs/instrumentation-cpu-tests.command.json` 和 `logs/instrumentation-cpu-tests-02.command.json`。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  .venv/bin/python -m pytest \
  docs/experiments/donkeynet-backward-profiling/scripts/test_profile_training.py -q

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  .venv/bin/python -m pytest \
  docs/experiments/donkeynet-backward-profiling/scripts/test_profile_training.py -q \
  --junitxml=docs/experiments/donkeynet-backward-profiling/logs/instrumentation-cpu-tests-02.xml
```

stdout/stderr 另重定向至对应 `.log`。CPU Tensor 逐元素比较，无哈希；Torch NVTX context 合约用假对象，不代表 GPU 验证。

## 历史 Trace 与 CPU 归因检查：已执行

初次分析保留未归一化 PID/TID 的派生数据，随后移动到独立服务器目录；公开版本仅将标识换成稳定标签，时间与数值不变：

```bash
python3 docs/experiments/donkeynet-backward-profiling/scripts/analyze_history.py \
  --input runs/nsys_dtype_fixed_01.sqlite \
  --output docs/experiments/donkeynet-backward-profiling/results/historical-nsys-fixed-01
mv docs/experiments/donkeynet-backward-profiling/results/historical-nsys-fixed-01 \
  runs/stage4a_historical_analysis_initial_01
python3 docs/experiments/donkeynet-backward-profiling/scripts/analyze_history.py \
  --input runs/nsys_dtype_fixed_01.sqlite \
  --output docs/experiments/donkeynet-backward-profiling/results/historical-nsys-fixed-01 \
  > runs/stage4a_historical_analysis_01.log

python3 docs/experiments/donkeynet-backward-profiling/scripts/analyze_trace.py \
  --input runs/nsys_dtype_fixed_01.sqlite \
  --output runs/stage4a_generic_history_analysis_01 \
  --label historical-fixed-validation > runs/stage4a_generic_history_analysis_01.log

python3 docs/experiments/donkeynet-backward-profiling/scripts/check_trace_analysis.py \
  --output runs/stage4a_trace_attribution_check_01 > runs/stage4a_trace_attribution_check_01.log
python3 docs/experiments/donkeynet-backward-profiling/scripts/check_trace_analysis.py \
  --output runs/stage4a_trace_attribution_check_02 > runs/stage4a_trace_attribution_check_02.log
```

两次合成归因检查均 PASSED；第二次因增加 correlationId 一致性检查和 schema 导出而复测，不是修复一个不存在的 GPU 失败。Fixture 明确标记 `SYNTHETIC_CPU_TEST_NOT_GPU_MEASUREMENT`，不混入 GPU 图表。

## 归档、热点和绘图：已执行

```bash
source docs/experiments/a100-stage2a-gemm/environment/activate.sh
python docs/experiments/a100-stage1/run_logged.py \
  --log docs/experiments/donkeynet-backward-profiling/logs/archive.log -- \
  python docs/experiments/donkeynet-backward-profiling/scripts/archive_evidence.py
python docs/experiments/a100-stage1/run_logged.py \
  --log docs/experiments/donkeynet-backward-profiling/logs/hotspots.log -- \
  python docs/experiments/donkeynet-backward-profiling/scripts/summarize_hotspots.py
MPLCONFIGDIR=/home/xymo/projects/hytorch/.cache/matplotlib .venv/bin/python \
  docs/experiments/donkeynet-backward-profiling/scripts/plot_results.py \
  > docs/experiments/donkeynet-backward-profiling/logs/plot-results.log 2>&1
```

阶段图首次检查后调整标注/图幅；相应记录在 `logs/plot-phase-layout.log`，没有改变实测数据。原始指标和单次训练结果同时保留。本轮新生成的 `.cache/matplotlib` 字体缓存移至已忽略的 `runs/donkeynet_4a_plot_cache_01/`，未纳入 Git；以后作图可直接使用环境脚本的 `.matplotlib_cache` 路径。索引生成命令如下，只读取文件路径与字节数：

```bash
.venv/bin/python docs/experiments/donkeynet-backward-profiling/scripts/index_evidence.py
```

## 下一次获准采集的命令模板：本轮未执行

以下仅是待复核的单个 Combined 示例。先以新的空闲设备 UUID 设置 `STAGE4A_GPU_UUID`，并替换所有输出目录为未存在的新路径；不能直接假设 GPU 6 继续可用。CuPy 和 Torch 对照应分别运行，与无算子插桩的控制组分开，数据/初始参数逐元素比较、核查有限值后再分析。

```bash
source docs/experiments/a100-stage2a-gemm/environment/activate.sh
python docs/experiments/donkeynet-backward-profiling/scripts/run_gpu.py \
  --gpu "$STAGE4A_GPU_UUID" --record runs/donkeynet_4a_detail_monitor_NEXT -- \
  nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
    -o runs/donkeynet_4a_combined_detail_NEXT \
    python docs/experiments/donkeynet-backward-profiling/scripts/profile_training.py \
      --operator-nvtx --memory-stats --nvtx --no-fingerprints \
      --framework hytorch --device cuda --gpu 0 --model donkey \
      --matmul shared --conv im2col --pool cupy --tile 16 \
      --samples 64 --batch 4 --epochs 1 --warmup 10 --seed 42 --lr 0.001 \
      --mode throughput --no-monitor --save-comparison-inputs \
      --output runs/donkeynet_4a_combined_detail_training_NEXT
nsys export --type sqlite --output runs/donkeynet_4a_combined_detail_NEXT.sqlite \
  runs/donkeynet_4a_combined_detail_NEXT.nsys-rep
python docs/experiments/donkeynet-backward-profiling/scripts/analyze_trace.py \
  --input runs/donkeynet_4a_combined_detail_NEXT.sqlite \
  --output runs/donkeynet_4a_combined_detail_analysis_NEXT --label combined
```

CuPy 配置改为 `--matmul cupy --conv cupy --pool cupy`；Torch 使用 `--framework torch` 且后端选项全 cupy，保留 `--operator-nvtx --nvtx`、去掉 `--memory-stats`。无算子插桩控制组去掉 `--operator-nvtx --memory-stats`。需要同步分阶段对照时独立改 `--mode phases`，不能将两种模式的时间混合；普通 benchmark 仍使用 `run_measurements.py` 中的无 NVTX 流程。

当前模板不采 CPU 栈/调度事件，无法单独区分 Python 计算和 OS 调度；是否增加 CPU 采样应在后续获准采集时评估，不通过修改 GPU 系统权限解决。也不因 NCU 受限伪造实际 Occupancy。

## Git 交付

仅 stage 本轮目录与 benchmark 的 NVTX 参数校验；先核对范围、文本、实际数据索引和敏感信息，然后 commit 与普通 push。具体提交 ID 由最终交付与 Git 历史给出，避免循环写入自身。未 force push、未合并 main，推送后停止。
