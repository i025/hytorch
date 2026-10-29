# 执行命令与参数索引

工作目录 `/home/xymo/projects/hytorch`（真实路径 `/srv/lab-disks/disk2/users/xymo/projects/hytorch`）。所有 GPU 工作只用选定 UUID 映射成逻辑 0；不根据物理序号猜测设备。

```bash
cd /home/xymo/projects/hytorch
source docs/experiments/a100-stage2a-gemm/environment/activate.sh
git switch -c experiment/conv-backward-shape-dispatch experiment/conv-backward-fusion
```

分支基线实际为 `d26667f160bf3af772ad24685f73049f6267a11a`。激活独立项目环境，CPU 任务保持 `CUDA_VISIBLE_DEVICES=''`；GPU guard 为子进程设置 UUID，其他 GPU 不可见。环境、所有外层日志的 `.command.json`、各原始目录 `commands.json`/guard `launch.json` 保留完整参数和时间。

## CPU 回归

最终主机命令（原始 logs/host-03.log 记录实际 argv）：

```bash
python -m pytest tests/test_conv_shape_dispatch.py tests/test_conv_backward_fusion_host.py \
  tests/test_conv_dinput_pruning.py tests/test_cuda_host.py -q -ra \
  --junitxml=docs/experiments/conv-shape-dispatch/results/host-03.xml
```

此前 host-01、host-02 的新模型测试也保留，不把重复运行相加成独立覆盖数。

## 跨形状测量

脚本中每个方法独立进程，5 次预热、30 次 CUDA Event/墙钟记录、3 次重复、轮换方法顺序。输入 FP32、seed=42+case 编号；所有原始输入/参考输出留在 NPZ。

```bash
python docs/experiments/conv-shape-dispatch/scripts/wait_gpu.py \
  --output runs/NEW_UNIQUE_WINDOW --seconds 3600 -- \
  python docs/experiments/conv-shape-dispatch/scripts/run_shapes.py \
  --split calibration --output '{output}'
```

重复其他 split 为 `held-out`、`validation-2`、`validation-final`，每次换目录。复现 GPU 需要在环境指定既有动态库：

```bash
export HYTORCH_CUDA_LIBRARY=/home/xymo/projects/hytorch/runs/conv_backward_fusion_build_01/libhytorch_cuda.so
```

本轮沿用 4D 已编译的 SM80 动态库，未修改/重编译 Kernel。原始冻结策略文件由其 frozen_at 标识先后；三次候选的独立集及冻结过程见 design-protocol.md。未使用验证结果重新调整同一候选。

分析的原始路径和实际命令保存在 logs/*analysis*.log.command.json，例如：

```bash
python docs/experiments/conv-shape-dispatch/scripts/analyze_shapes.py \
  --raw runs/conv_shape_calibration_wait_02/attempt-01/results --split calibration \
  --output docs/experiments/conv-shape-dispatch/results/calibration-01
python docs/experiments/conv-shape-dispatch/scripts/refine_rule.py \
  --calibration docs/experiments/conv-shape-dispatch/results/calibration-01 \
  --previous-validation docs/experiments/conv-shape-dispatch/results/held-out-01 \
  --output docs/experiments/conv-shape-dispatch/results/refined-policy-01
```

上述为本次实际目录，重跑必须换输出名，不能覆盖。

## GPU 回归、实际 Auto 复测及匹配训练

```bash
python docs/experiments/conv-shape-dispatch/scripts/wait_gpu.py \
  --output runs/conv_shape_measurements_wait_02 --seconds 3600 -- \
  python docs/experiments/conv-shape-dispatch/scripts/run_validation.py \
  --only measurements --output '{output}' --gpu-uuid '{gpu}' \
  --library /home/xymo/projects/hytorch/runs/conv_backward_fusion_build_01/libhytorch_cuda.so
```

先执行真实 GPU pytest，然后 `run_shapes.py --split validation-final --auto`。随后两个模型、throughput/phases 两种计时、A/D/F/G/E 五组、3 次重复，共 60 个独立训练进程。每进程 batch4、10 次预热并恢复初值、256 samples×4 epochs=256 step，seed42、SGD lr0.001、FP32、Dropout0、workers0、TF32 关闭，全部 `--no-fingerprints --save-comparison-inputs --no-monitor`。

各组参数只有：A=CuPy/普通SGD/不剪枝；D=`--matmul shared --conv im2col --pool cupy --tile 16 --fused-sgd --prune-input-grad`；F=D+`--conv-backward fused`；G=D+`--conv-backward auto`；E=`--framework torch`。A 故意指定不存在的 native library，并核查 dispatch 和 Trace；不能把 CuPy 内部 CUDA/cuBLAS 当成本项目 Kernel。

```bash
python docs/experiments/conv-shape-dispatch/scripts/analyze_training.py \
  --raw runs/conv_shape_measurements_wait_02/attempt-01/results \
  --output docs/experiments/conv-shape-dispatch/results/training-01
python docs/experiments/conv-shape-dispatch/scripts/analyze_shapes.py \
  --raw runs/conv_shape_measurements_wait_02/attempt-01/results/auto-shapes \
  --split validation-final \
  --policy docs/experiments/conv-shape-dispatch/results/pointwise-frozen-policy.json \
  --output docs/experiments/conv-shape-dispatch/results/actual-auto-01
```

## Nsight 与导出

普通训练完成后另开 profiling 窗口。GPU6 无进程、利用率0，但连续保留3 MiB，严格零内存等待器没有启动 GPU；保留原等待记录后，以显式 `--idle-memory-mib 4` 接纳少量驱动驻留，仍要求无客户端、0利用率，并持续检查外部进程。没有 reset、sudo 或系统配置修改。

```bash
python docs/experiments/conv-shape-dispatch/scripts/wait_gpu.py \
  --output runs/conv_shape_traces_wait_02 --seconds 3600 --idle-memory-mib 4 -- \
  python docs/experiments/conv-shape-dispatch/scripts/run_validation.py \
  --only traces --output '{output}' --gpu-uuid '{gpu}' \
  --library /home/xymo/projects/hytorch/runs/conv_backward_fusion_build_01/libhytorch_cuda.so
```

每个 Trace 使用 `nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none -o UNIQUE ...`，随后 `nsys export --type sqlite --output UNIQUE.sqlite UNIQUE.nsys-rep`；完整命令均在该窗口 commands.json。训练 Trace 为每模型每组16 step（samples64/epochs1），另有100形状×3方法及最后24形状的auto，每形状5预热/3记录。

```bash
python docs/experiments/conv-shape-dispatch/scripts/analyze_profiles.py \
  --raw runs/conv_shape_traces_wait_02/attempt-01/results \
  --analysis runs/conv_shape_trace_analysis_01 \
  --output docs/experiments/conv-shape-dispatch/results/profiles-01
python docs/experiments/conv-shape-dispatch/scripts/plot_results.py
python docs/experiments/conv-shape-dispatch/scripts/summarize_results.py
```

最终高阶/路由测试加入后，另在单卡guard内重新执行完整五文件GPU测试集合（188项），实际argv见 `logs/gpu-final-01.log.command.json` 和 `runs/conv_shape_final_tests_wait_01/attempt-01/guard/launch.json`。设置 `HYTORCH_REQUIRE_CUDA=1 HYTORCH_PRUNING_DEVICE=cuda` 和各比较数据目录，XML保存为该窗口 `results/gpu-tests.xml`；未使用CPU skipped冒充GPU通过。

## 留档与发布

所有 run_logged.py 外层命令、退出码、原始stdout/stderr、子进程命令、失败和中断均留存。归档复用未改动的 `docs/experiments/fused-sgd-cuda/scripts/archive_results.py`，复制文本，超过1 MB的文本 gzip；`.npz`、`.nsys-rep`、SQLite 和动态库仅列服务器路径与大小，不上传，不计算内容哈希。完整11个源窗口的实际归档argv见 `logs/archive-raw-01.log.command.json`；完整23份解析的归档见 `logs/archive-profiles-01.log.command.json`。

最终检查并提交到 `experiment/conv-backward-shape-dispatch`，普通 push；不合并 main，不开始后续 Kernel 开发。
