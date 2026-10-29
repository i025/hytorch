# 真实运行命令索引

下列时间来自即时执行元数据，均为 UTC（北京时间加 8 小时）。错误、中断和最终成功均保留。

## 2026-10-29T07:41:15.807243+00:00 — exit=0

[原始输出](environment/cpu-environment.txt) · [命令元数据](environment/cpu-environment.txt.command.json)

```bash
python docs/experiments/a100-stage1/run_logged.py --log docs/experiments/gemm-block32-tree/environment/cpu-environment.txt -- python docs/experiments/gemm-fp32-accuracy/scripts/environment.py
```

## 2026-10-29T07:41:15.808713+00:00 — exit=0

[原始输出](logs/cpu-tests-01.log) · [命令元数据](logs/cpu-tests-01.log.command.json)

```bash
python docs/experiments/a100-stage1/run_logged.py --log docs/experiments/gemm-block32-tree/logs/cpu-tests-01.log -- python -m pytest experiments/gemm_block32_tree/test_cpu_tree.py -v --junitxml=docs/experiments/gemm-block32-tree/results/cpu-tests-01.xml
```

## 2026-10-29T07:41:15.809357+00:00 — exit=0

[原始输出](logs/build-01.log) · [命令元数据](logs/build-01.log.command.json)

```bash
python docs/experiments/a100-stage1/run_logged.py --log docs/experiments/gemm-block32-tree/logs/build-01.log -- python experiments/gemm_block32_tree/build.py --output runs/gemm_block32_tree_build_01
```

## 2026-10-29T07:41:29.423791+00:00 — exit=0

[原始输出](logs/new-resources.log) · [命令元数据](logs/new-resources.log.command.json)

```bash
python docs/experiments/a100-stage1/run_logged.py --log docs/experiments/gemm-block32-tree/logs/new-resources.log -- cuobjdump --dump-resource-usage runs/gemm_block32_tree_build_01/libblock32_tree.so
```

## 2026-10-29T07:41:29.469323+00:00 — exit=0

[原始输出](logs/new-ptx.log) · [命令元数据](logs/new-ptx.log.command.json)

```bash
python docs/experiments/a100-stage1/run_logged.py --log docs/experiments/gemm-block32-tree/logs/new-ptx.log -- cuobjdump --dump-ptx runs/gemm_block32_tree_build_01/libblock32_tree.so
```

## 2026-10-29T07:41:29.517024+00:00 — exit=0

[原始输出](logs/new-sass.log) · [命令元数据](logs/new-sass.log.command.json)

```bash
python docs/experiments/a100-stage1/run_logged.py --log docs/experiments/gemm-block32-tree/logs/new-sass.log -- cuobjdump --dump-sass runs/gemm_block32_tree_build_01/libblock32_tree.so
```

## 2026-10-29T07:43:13.901568+00:00 — wrapper exit=0, child exit=0

[原始输出](logs/correctness-monitor-01/terminal.log) · [命令元数据](logs/correctness-monitor-01/launch.json)

```bash
python docs/experiments/gemm-block32-tree/scripts/run_gpu.py --gpu GPU-6deac1db-e8b2-558c-0285-ff9639a06a43 --record runs/gemm_block32_tree_correctness_monitor_01 -- python docs/experiments/gemm-block32-tree/scripts/correctness.py --library runs/gemm_block32_tree_build_01/libblock32_tree.so --output docs/experiments/gemm-block32-tree/results/correctness-01
```

## 2026-10-29T07:46:33.805057+00:00 — wrapper exit=0, child exit=0

[原始输出](logs/accuracy-monitor-01/terminal.log) · [命令元数据](logs/accuracy-monitor-01/launch.json)

```bash
python docs/experiments/gemm-block32-tree/scripts/run_gpu.py --gpu GPU-6deac1db-e8b2-558c-0285-ff9639a06a43 --record runs/gemm_block32_tree_accuracy_monitor_01 -- python docs/experiments/gemm-block32-tree/scripts/full_accuracy.py --library runs/gemm_block32_tree_build_01/libblock32_tree.so --correctness docs/experiments/gemm-block32-tree/results/correctness-01 --output docs/experiments/gemm-block32-tree/results/accuracy-01 --raw runs/gemm_block32_tree_accuracy_raw_01
```

## 2026-10-29T07:48:48.528613+00:00 — wrapper exit=0, child exit=0

[原始输出](logs/benchmark-monitor-01/terminal.log) · [命令元数据](logs/benchmark-monitor-01/launch.json)

```bash
python docs/experiments/gemm-block32-tree/scripts/run_gpu.py --gpu GPU-6deac1db-e8b2-558c-0285-ff9639a06a43 --record runs/gemm_block32_tree_benchmark_monitor_01 -- python docs/experiments/gemm-block32-tree/scripts/benchmark.py --library runs/gemm_block32_tree_build_01/libblock32_tree.so --accuracy docs/experiments/gemm-block32-tree/results/accuracy-01 --output docs/experiments/gemm-block32-tree/results/benchmark-01
```

## 2026-10-29T07:50:20.848826+00:00 — wrapper exit=1, child exit=1

[原始输出](logs/ncu-monitor-01/terminal.log) · [命令元数据](logs/ncu-monitor-01/launch.json)

```bash
python docs/experiments/gemm-block32-tree/scripts/run_gpu.py --gpu GPU-6deac1db-e8b2-558c-0285-ff9639a06a43 --record runs/gemm_block32_tree_ncu_monitor_01 -- ncu --set basic --section Occupancy --launch-count 1 --kernel-name regex:gemm_block32_tree --csv python docs/experiments/gemm-block32-tree/scripts/profile.py --library runs/gemm_block32_tree_build_01/libblock32_tree.so --method block32_tree --warmup 0 --iterations 1
```

## 2026-10-29T07:50:49.766479+00:00 — wrapper exit=75, child exit=0

[原始输出](logs/nsys-monitor-01/terminal.log) · [命令元数据](logs/nsys-monitor-01/launch.json)

```bash
python docs/experiments/gemm-block32-tree/scripts/run_gpu.py --gpu GPU-6deac1db-e8b2-558c-0285-ff9639a06a43 --record runs/gemm_block32_tree_nsys_monitor_01 -- nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --output runs/gemm_block32_tree_nsys_01 python docs/experiments/gemm-block32-tree/scripts/profile.py --library runs/gemm_block32_tree_build_01/libblock32_tree.so --method all --warmup 5 --iterations 3
```

## 2026-10-29T07:53:50.545418+00:00 — wrapper exit=0, child exit=0

[原始输出](logs/nsys-monitor-02/terminal.log) · [命令元数据](logs/nsys-monitor-02/launch.json)

```bash
python docs/experiments/gemm-block32-tree/scripts/run_gpu.py --gpu GPU-6deac1db-e8b2-558c-0285-ff9639a06a43 --record runs/gemm_block32_tree_nsys_monitor_02 -- nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --output runs/gemm_block32_tree_nsys_02 python docs/experiments/gemm-block32-tree/scripts/profile.py --library runs/gemm_block32_tree_build_01/libblock32_tree.so --method all --warmup 5 --iterations 3
```

## 2026-10-29T07:55:52.127572+00:00 — exit=0

[原始输出](logs/nsys-export.log) · [命令元数据](logs/nsys-export.log.command.json)

```bash
python docs/experiments/a100-stage1/run_logged.py --log docs/experiments/gemm-block32-tree/logs/nsys-export.log -- nsys export --type sqlite --output runs/gemm_block32_tree_nsys_02.sqlite runs/gemm_block32_tree_nsys_02.nsys-rep
```

## 2026-10-29T07:55:52.219381+00:00 — exit=0

[原始输出](logs/nsys-stats.log) · [命令元数据](logs/nsys-stats.log.command.json)

```bash
python docs/experiments/a100-stage1/run_logged.py --log docs/experiments/gemm-block32-tree/logs/nsys-stats.log -- nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum --format csv --output docs/experiments/gemm-block32-tree/results/nsys-02/summary runs/gemm_block32_tree_nsys_02.sqlite
```

## 2026-10-29T07:55:55.811947+00:00 — exit=0

[原始输出](environment/gpu-after.txt) · [命令元数据](environment/gpu-after.txt.command.json)

```bash
python docs/experiments/a100-stage1/run_logged.py --log docs/experiments/gemm-block32-tree/environment/gpu-after.txt -- nvidia-smi --id=GPU-6deac1db-e8b2-558c-0285-ff9639a06a43 --query-gpu=index,uuid,name,driver_version,memory.total,memory.used,utilization.gpu --format=csv
```

## 2026-10-29T07:57:12.751605+00:00 — exit=0

[原始输出](logs/archive.log) · [命令元数据](logs/archive.log.command.json)

```bash
python docs/experiments/a100-stage1/run_logged.py --log docs/experiments/gemm-block32-tree/logs/archive.log -- python docs/experiments/gemm-block32-tree/scripts/archive.py
```

## 2026-10-29T07:57:12.838484+00:00 — exit=0

[原始输出](logs/analysis-01.log) · [命令元数据](logs/analysis-01.log.command.json)

```bash
python docs/experiments/a100-stage1/run_logged.py --log docs/experiments/gemm-block32-tree/logs/analysis-01.log -- python docs/experiments/gemm-block32-tree/scripts/analyze_results.py
```

## 2026-10-29T07:59:14.432562+00:00 — exit=0

[原始输出](logs/figures-01.log) · [命令元数据](logs/figures-01.log.command.json)

```bash
python docs/experiments/a100-stage1/run_logged.py --log docs/experiments/gemm-block32-tree/logs/figures-01.log -- python docs/experiments/gemm-block32-tree/scripts/plot_results.py
```

独立 nvcc 原始命令另见 [build.json](environment/build.json)。分支操作、源码读取及执行门槛见 [commands.md](commands.md)。
