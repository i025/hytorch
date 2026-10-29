# 逐次真实执行记录

以下从即时生成的命令元数据整理。时间为 UTC；退出码非零的失败记录完整保留。

每个 CPU 命令由 `docs/experiments/a100-stage1/run_logged.py --log <对应日志> -- <命令>` 执行。

## 2026-10-29T07:00:11.154204+00:00 — exit 0

[原始日志](evidence/cuda-sass.log) · [完整命令/环境元数据](evidence/cuda-sass.log.command.json)

```bash
cuobjdump --dump-sass hytorch/backends/cuda/build/libhytorch_cuda.so
```

## 2026-10-29T07:00:11.947162+00:00 — exit 0

[原始日志](evidence/cuda-ptx.log) · [完整命令/环境元数据](evidence/cuda-ptx.log.command.json)

```bash
cuobjdump --dump-ptx hytorch/backends/cuda/build/libhytorch_cuda.so
```

## 2026-10-29T07:01:52.041718+00:00 — exit 2

[原始日志](results/arithmetic-tests.log) · [完整命令/环境元数据](results/arithmetic-tests.log.command.json)

```bash
python -m pytest docs/experiments/gemm-fp32-accuracy/scripts/test_accuracy.py -v --junitxml=docs/experiments/gemm-fp32-accuracy/results/arithmetic-tests.xml
```

## 2026-10-29T07:01:52.603729+00:00 — exit 1

[原始日志](results/diagnosis.log) · [完整命令/环境元数据](results/diagnosis.log.command.json)

```bash
python docs/experiments/gemm-fp32-accuracy/scripts/cpu_diagnosis.py --output docs/experiments/gemm-fp32-accuracy/results
```

## 2026-10-29T07:02:03.253929+00:00 — exit 0

[原始日志](results/arithmetic-tests-02.log) · [完整命令/环境元数据](results/arithmetic-tests-02.log.command.json)

```bash
python -m pytest docs/experiments/gemm-fp32-accuracy/scripts/test_accuracy.py -v --junitxml=docs/experiments/gemm-fp32-accuracy/results/arithmetic-tests-02.xml
```

## 2026-10-29T07:02:03.607518+00:00 — exit 1

[原始日志](results/diagnosis-02.log) · [完整命令/环境元数据](results/diagnosis-02.log.command.json)

```bash
python docs/experiments/gemm-fp32-accuracy/scripts/cpu_diagnosis.py --output docs/experiments/gemm-fp32-accuracy/results
```

## 2026-10-29T07:03:02.493904+00:00 — exit 0

[原始日志](results/diagnosis-03.log) · [完整命令/环境元数据](results/diagnosis-03.log.command.json)

```bash
python docs/experiments/gemm-fp32-accuracy/scripts/cpu_diagnosis.py --output docs/experiments/gemm-fp32-accuracy/results/completed-03
```

## 2026-10-29T07:03:03.660781+00:00 — exit 0

[原始日志](environment.txt) · [完整命令/环境元数据](environment.txt.command.json)

```bash
python docs/experiments/gemm-fp32-accuracy/scripts/environment.py
```

## 2026-10-29T07:04:08.723466+00:00 — exit 0

[原始日志](results/cpu-analysis.log) · [完整命令/环境元数据](results/cpu-analysis.log.command.json)

```bash
python docs/experiments/gemm-fp32-accuracy/scripts/analyze.py --results docs/experiments/gemm-fp32-accuracy/results
```

## 2026-10-29T07:07:40.838319+00:00 — exit 0

[原始日志](results/analysis.log) · [完整命令/环境元数据](results/analysis.log.command.json)

```bash
python docs/experiments/gemm-fp32-accuracy/scripts/analyze.py --results docs/experiments/gemm-fp32-accuracy/results --gpu docs/experiments/gemm-fp32-accuracy/results/gpu-01
```

## 2026-10-29T07:09:17.603962+00:00 — exit 1

[原始日志](results/figures.log) · [完整命令/环境元数据](results/figures.log.command.json)

```bash
python docs/experiments/gemm-fp32-accuracy/scripts/plot_results.py --results docs/experiments/gemm-fp32-accuracy/results --output docs/experiments/gemm-fp32-accuracy/figures
```

## 2026-10-29T07:09:17.605273+00:00 — exit 0

[原始日志](results/audit.log) · [完整命令/环境元数据](results/audit.log.command.json)

```bash
python docs/experiments/gemm-fp32-accuracy/scripts/audit_results.py --results docs/experiments/gemm-fp32-accuracy/results
```

## 2026-10-29T07:09:20.239045+00:00 — exit 0

[原始日志](evidence/gpu-environment-after.txt) · [完整命令/环境元数据](evidence/gpu-environment-after.txt.command.json)

```bash
nvidia-smi --id=GPU-158426a4-6684-7863-8cbe-ed6595f97683 --query-gpu=index,uuid,name,memory.total,driver_version,memory.used,utilization.gpu --format=csv
```

## 2026-10-29T07:09:51.745227+00:00 — exit 0

[原始日志](results/figures-02.log) · [完整命令/环境元数据](results/figures-02.log.command.json)

```bash
python docs/experiments/gemm-fp32-accuracy/scripts/plot_results.py --results docs/experiments/gemm-fp32-accuracy/results --output docs/experiments/gemm-fp32-accuracy/figures
```

## GPU 2026-10-29T07:07:02.707635+00:00 — exit 0

[原始日志](evidence/gpu-monitor-01/benchmark.log) · [启动](evidence/gpu-monitor-01/launch.json) · [结束](evidence/gpu-monitor-01/completion.json)

```bash
python docs/experiments/gemm-fp32-accuracy/scripts/gpu_compare.py --positions docs/experiments/gemm-fp32-accuracy/results/sample-positions.json --output docs/experiments/gemm-fp32-accuracy/results/gpu-01
```

外层 GPU 空闲检查/隔离、归档与其他操作见 [commands.md](commands.md)。
