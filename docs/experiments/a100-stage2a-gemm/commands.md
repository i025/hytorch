# 已实际执行的命令

工作目录 `/home/xymo/projects/hytorch`，记录时间为 UTC。所有 CPU 检查均隐藏 GPU；完整设置见 `environment/activate.sh`。该脚本的 TF32 变量未被复用的阶段 1 日志器列入白名单，但由环境脚本和 benchmark summary 记录。

## checks/cpu-format.log.command.json

UTC 2026-10-28T15:40:18.129392+00:00；退出码 0；耗时 0.8605514210648835 s。

```bash
python benchmarks/bench_cuda_gemm.py --cpu-only --sizes 64 128 --tiles 16 32 --repeats 10 --warmup 5 --seed 42 --output runs/gemm_stage2a_cpu_format_01
```

## environment/python-info.txt.command.json

UTC 2026-10-28T15:40:19.037137+00:00；退出码 0；耗时 0.4388616709038615 s。

```bash
python -c 'import sys,numpy,cupy; print(sys.version); print("NumPy",numpy.__version__); print("CuPy",cupy.__version__); numpy.show_config()'
```

## environment/system-info.txt.command.json

UTC 2026-10-28T15:40:22.519062+00:00；退出码 0；耗时 0.09763793204911053 s。

```bash
bash -c 'date -Is; hostname; cat /etc/os-release; lscpu; gcc --version; nvcc --version; git --version; git log -1 --format=%H'
```

## environment/gpu-info-01.txt.command.json

UTC 2026-10-28T15:40:22.657407+00:00；退出码 0；耗时 0.20879960898309946 s。

```bash
nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.total,memory.used,utilization.gpu,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,pstate --format=csv
```

## environment/gpu-processes-01.txt.command.json

UTC 2026-10-28T15:40:22.908032+00:00；退出码 0；耗时 0.133096172939986 s。

```bash
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
```

## checks/record-check.log.command.json

UTC 2026-10-28T15:43:24.459995+00:00；退出码 0；耗时 0.11544386902824044 s。

```bash
python docs/experiments/a100-stage2a-gemm/scripts/check_cpu_records.py
```

## checks/host-regression.log.command.json

UTC 2026-10-28T15:43:24.625684+00:00；退出码 0；耗时 2.0688672908581793 s。

```bash
python -m pytest tests/test_cuda_host.py -q -ra --junitxml=docs/experiments/a100-stage2a-gemm/checks/host-regression.xml
```

## environment/gpu-info-02.txt.command.json

UTC 2026-10-28T15:43:27.388260+00:00；退出码 0；耗时 0.15998716792091727 s。

```bash
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv
```

## environment/gpu-processes-02.txt.command.json

UTC 2026-10-28T15:43:27.585238+00:00；退出码 0；耗时 0.12628702609799802 s。

```bash
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
```

其他准备命令：

```bash
git switch -c experiment/a100-stage2a-gemm experiment/a100-stage1
python -m py_compile benchmarks/bench_cuda_gemm.py docs/experiments/a100-stage2a-gemm/scripts/plot_gemm.py docs/experiments/a100-stage2a-gemm/scripts/run_monitored.py
```

正式 GPU 命令尚未执行，见 README，不混入实际命令清单。
