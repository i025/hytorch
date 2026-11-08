# 复现命令

本次环境未安装/升级驱动、修改权限或安装新依赖。Python环境为 /home/xieminhui_pubuse/hytorch-env；NVML binding、psutil、matplotlib已存在。先检查GPU占用，仅在有权限的空闲GPU上运行；下例物理GPU1只适用于确认其仍空闲时。禁止影响其他用户任务。

```bash
cd /home/xieminhui_pubuse/hytorch
nvidia-smi
export CUDA_VISIBLE_DEVICES=1
export NVIDIA_TF32_OVERRIDE=0
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH=/home/xieminhui_pubuse/hytorch
PY=/home/xieminhui_pubuse/hytorch-env/bin/python
$PY -m pytest -q tests/test_training_monitor.py
HYTORCH_REQUIRE_CUDA=1 $PY -m pytest -q tests
$PY examples/training_monitor.py --device cpu --output runs/monitor_example_fresh
$PY benchmarks/bench_cuda_training.py --framework hytorch --device cuda --gpu 0 --model donkey --samples 256 --batch 4 --epochs 3 --final-config --monitor --monitor-interval .2 --output runs/monitor_cli_fresh
```

TinyNet用--model tiny；Torch参考用--framework torch；资源趋势用--monitor-components resources；关闭新monitor开销对比用--monitor --monitor-components disabled（保持相同helper循环）；latency diagnostic用--monitor-mode latency。普通训练不传--monitor默认关闭新模块。CPU例子无需CUDA_VISIBLE_DEVICES。

为保护本次记录，新建同深度实验目录，脚本从自身位置推导输出和仓库根。不要在现有目录重跑并覆盖CSV。

```bash
NEW=runs/a100_20261108/training_monitor_recheck_$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$NEW/diagnostics" "$NEW/raw" "$NEW/traces" "$NEW/figures"
cp runs/a100_20261108/training_monitor/diagnostics/{experiment,analyze,figures}.py "$NEW/diagnostics/"
$PY "$NEW/diagnostics/experiment.py" > "$NEW/raw/experiment.log" 2>&1
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --output "$NEW/traces/training" "$PY" "$NEW/diagnostics/experiment.py" --trace > "$NEW/raw/nsys.log" 2>&1
nsys export --type sqlite --output "$NEW/traces/training.sqlite" "$NEW/traces/training.nsys-rep"
$PY "$NEW/diagnostics/analyze.py" > "$NEW/raw/analysis.log" 2>&1
$PY "$NEW/diagnostics/figures.py" > "$NEW/raw/figures.log" 2>&1
```

experiment.py执行12轮四模式平衡配对、独立allocator/latency诊断、30epochs趋势、3轮Torch参考、TinyNet四模式；固定seed42、相同初始化和10step warmup，最终同步相同。Torch在CuPy设备初始化前导入，规避本机混合CUDA runtime动态符号问题。Nsight运行只20steps，不作为正式wall验收。分析需要sqlite；压缩包不含大二进制trace，服务器原文件路径与SHA见EXTERNAL_ARTIFACTS.json，或按上述命令重新生成。图表仅依赖CSV，可直接复绘。

本次测试与初次失败日志均在raw/；RUNS.csv中run_dir是本次实际唯一目录。下载归档后路径需按解压位置调整。git_diff.patch相对于baseline，不相对于仓库HEAD；不要直接覆盖此前未提交算法修改。
