# 执行命令与复现说明

全部命令从 `/home/xymo/projects/hytorch` 执行。逐条真实运行时间、参数、退出码和日志见 [command-records.md](command-records.md)，其来源为每次执行即时写出的 `.command.json` 及 GPU `launch.json`。失败尝试也列在其中。

## 1. 基线、环境与源码读取

本分支从远程 `experiment/a100-gpu-complete` 的 `aad4d0562fd968083a727099e478e7db64d35dd0` 创建，名称为 `experiment/gemm-fp32-accuracy`。未合并 main，未覆盖其他分支工作。

```bash
cd /home/xymo/projects/hytorch
source docs/experiments/a100-stage2a-gemm/environment/activate.sh
```

此环境脚本激活已有独立 `.venv`，设置 CPU 阶段 `CUDA_VISIBLE_DEVICES=''`、`OPENBLAS_NUM_THREADS=4`、`OMP_NUM_THREADS=4`、`CUPY_TF32=0`、`NVIDIA_TF32_OVERRIDE=0` 和项目缓存路径。未安装依赖，没有 sudo 或系统环境更改。

已阅读 `benchmarks/bench_cuda_gemm.py`、`hytorch/backends/cuda/runtime.py`、`hytorch/backends/cuda/kernels.cu`、历史 `diagnose_gemm.py`、`diagnose_accumulation.py`，以及旧的异常位置 JSON、原编译记录。读取 PTX/SASS 的实际命令为：

```bash
python docs/experiments/a100-stage1/run_logged.py --log docs/experiments/gemm-fp32-accuracy/evidence/cuda-ptx.log -- cuobjdump --dump-ptx hytorch/backends/cuda/build/libhytorch_cuda.so
python docs/experiments/a100-stage1/run_logged.py --log docs/experiments/gemm-fp32-accuracy/evidence/cuda-sass.log -- cuobjdump --dump-sass hytorch/backends/cuda/build/libhytorch_cuda.so
python docs/experiments/a100-stage1/run_logged.py --log docs/experiments/gemm-fp32-accuracy/environment.txt -- python docs/experiments/gemm-fp32-accuracy/scripts/environment.py
```

只反汇编已有文件，未执行编译、Nsight 或性能 benchmark。`evidence/native-build.json` 是已有 `hytorch/backends/cuda/build/build.json` 的直接副本，没有计算哈希。

## 2. CPU 离线计算

首次语法错误及第二次汇总序列化错误见原始日志，具体每条命令见 command-records。修正后的成功命令为：

```bash
python docs/experiments/a100-stage1/run_logged.py --log docs/experiments/gemm-fp32-accuracy/results/arithmetic-tests-02.log -- python -m pytest docs/experiments/gemm-fp32-accuracy/scripts/test_accuracy.py -v --junitxml=docs/experiments/gemm-fp32-accuracy/results/arithmetic-tests-02.xml
python docs/experiments/a100-stage1/run_logged.py --log docs/experiments/gemm-fp32-accuracy/results/diagnosis-03.log -- python docs/experiments/gemm-fp32-accuracy/scripts/cpu_diagnosis.py --output docs/experiments/gemm-fp32-accuracy/results/completed-03
python docs/experiments/a100-stage1/run_logged.py --log docs/experiments/gemm-fp32-accuracy/results/cpu-analysis.log -- python docs/experiments/gemm-fp32-accuracy/scripts/analyze.py --results docs/experiments/gemm-fp32-accuracy/results
```

`results/samples.csv` 和 `sample-positions.json` 是第二次执行中数值计算已完成的文件。第三次完整成功输出保存在 `completed-03/`。对两次 10,836 行逐字段比较完全相等，不通过哈希判断；`results/cpu-summary.json.partial` 保留第二次失败的 JSON 原始内容。

## 3. GPU 空闲检查与一次性数值对照

CPU 完成后，用原 `wait_for_gpu.py` 的 `snapshot()` 与 `idle()` 做了间隔 10 秒的两次检查，分别保存为 `evidence/gpu-availability.json` 的 first/second。底层只读查询为：

```bash
nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits
```

选择物理 2；启动器再次确认没有进程、显存与利用率均为零，然后把所选 UUID 映射到唯一逻辑设备 0。实际执行：

```bash
python docs/experiments/a100-stage2a-gemm/scripts/run_monitored.py --gpu GPU-158426a4-6684-7863-8cbe-ed6595f97683 --record runs/gemm_accuracy_gpu_monitor_01 -- python docs/experiments/gemm-fp32-accuracy/scripts/gpu_compare.py --positions docs/experiments/gemm-fp32-accuracy/results/sample-positions.json --output docs/experiments/gemm-fp32-accuracy/results/gpu-01
python docs/experiments/a100-gpu-window/scripts/archive_results.py runs/gemm_accuracy_gpu_monitor_01 docs/experiments/gemm-fp32-accuracy/evidence/gpu-monitor-01
python docs/experiments/a100-stage1/run_logged.py --log docs/experiments/gemm-fp32-accuracy/evidence/gpu-environment-after.txt -- nvidia-smi --id=GPU-158426a4-6684-7863-8cbe-ed6595f97683 --query-gpu=index,uuid,name,memory.total,driver_version,memory.used,utilization.gpu --format=csv
```

`gpu_compare.py` 本身不选择物理卡；它要求只可见一张卡并使用逻辑 0。比较的是全部矩阵值与已有参考，没有计时排名、warmup/repeat 性能实验。保留全部失败坐标，即使原容差超差也完成诊断输出；日志 returncode=0 表示诊断执行完成，不代表原 GEMM 验收通过。

## 4. 汇总、交叉核查与作图

```bash
python docs/experiments/a100-stage1/run_logged.py --log docs/experiments/gemm-fp32-accuracy/results/analysis.log -- python docs/experiments/gemm-fp32-accuracy/scripts/analyze.py --results docs/experiments/gemm-fp32-accuracy/results --gpu docs/experiments/gemm-fp32-accuracy/results/gpu-01
python docs/experiments/a100-stage1/run_logged.py --log docs/experiments/gemm-fp32-accuracy/results/audit.log -- python docs/experiments/gemm-fp32-accuracy/scripts/audit_results.py --results docs/experiments/gemm-fp32-accuracy/results
python docs/experiments/a100-stage1/run_logged.py --log docs/experiments/gemm-fp32-accuracy/results/figures-02.log -- python docs/experiments/gemm-fp32-accuracy/scripts/plot_results.py --results docs/experiments/gemm-fp32-accuracy/results --output docs/experiments/gemm-fp32-accuracy/figures
python docs/experiments/gemm-fp32-accuracy/scripts/index_evidence.py
```

首次画图的 `figures.log` 记录发现零误差后的主动失败；修正后的图保留零误差统计并标注数量，不给零误差人为加 epsilon。未使用图片生成工具，未伪造终端截图。

这些日志/结果使用独占创建模式。复现时应把输出换成新的目录，例如新 `runs/gemm_accuracy_reproduction_日期/`，不要直接重放上述路径覆盖本次记录。CPU 计算脚本可以直接使用新的 `--output`；汇总脚本的 `completed-03` 是本次归档布局，复现者应相应组织目录或直接读取新 CPU summary。

## 5. 提交边界

只提交 `docs/experiments/gemm-fp32-accuracy/` 下的新文件。提交前检查变更清单、文本格式、链接及文件体积；不上传缓存、动态库、权重、输入大矩阵或凭据。Git commit 与 push 已由本轮用户明确授权；完成后停止，不合并 main。
