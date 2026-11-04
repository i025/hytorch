# 阶段4H全部实验命令

工作目录`/home/xymo/projects/hytorch`（规范路径`/srv/lab-disks/disk2/users/xymo/projects/hytorch`）。下列是实际目录；重跑必须更换输出后缀，所有脚本拒绝覆盖原始实验。真实argv/环境/起止/返回码见[logs](logs)的`*.command.json`，逐项GPU命令见最终窗口[commands.json](raw-final-01/udacity_real_window_02/commands.json)。不执行哈希或fingerprint。

## A：CPU离线准备

```bash
cd /home/xymo/projects/hytorch
git switch -c experiment/real-udacity-training
source docs/experiments/a100-stage2a-gemm/environment/activate.sh
# CUDA_VISIBLE_DEVICES=''，CUDA_PATH=/usr/local/cuda-12.2
python docs/experiments/udacity-real-training/scripts/prepare_data.py \
  --csv /home/xymo/datasets/udacity/data/driving_log.csv \
  --output runs/udacity_real_prepared_01
python -m pytest tests/test_udacity_dataset.py tests/test_udacity_experiment.py \
  tests/test_conv_dweight_host.py tests/test_conv_dinput_opt_host.py \
  tests/test_conv_shape_dispatch.py tests/test_conv_backward_fusion_host.py \
  tests/test_conv_dinput_pruning.py tests/test_cuda_host.py \
  tests/test_conv2d_gemm_cpu.py tests/test_gemm_blocked_packed.py -q -ra \
  --junitxml=docs/experiments/udacity-real-training/results/host-tests-02.xml
python docs/experiments/udacity-real-training/scripts/check_correctness.py \
  --device cpu --prepared runs/udacity_real_prepared_01 \
  --output runs/udacity_real_cpu_correctness_01
python -m compileall -q donkeycar_hy/udacity_dataset.py \
  docs/experiments/udacity-real-training/scripts tests/test_udacity_dataset.py tests/test_udacity_experiment.py
python -c 'import ctypes; lib=ctypes.CDLL("runs/conv_dweight_build_01/libhytorch_cuda.so"); print("ABI",lib.hy_cuda_abi_version(),"dWeight",bool(lib.hy_conv_dweight))'
python docs/experiments/udacity-real-training/scripts/run_window.py \
  --prepared runs/udacity_real_prepared_01 --output runs/udacity_real_window_02 \
  --library runs/conv_dweight_build_01/libhytorch_cuda.so \
  --plan docs/experiments/udacity-real-training/results/gpu-plan-02.json
```

复用4G编译产物，不改Kernel，不为新benchmark编译第二个方案。第一次80项CPU测试、最终82项测试、408组CPU真实数值输出均保留。环境记录命令date/os-release/lscpu/python/pip freeze/nvcc/nsys见[cpu-environment.log.command.json](environment/cpu-environment.log.command.json)。

关键命令均使用既有日志器，例如：

```bash
python docs/experiments/a100-stage1/run_logged.py \
  --log docs/experiments/udacity-real-training/logs/dataset-01.log -- \
  python docs/experiments/udacity-real-training/scripts/prepare_data.py \
    --csv /home/xymo/datasets/udacity/data/driving_log.csv --output runs/udacity_real_prepared_01
```

## B：固定共享物理GPU1

控制器内部先运行`nvidia-smi --id=1 --query-gpu=...`和`--query-compute-apps=...`，取真实UUID后固定CUDA_VISIBLE_DEVICES；不查询其他卡以作设备选择，所有模型仅逻辑0。pmon固定`nvidia-smi pmon -i 1 -s um -c 1`。不使用4G的独占空闲卡guard或其deviceReset释放方式。

首次窗口命令为以下同入口，输出`runs/udacity_real_window_01`、外层日志`gpu-window-01.log`；正确性通过后原进程退出，控制器停止本轮。用户确认正常结束并明确允许同GPU1完整补测，第二次实际命令：

```bash
source docs/experiments/a100-stage2a-gemm/environment/activate.sh
python docs/experiments/a100-stage1/run_logged.py \
  --log docs/experiments/udacity-real-training/logs/gpu-window-02.log -- \
  python docs/experiments/udacity-real-training/scripts/run_window.py \
    --prepared runs/udacity_real_prepared_01 --output runs/udacity_real_window_02 \
    --library runs/conv_dweight_build_01/libhytorch_cuda.so
```

47项全序列：1个真实GPU数值对照、4烟测、3种计时×3重复×4后端、4个全数据三epoch训练、2份Nsight。子任务CUDA_VISIBLE_DEVICES为`GPU-c6683db1-d2aa-97b9-c1b4-e497bd7ff0c5`，A另外将HYTORCH_CUDA_LIBRARY设为不存在的文件。每条准确命令、返回码、PID和起止时间均在[commands.json](raw-final-01/udacity_real_window_02/commands.json)，完整终端输出分别在每个同名`.log`。

代表性子命令（由控制器组织，非另开GPU窗口运行）：

```bash
python docs/experiments/udacity-real-training/scripts/check_correctness.py \
  --prepared runs/udacity_real_prepared_01 --output runs/udacity_real_window_02/correctness-real
python docs/experiments/udacity-real-training/scripts/train.py \
  --method I --task io --prepared runs/udacity_real_prepared_01 --no-fingerprints \
  --output runs/udacity_real_window_02/io-I-r1
python docs/experiments/udacity-real-training/scripts/train.py \
  --method I --task train --prepared runs/udacity_real_prepared_01 --no-fingerprints \
  --output runs/udacity_real_window_02/train-I
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  -o runs/udacity_real_window_02/trace-I \
  python docs/experiments/udacity-real-training/scripts/train.py \
    --method I --task trace --prepared runs/udacity_real_prepared_01 --no-fingerprints \
    --output runs/udacity_real_window_02/trace-I
```

协议所有固定参数在[protocol.json](raw-preparation-01/udacity_real_prepared_01/protocol.json)：seed42、FP32、batch4、SGD .001、dropout0、TF32关闭，6330/1608固定训练/验证索引；吞吐1024图/256步、10预热、三次重复并轮换AFIE/FIEA/IEAF；全训练三epoch的完整索引顺序同文件保存。没有隐藏的默认训练长度或数据增强。

结束释放依据为子进程退出、监控join及after.json中无本轮进程。没有GPU Reset、全局电源/时钟修改、MPS/MIG或操作原任务。

## C：GPU释放后CPU分析、绘图与归档

```bash
source docs/experiments/a100-stage2a-gemm/environment/activate.sh
# CUDA_VISIBLE_DEVICES=''
python docs/experiments/udacity-real-training/scripts/analyze.py \
  --raw runs/udacity_real_window_02 --prepared runs/udacity_real_prepared_01 \
  --analysis runs/udacity_real_analysis_01 \
  --output docs/experiments/udacity-real-training/results/acceptance-01
python docs/experiments/udacity-real-training/scripts/diagnose_drift.py \
  --raw runs/udacity_real_window_02 --prepared runs/udacity_real_prepared_01 \
  --output docs/experiments/udacity-real-training/results/drift-01
python docs/experiments/udacity-real-training/scripts/plot.py \
  --results docs/experiments/udacity-real-training/results/acceptance-01 \
  --prepared runs/udacity_real_prepared_01 \
  --output docs/experiments/udacity-real-training/figures/final-01
python docs/experiments/udacity-real-training/scripts/extra_analysis.py \
  --results docs/experiments/udacity-real-training/results/acceptance-01 \
  --raw runs/udacity_real_window_02 \
  --output docs/experiments/udacity-real-training/results/extra-01
python docs/experiments/fused-sgd-cuda/scripts/archive_results.py \
  --destination docs/experiments/udacity-real-training/raw-preparation-01 \
  runs/udacity_real_prepared_01 runs/udacity_real_cpu_correctness_01 runs/udacity_real_window_01
python docs/experiments/fused-sgd-cuda/scripts/archive_results.py \
  --destination docs/experiments/udacity-real-training/raw-final-01 \
  runs/udacity_real_window_02 runs/udacity_real_analysis_01
```

两份Nsight导出/解析的实际命令与返回码在[analysis commands.jsonl](raw-final-01/udacity_real_analysis_01/commands.jsonl)：CPU `nsys export --type sqlite --output ... trace-F/I.nsys-rep`，训练Trace复用4A `analyze_trace.py`，不改历史脚本/数据。

准备阶段的两张图由`plot_preparation.py --prepared runs/udacity_real_prepared_01 --cpu runs/udacity_real_cpu_correctness_01 --gpu runs/udacity_real_window_01/correctness-real --output docs/experiments/udacity-real-training/figures/preparation-01`生成。最终10张图和两张负结果图均为实测CSV/CUPTI可视化，不是截图。

漂移分析在CPU上读取4749步Loss及末态权重，保持1e-6+1e-5相对容差，输出实际FAIL；没有为解决长期失配重启GPU、调整阈值或开发新Kernel。

## 发布

分支`experiment/real-udacity-training`，只提交数据适配器/CPU测试、4H脚本/记录/图及根README链接。GPU全部结束并释放后进行普通commit/push，不force、不合并main。凭据通过独立服务器交互脚本恢复，Token和模型权重不进Git。推送后停止，不开始新优化或Unity集成。
