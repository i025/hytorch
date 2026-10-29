# 阶段 3B 执行命令

工作目录 `/home/xymo/projects/hytorch`。逐次真实命令、UTC 起止时间、退出码与原始日志见 [command-records.md](command-records.md)，不是仅给复现示例。新旧失败记录均保留；重新运行时必须换输出目录，脚本拒绝覆盖已有产物。

## 基线与环境

```bash
git fetch origin experiment/gemm-fp32-accuracy
git switch -c experiment/gemm-block32-tree origin/experiment/gemm-fp32-accuracy
source docs/experiments/a100-stage2a-gemm/environment/activate.sh
```

创建分支时远程基线是 `3bdd482f1c8e1cece33ff90f8f895feb80151d62`。环境脚本复用已有项目 `.venv`、CUDA 路径与项目缓存，CPU 阶段 `CUDA_VISIBLE_DEVICES=''`，4 个 BLAS/OMP 线程，`CUPY_TF32=0`、`NVIDIA_TF32_OVERRIDE=0`。本轮未安装软件。

读取了任务指定的生产 Kernel/runtime/benchmark、阶段 3A 文档和诊断代码，以及既有 Nsight 权限记录。只读 GPU 查询使用 `nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu --format=csv,noheader,nounits` 与 `--query-compute-apps=gpu_uuid`；本轮选择物理 GPU 3，其 UUID 始终不变。

## CPU 语义、编译和二进制检查

```bash
python docs/experiments/a100-stage1/run_logged.py --log docs/experiments/gemm-block32-tree/logs/cpu-tests-01.log -- python -m pytest experiments/gemm_block32_tree/test_cpu_tree.py -v --junitxml=docs/experiments/gemm-block32-tree/results/cpu-tests-01.xml
python docs/experiments/a100-stage1/run_logged.py --log docs/experiments/gemm-block32-tree/logs/build-01.log -- python experiments/gemm_block32_tree/build.py --output runs/gemm_block32_tree_build_01
python docs/experiments/a100-stage1/run_logged.py --log docs/experiments/gemm-block32-tree/environment/cpu-environment.txt -- python docs/experiments/gemm-fp32-accuracy/scripts/environment.py
```

`build.json` 保留完整 nvcc 参数。独立库在 `runs/gemm_block32_tree_build_01/`，生产动态库没有重建。使用 `cuobjdump --dump-resource-usage`、`--dump-ptx`、`--dump-sass` 对新库分别记录 `logs/new-resources.log`、`new-ptx.log`、`new-sass.log`；原 Shared32 资源还由实际 CUDA function attributes 查询，原 SASS 复用 3A 数据。

## GPU 顺序与门槛

每一条 GPU 命令都由以下监控器启动，先检查空闲，仅暴露指定 UUID；有外部 GPU 客户端进入时停止本实验。完整包装命令自动列在 command-records，不要直接在默认设备上启动子命令。

```bash
python docs/experiments/gemm-block32-tree/scripts/run_gpu.py \
  --gpu GPU-6deac1db-e8b2-558c-0285-ff9639a06a43 \
  --record runs/gemm_block32_tree_correctness_monitor_01 -- \
  python docs/experiments/gemm-block32-tree/scripts/correctness.py \
  --library runs/gemm_block32_tree_build_01/libblock32_tree.so \
  --output docs/experiments/gemm-block32-tree/results/correctness-01
```

12 组小矩阵全部通过后，执行 `full_accuracy.py --library runs/gemm_block32_tree_build_01/libblock32_tree.so --correctness docs/experiments/gemm-block32-tree/results/correctness-01 --output docs/experiments/gemm-block32-tree/results/accuracy-01 --raw runs/gemm_block32_tree_accuracy_raw_01`，外层 record 为 `runs/gemm_block32_tree_accuracy_monitor_01`。

九组完整精度评估结束，且基本语义/已有大矩阵 CPU 样本一致后，执行 `benchmark.py --library runs/gemm_block32_tree_build_01/libblock32_tree.so --accuracy docs/experiments/gemm-block32-tree/results/accuracy-01 --output docs/experiments/gemm-block32-tree/results/benchmark-01`，外层 record 为 `runs/gemm_block32_tree_benchmark_monitor_01`。原 NumPy 标准有一个失败作为实验结果保留；没有因此阻止已通过基本正确性的新算法成本测量，也没有将这个失败改成 PASS。

参数固定在已提交脚本中：N=1024/2048/4096、seed bases=42/1729/20261029、实际 seed=base+N、warmup=5、repeat=10、FP32、默认 stream 0、预分配 `out`。正式 benchmark 只运行一次，没有修改或重编译第二版 Kernel。

## 独立采集与失败保留

正式计时后，NCU 使用 `--set basic --section Occupancy --launch-count 1 --kernel-name regex:gemm_block32_tree --csv`，运行 `profile.py --method block32_tree --warmup 0 --iterations 1`。实际返回 ERR_NVGPUCTRPERM，记录为 BLOCKED，没有提升权限。

NSYS 使用 `--trace=cuda,nvtx --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop`，运行 `profile.py --method all --warmup 5 --iterations 3`。第一次被监控的进程组误判中断，保留 `nsys-monitor-01` 和旧监控代码。只修正进程归属识别后以同样参数运行第二次，成功生成 `runs/gemm_block32_tree_nsys_02.nsys-rep`。采集内的 warmup 不计入正式 benchmark。

```bash
python docs/experiments/a100-stage1/run_logged.py --log docs/experiments/gemm-block32-tree/logs/nsys-export.log -- nsys export --type sqlite --output runs/gemm_block32_tree_nsys_02.sqlite runs/gemm_block32_tree_nsys_02.nsys-rep
python docs/experiments/a100-stage1/run_logged.py --log docs/experiments/gemm-block32-tree/logs/nsys-stats.log -- nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum --format csv --output docs/experiments/gemm-block32-tree/results/nsys-02/summary runs/gemm_block32_tree_nsys_02.sqlite
```

## 归档、全位置核查和图像

```bash
python docs/experiments/a100-stage1/run_logged.py --log docs/experiments/gemm-block32-tree/logs/archive.log -- python docs/experiments/gemm-block32-tree/scripts/archive.py
python docs/experiments/a100-stage1/run_logged.py --log docs/experiments/gemm-block32-tree/logs/analysis-01.log -- python docs/experiments/gemm-block32-tree/scripts/analyze_results.py
python docs/experiments/a100-stage1/run_logged.py --log docs/experiments/gemm-block32-tree/logs/figures-01.log -- python docs/experiments/gemm-block32-tree/scripts/plot_results.py
python docs/experiments/gemm-block32-tree/scripts/index_evidence.py
```

分析从原矩阵重新核对全部 66,060,288 个改善/退化分类，不计算 checksum。公开分类图可用下面的只读命令检查某个位置：

```bash
python docs/experiments/gemm-block32-tree/scripts/read_change_map.py docs/experiments/gemm-block32-tree/results/accuracy-01/n4096-seed4138-improvement-map.bin.gz --n 4096 --row 4081 --col 3777
```

上面的分类图读取命令是复现示例；实际全位置核查由 analyze_results.py 完成。全部结果链接在 [README](README.md)。Git 提交范围仅包括独立实验实现、新实验文档/数据和根 README 入口；不合并 main，推送后停止。
