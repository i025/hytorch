# A100 阶段 2B：Conv2d / Pooling

**PASS：24/24 行数值检查通过，18 组 GPU 算子配置实际执行。** 物理 A100 0 / GPU-f099fdbd-adf2-6157-3462-4145bc2a8359，进程内逻辑 GPU 0。精确起止时间、环境及命令见 monitor-01/launch.json 和 completion.json。

输入 FP32 [batch,3,66,200]，batch=1/4/16；Conv 权重 [24,3,5,5]、stride=2、pad=0，Pooling 2×2、stride=2。每组 1 次预热、10 次计时，seed=42+batch，atol=rtol=1e-4。CUDA Event 口径包含算子 workspace 分配/布局处理，不包括 H2D/D2H；不能和 GEMM 预分配输出时间直接混用。未修改内核。

| 算子 | batch | 实现 | Mean ms | Std ms | vs CuPy | 最大绝对误差 | 相对 L2 |
|---|---:|---|---:|---:|---:|---:|---:|
| conv | 1 | numpy | 0.9903345 | 0.1361025 | 0.1543753 | 0 | 0 |
| conv | 1 | cupy | 0.1528832 | 0.02869251 | 1 | 1.335144e-05 | 1.72599e-07 |
| conv | 1 | cuda_direct | 0.0768 | 0.007028483 | 1.990667 | 9.536743e-06 | 1.154002e-07 |
| conv | 1 | cuda_im2col_16 | 0.155136 | 0.008152794 | 0.9854785 | 0 | 0 |
| conv | 1 | cuda_im2col_32 | 0.1506304 | 0.003784032 | 1.014956 | 0 | 0 |
| pool | 1 | numpy | 0.3399863 | 0.009275294 | 0.4288925 | 0 | 0 |
| pool | 1 | cupy | 0.1458176 | 0.01068761 | 1 | 0 | 0 |
| pool | 1 | cuda_native | 0.0516096 | 0.004734577 | 2.825397 | 0 | 0 |
| conv | 4 | numpy | 5.627174 | 1.549925 | 0.04882366 | 0 | 0 |
| conv | 4 | cupy | 0.2747392 | 0.3149802 | 1 | 0 | 0 |
| conv | 4 | cuda_direct | 0.1153024 | 0.003209259 | 2.382771 | 1.144409e-05 | 1.137914e-07 |
| conv | 4 | cuda_im2col_16 | 0.1738752 | 0.00632711 | 1.580094 | 0 | 0 |
| conv | 4 | cuda_im2col_32 | 0.1753088 | 0.002967831 | 1.567173 | 0 | 0 |
| pool | 4 | numpy | 1.165074 | 0.01161948 | 0.1217296 | 0 | 0 |
| pool | 4 | cupy | 0.141824 | 0.006813855 | 1 | 0 | 0 |
| pool | 4 | cuda_native | 0.0553984 | 0.008668119 | 2.560074 | 0 | 0 |
| conv | 16 | numpy | 14.83164 | 0.9852226 | 0.01921427 | 0 | 0 |
| conv | 16 | cupy | 0.2849792 | 0.04184099 | 1 | 0 | 0 |
| conv | 16 | cuda_direct | 0.2807808 | 0.005370986 | 1.014953 | 1.144409e-05 | 1.123523e-07 |
| conv | 16 | cuda_im2col_16 | 0.3068928 | 0.00434581 | 0.9285952 | 0 | 0 |
| conv | 16 | cuda_im2col_32 | 0.3202048 | 0.001743811 | 0.8899904 | 0 | 0 |
| pool | 16 | numpy | 4.455699 | 0.06302741 | 0.03380623 | 0 | 0 |
| pool | 16 | cupy | 0.1506304 | 0.008283227 | 1 | 0 | 0 |
| pool | 16 | cuda_native | 0.054272 | 0.003413334 | 2.775472 | 0 | 0 |

当前输入下 Direct Conv 比 Im2col 更快，batch 16 对 CuPy 仅约 1.015×，不宜将这点差异解释为稳定优势。Native Pooling 在三个 batch 都快于 CuPy；测量仅覆盖此输入与前向路径，不代表反向或端到端训练。

## 文件与复现

- results-01/ops.csv：全部逐次耗时、均值、标准差、加速比及数值误差。
- results-01/config.json：输入/设备/版本/计时口径。
- monitor-01/：完整终端输出、实际命令、开始结束时间与每秒 GPU 监控，发布版移除 PID；服务器原始监控 runs/gpu_window_ops_monitor_01/。
- figures/：Conv 延迟、相对 CuPy 加速比、Direct vs Im2col、Pooling 延迟/加速比、Tile 对比五张图。
- 绘图源码 ../a100-gpu-window/scripts/plot_ops.py。

```bash
python benchmarks/bench_cuda_ops.py --device 0 --batches 1 4 16 --repeats 10 --output runs/ops_stage2b_01
python docs/experiments/a100-gpu-window/scripts/plot_ops.py --input runs/ops_stage2b_01 --output docs/experiments/a100-stage2b-ops/figures
```

命令通过现有 run_monitored.py 包装，并以物理 UUID 设置 CUDA_VISIBLE_DEVICES。服务器项目根路径 /home/xymo/projects/hytorch，原始数值目录 runs/ops_stage2b_01/。
