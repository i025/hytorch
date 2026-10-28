# 合成数据 GPU 训练对照

**PASS：TinyNet 7 种配置 × 3 次，DonkeyNet 6 种配置 × 3 次，39 次 GPU 训练全部完成；每组初始化与输入数组完全相同，训练后参数全部满足 atol=1e-4 / rtol=1e-3。** 全部使用同一张物理 A100 0（UUID 见 monitor），逻辑 0；FP32、64 个 synthetic 样本、batch=4、epoch=1、SGD lr=.001、关闭 Dropout/TF32、固定顺序、warmup=2 后恢复初始参数。

没有真实数据集。这些 Loss 只证明合成训练执行链路，不是自动驾驶准确率或真实任务效果；Validation Loss 未测量，合成输入没有验证集。

throughput 模式没有同步分段计时，因此 steps 中 forward/backward/update/data/transfer 为 null，不等于零。另行 phases 实验结果将分开保存，不能混入吞吐量统计。下表 wall time 计入每步 Loss 回读与日志开销，排除预热、初始化及验证；短时任务硬件每秒采样可能只取得很少的样本，不能当作精确内存峰值。

| 模型 | 实现 | 平均训练秒 | Samples/s | vs 原 HyTorch | 最终参数最大绝对差 | 最大逐参数相对 L2 |
|---|---|---:|---:|---:|---:|---:|
| tiny | hytorch:m=shared,c=im2col,p=native,t=16 | 0.064342204 | 994.6815 | 1.0139693 | 1.4901161e-08 | 1.494998e-07 |
| tiny | hytorch:m=cupy,c=direct,p=cupy,t=16 | 0.067738133 | 944.81494 | 0.96313578 | 1.4901161e-08 | 1.6576001e-07 |
| tiny | hytorch:m=cupy,c=im2col,p=cupy,t=16 | 0.068206002 | 938.33384 | 0.95652901 | 0 | 0 |
| tiny | hytorch:m=cupy,c=cupy,p=cupy,t=16 | 0.065241019 | 980.97793 | 1 | 0 | 0 |
| tiny | hytorch:m=shared,c=cupy,p=cupy,t=16 | 0.063572509 | 1006.7245 | 1.0262458 | 1.4901161e-08 | 1.494998e-07 |
| tiny | hytorch:m=cupy,c=cupy,p=native,t=16 | 0.063496958 | 1007.9223 | 1.0274669 | 0 | 0 |
| tiny | torch | 0.047620231 | 1343.9666 | 1.3700274 | 7.4505806e-09 | 1.9207799e-07 |
| donkey | hytorch:m=shared,c=im2col,p=cupy,t=16 | 0.13388489 | 478.02257 | 1.0121513 | 4.1443855e-08 | 0.000826739 |
| donkey | hytorch:m=cupy,c=direct,p=cupy,t=16 | 0.12951081 | 494.16724 | 1.0463355 | 1.4901161e-08 | 3.9505604e-07 |
| donkey | hytorch:m=cupy,c=im2col,p=cupy,t=16 | 0.1308679 | 489.04277 | 1.0354851 | 4.1443855e-08 | 0.00082681082 |
| donkey | hytorch:m=cupy,c=cupy,p=cupy,t=16 | 0.13551176 | 472.28374 | 1 | 0 | 0 |
| donkey | hytorch:m=shared,c=cupy,p=cupy,t=16 | 0.13701094 | 467.11599 | 0.98905796 | 7.4505806e-09 | 3.9535281e-07 |
| donkey | torch | 0.081252366 | 787.66937 | 1.6677885 | 4.1443855e-08 | 0.00082673469 |

## 原始证据

- tiny-throughput-01/、donkey-throughput-01/：39 次运行各自的 config.json、summary.json、steps.jsonl、losses.jsonl、hardware.jsonl、完整日志和所有命令。
- tiny-report-01/、donkey-report-01/：由原始记录生成的比较 CSV、吞吐量/时间图、每次运行硬件与 step 时间线。
- tiny-monitor-01/、donkey-monitor-01/：外层启动/结束与 GPU 负载日志，已去除 PID。
- 每次实际 initial_weights.npz、comparison_inputs.npz、final_weights.npz 留在服务器 runs/synthetic_{tiny,donkey}_comparison_01/，不上传权重或样本。report_cuda.py 逐数组 assert_array_equal 初始权重和输入，最终参数逐数组 allclose，未使用任何哈希。
- 正式报告命令的完整输出：服务器 runs/synthetic_{tiny,donkey}_report_01.log。

修改 benchmarks/bench_cuda_training.py 保存实际初始化及输入数组；scripts/run_cuda_comparison.py 传递无哈希参数；scripts/report_cuda.py 使用实际数组匹配并导出初始与最终参数误差。先用 CPU TinyNet 两条路径验证新接口，日志 runs/comparison_cpu_{hytorch,torch,report}_01.log，最终参数最大绝对差 7.4505806e-9，不能视为 GPU 性能记录。

每个模型实际命令都保存在 commands.json；主要参数如下：

```bash
python scripts/run_cuda_comparison.py --model tiny --samples 64 --batch 4 --epochs 1 --repeats 3 --include-torch --mode throughput --no-fingerprints --output runs/synthetic_tiny_comparison_01
python scripts/run_cuda_comparison.py --model donkey --samples 64 --batch 4 --epochs 1 --repeats 3 --include-torch --mode throughput --no-fingerprints --output runs/synthetic_donkey_comparison_01
```

服务器项目根目录：/home/xymo/projects/hytorch。所有 npz 保留原样，未上传。
