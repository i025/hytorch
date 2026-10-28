# 合成数据 GPU 训练对照：FP32 修复后最终结果

**PASS：39 次 throughput（TinyNet 7 配置 × 3，DonkeyNet 6 配置 × 3），加 13 次独立 phases 运行全部完成；输入/初始参数完全一致，最终参数数值对齐通过。** 每模型各组均在物理 A100 2（GPU-158426a4-6684-7863-8cbe-ed6595f97683）完成，UUID 映射逻辑 GPU 0。

固定 FP32、seed=42、64 个 synthetic 样本、batch=4、epoch=1、SGD lr=.001，无 momentum/weight decay，关闭 Dropout/TF32，顺序不 shuffle。warmup=2 后恢复初始化，不做哈希，实际初始权重与输入逐元素 assert_array_equal。最终参数 atol=1e-4、rtol=1e-3，没有放宽容差。

下表只用修复后独立重复的 throughput 数据；计时排除预热与初始化，包含每步 Loss 回读和日志。训练很短，结果限于该规模，不是实际自动驾驶准确率或长期训练吞吐量。真实数据未取得，Validation Loss 未测量。

| 模型 | 实现 | 平均秒 | Std 秒 | Samples/s | vs 原 HyTorch/CuPy | 最终参数最大绝对差 | 最大逐参数相对 L2 |
|---|---|---:|---:|---:|---:|---:|---:|
| tiny | hytorch:m=shared,c=im2col,p=native,t=16 | 0.063612898 | 0.0012318575 | 1006.0853 | 1.0121953 | 7.4505806e-09 | 7.3015325e-08 |
| tiny | hytorch:m=cupy,c=direct,p=cupy,t=16 | 0.065174278 | 0.00069787959 | 981.98249 | 0.98794616 | 7.4505806e-09 | 1.9207794e-07 |
| tiny | hytorch:m=cupy,c=im2col,p=cupy,t=16 | 0.061772331 | 0.0020831681 | 1036.0626 | 1.0423547 | 0 | 0 |
| tiny | hytorch:m=cupy,c=cupy,p=cupy,t=16 | 0.064388677 | 0.0029982263 | 993.96358 | 1 | 0 | 0 |
| tiny | hytorch:m=shared,c=cupy,p=cupy,t=16 | 0.062955941 | 0.0021521273 | 1016.584 | 1.0227578 | 7.4505806e-09 | 7.3015325e-08 |
| tiny | hytorch:m=cupy,c=cupy,p=native,t=16 | 0.06266139 | 0.0028673654 | 1021.3626 | 1.0275654 | 0 | 0 |
| tiny | torch | 0.050786963 | 0.0015564644 | 1260.1659 | 1.267819 | 7.4505806e-09 | 1.380921e-07 |
| donkey | hytorch:m=shared,c=im2col,p=cupy,t=16 | 0.12097629 | 0.0012427652 | 529.0293 | 1.0071009 | 4.1443855e-08 | 0.00082673821 |
| donkey | hytorch:m=cupy,c=direct,p=cupy,t=16 | 0.11439284 | 0.0013633228 | 559.47556 | 1.0650607 | 1.4901161e-08 | 3.8614112e-07 |
| donkey | hytorch:m=cupy,c=im2col,p=cupy,t=16 | 0.12377022 | 0.0042899697 | 517.08724 | 0.98436705 | 4.1443855e-08 | 0.00082678249 |
| donkey | hytorch:m=cupy,c=cupy,p=cupy,t=16 | 0.12183532 | 0.0012280587 | 525.29921 | 1 | 0 | 0 |
| donkey | hytorch:m=shared,c=cupy,p=cupy,t=16 | 0.12032645 | 0.0017534529 | 531.88639 | 1.0125399 | 7.4505806e-09 | 4.7291296e-07 |
| donkey | torch | 0.079894889 | 0.0019117412 | 801.05249 | 1.5249452 | 4.1443855e-08 | 0.00082673358 |

## Loss 与真实 dispatch

| 模型 | 实现 | Epoch Loss（第 1 次）| dispatch_counts |
|---|---|---:|---|
| tiny | cupy | 0.0861965334043 | `{}` |
| tiny | combined | 0.0861965430668 | `{"gemm_shared": 32, "conv_im2col": 16, "pool_native": 16}` |
| tiny | torch | 0.0861965325603 | `{}` |
| donkey | cupy | 0.000744870904782 | `{}` |
| donkey | combined | 0.000744870360904 | `{"gemm_shared": 128, "conv_im2col": 80}` |
| donkey | torch | 0.000744870113522 | `{}` |

DonkeyNet 最大逐参数相对 L2 约 8.27e-4 出现在很小参数量级，最大绝对差仅 4.15e-8；全部逐参数误差在 *-fixed-details-01/parameter-errors.csv，不能只看相对值或只报通过。

## 分阶段时间、硬件与数据位置

*-phases-fixed-details-01/phase-means.csv 和 phases.png：真实同步 data/transfer/forward/backward/update/other 平均耗时；与 throughput 分开，不填补 throughput 日志中的 null。*-throughput-fixed-details-01/loss.png 绘制每次真实逐步 MSE，epoch-results.csv 保留每次 epoch 结果。*-fixed-report-01/ 含每次 step 与硬件采样图；*-fixed-monitor-01/ 是外层每秒整卡时钟/功耗/温度/负载记录。短训练每秒采样通常只有一个点，不能推断精确显存峰值或持续利用率。

- *-throughput-fixed-01/、*-phases-fixed-01/：所有配置、原始步骤、Loss、硬件 JSONL、summary、stdout、commands.json。
- *-fixed-details-01/：逐参数初始/最终误差、每轮 Loss、phase 表与图。
- *-fixed-report-01/：比较 CSV、时间图和每次时间线。
- 原始 NPZ（初始参数、输入、训练后参数）留在服务器 /home/xymo/projects/hytorch/runs/synthetic_{tiny,donkey}_{throughput,phases}_fixed_01/，不上传权重/样本。
- 修复前报告 [README-before-fix.md](README-before-fix.md) 及所有原始目录保留；旧结果存在反向 FP64，不作为严格 FP32 结论。修复记录见 [analysis](../analysis/README.md)。

完整启动命令和起止时间见 *-fixed-monitor-01/launch.json、completion.json；主命令使用 run_cuda_comparison.py --model tiny/donkey --samples 64 --batch 4 --epochs 1 --repeats 3 --include-torch --mode throughput --no-fingerprints；phases 使用 --repeats 1 --mode phases，输出唯一目录。
