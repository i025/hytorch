# Udacity 驾驶数据：本地准备记录

2026-11-06 下载，供后续 HyTorch 算子优化与数据驱动的训练性能实验使用。

- 来源：<https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip>
- 压缩包：333,137,665 字节；ZIP 内文件未压缩大小合计 329,338,837 字节。
- SHA-256：`7ca6aba7f72df475de32959b3b7a5a825b345c94307e715639dc2a13eb61dd0c`。
- ZIP 全量 CRC 校验通过。保留原包及原始解压文件，包括附带的 macOS 元数据。
- `driving_log.csv` 有 8,036 条记录，每条包含 center/left/right 图片和 steering/throttle/brake/speed。
- 三路相机共 24,108 张图片；中心相机的 8,036 张图片全部通过 PIL verify，尺寸为 320×160。
- 转向角范围：[-0.9426954, 1.0]。

## 本地位置

仓库根目录：`/home/xieminhui_pubuse/hytorch`。

- 原始包：`data/udacity/data.zip`
- 原始 CSV：`data/udacity/extracted/data/driving_log.csv`
- 原始图片：`data/udacity/extracted/data/IMG/`
- HyTorch 兼容入口：`data/udacity/donkey_center/`
- 下载及校验信息：`data/udacity/download_manifest.json`
- 每个样本的原始图片、标签及软链接映射：`data/udacity/center_manifest.json`
- 本地准备脚本：`data/udacity/inspect_archive.py`、`data/udacity/prepare_center.py`

兼容入口按 CSV 顺序建立 `{index}_{angle}.jpg` 相对软链接，标签取自 CSV 的 steering，
不复制或重新编码图片。仅选择中心相机；原始左右相机保留，未添加人为转向角修正。
数据和本地准备产物均在 Git 忽略的 `data/` 下。

已验证原有 `DonkeyDataset` 可以加载，并核对全部标签；抽查输出为 `(3,66,200)` FP32，
像素范围 [-1,1]，标签 `(1,)` FP32。使用原有预处理：裁去顶部 40 行后缩放，
未为该数据集重新优化裁剪策略。默认 seed=42 的随机划分为 7,233 训练 / 803 验证样本。

## 后续性能实验入口

以下命令尚未执行；本次只完成下载、准备和加载验证。输出目录必须不存在。

```bash
source /home/xieminhui_pubuse/hytorch-env/bin/activate
export CUDA_VISIBLE_DEVICES=1 CUDA_PATH=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
python benchmarks/bench_cuda_training.py \
  --data data/udacity/donkey_center --model donkey --device cuda \
  --batch 4 --epochs 1 --matmul shared --conv im2col --pool native \
  --mode phases --output runs/udacity_profile_01
```

此数据来源与原 DonkeyCar 数据不同；后续报告应标为 Udacity 中心相机数据，不能与合成实验混为同一基线。
相邻帧高度相关，原加载器的随机划分适合保持性能对照条件，不能据此声称独立场景泛化能力；
若研究任务精度，应另行按连续片段或驾驶场景划分。
