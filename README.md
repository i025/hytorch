# HyTorch CUDA 开发与 GPU 验收

供队友在 NVIDIA GPU 上编译、验证和运行课程实验的源码包。

**A100 空闲窗口实测已完成（2026-10-28）：主机 25 passed / 0 skipped，GPU 39 passed / 0 skipped；Conv/Pooling、修复后的 HyTorch/PyTorch 合成训练与三个官方预训练 CNN 验证通过。GEMM 2048/4096 未达到既定数值容差，Nsight Compute 受权限阻止，真实训练缺少数据。详见 [完整实验报告与原始数据](docs/experiments/a100-gpu-window/README.md)。**

[阶段 1 历史验收记录](docs/experiments/a100-stage1/README.md) 保留；阶段 1 当时未安装 PyTorch 的跳过记录不改写，后续已实际补齐。实验没有自动推进 Kernel 优化。

## 队友第一次运行：只做这四步

### 1. 获取代码，确认环境

```bash
git clone https://github.com/i025/hytorch.git
cd hytorch
nvidia-smi
nvcc --version
python --version
python -m pip install -r requirements-cuda.txt
```

需要 NVIDIA GPU、驱动、CUDA Toolkit（含 nvcc）和匹配的 CuPy。比如 **CUDA 12.x 环境**可安装：

```bash
python -m pip install cupy-cuda12x
python -c "import cupy as cp; cp.show_config(); print(cp.cuda.runtime.getDeviceCount())"
```

其他 CUDA 版本请选匹配的 CuPy 包，不要同时安装多个 CuPy 包。Windows 原生编译请使用 x64 Native Tools Command Prompt，确保 `cl`、`nvcc` 可用；Linux/WSL2 直接使用终端。详细配置见 [完整指南](docs/CUDA开发与GPU验收指南.md)。

### 2. 编译 CUDA C 动态库

```bash
python scripts/build_cuda.py --device 0
```

自动读取 GPU 架构；失败时先回传 `hytorch/backends/cuda/build/build.log`，不要继续跑性能实验。

### 3. 运行 GPU 正确性测试

```bash
python scripts/validate_cuda.py --device 0 --output runs/gpu_check_01
```

检查 `runs/gpu_check_01/environment.json`：必须同时是 `status=PASSED`、`gpu_status=EXECUTED`。缺 GPU/动态库会失败，不能用普通 pytest 的 skipped 代替通过。

### 4. 小型训练烟测，不需要数据集

```bash
python benchmarks/bench_cuda_training.py --device cuda --model tiny --samples 8 --batch 2 --matmul shared --conv im2col --pool native --output runs/gpu_smoke_01
```

检查 `runs/gpu_smoke_01/summary.json`：训练完成、loss 有限、`dispatch_counts` 有 `gemm_shared`、`conv_im2col` 和 `pool_native`。以上通过后再跑后续实验。

每次重跑换一个输出目录，例如 `_02`；脚本故意拒绝覆盖原实验。

## 第二阶段：算子性能

阶段 2A 七配置 GEMM 基线已测量并保留大矩阵失败记录，见 [GEMM 报告](docs/experiments/a100-stage2a-gemm/README.md)；阶段 2B 全部数值检查通过，见 [Conv/Pooling 报告](docs/experiments/a100-stage2b-ops/README.md)。

```bash
python benchmarks/bench_cuda_gemm.py --sizes 1024 2048 4096 --tiles 16 32 --repeats 10 --warmup 5 --seed 42 --output runs/gemm_01
python benchmarks/bench_cuda_ops.py --batches 1 4 16 --repeats 10 --output runs/ops_01
python scripts/report_cuda.py --gemm runs/gemm_01 --output outputs/gemm_report_01
```

比较 NumPy、CuPy、CUDA basic/blocked/shared；报告误差、时间、加速比和 H2D/D2H。笔记本先以 `--sizes 128 256` 小测，确认显存与执行时间可接受后再跑完整规模。

## 第三阶段：真实 DonkeyCar 训练

数据不随代码上传，队友需单独取得现有 `{index}_{angle}.jpg` 数据。将命令中的 `/path/to/donkey_images` 替换成真实路径。

```bash
python scripts/run_cuda_comparison.py --data /path/to/donkey_images --model donkey --batch 4 --epochs 1 --repeats 3 --mode throughput --output runs/driving_01
python scripts/report_cuda.py --comparison runs/driving_01 --output outputs/driving_report_01
```

需要 PyTorch 对照时，先安装适合目标 GPU 环境的 PyTorch，再给第一条命令加 `--include-torch`。各组使用同一数据加载器、固定初始化与顺序、FP32、SGD、关闭 dropout；不是沿用旧脚本的 AdamW 设置。

单独定位耗时阶段：

```bash
python benchmarks/bench_cuda_training.py --data /path/to/donkey_images --model donkey --batch 4 --conv im2col --mode phases --epochs 1 --output runs/driving_profile_01
```

日志包含加载、传输、前向、反向、更新、step 时间和 samples/sec；监控包含 GPU 利用率、显存、温度、功耗与 CPU。显存峰值是整卡采样峰值，不是精确的本进程峰值。Nsight Systems/Compute 命令见完整指南。

## 第四阶段：预训练与迁移训练

需要兼容的 `torch`、`torchvision`；权重在运行时从官方来源下载，未放入 Git。

```bash
python scripts/convert_imagenet_weights.py --model alexnet --output weights_imagenet/alexnet.npz --verify --verify-device cuda --no-checksums
python scripts/convert_imagenet_weights.py --model vgg11 --output weights_imagenet/vgg11.npz --verify --verify-device cuda --no-checksums
python scripts/convert_imagenet_weights.py --model vgg16 --output weights_imagenet/vgg16.npz --verify --verify-device cuda --no-checksums
```

分类数据使用 `train/类别名/图片` 和 `val/类别名/图片`，例如：

```bash
python examples/finetune_imagenet.py --model alexnet --weights weights_imagenet/alexnet.npz --data /path/to/classification_dataset --device cuda --head-only --batch 1 --epochs 3 --output runs/transfer_alexnet_01
```

另外两个模型可使用相同入口。默认冻结特征并训练分类器；`--head-only` 只更新最后一层；`--train-all` 全量训练。使用新模型库的 RGB/ImageNet 预处理，不要混用原 VGG16 的 BGR 预处理。

## 需要回传哪些结果

### 长期 DonkeyNet 性能基线

阶段 4C 建立了[同一 GPU 窗口的 A–E 性能表与原始数据](docs/experiments/conv-dinput-pruning/README.md)：A 为当前代码复现的 CuPy 无自研 Native CUDA C 基线，B 为 Combined，C 加 Fused SGD，D 加显式 dInput 剪枝，E 为 PyTorch 对照。

后续训练实验保留这些配置并加入新版本，在同一 GPU/窗口、相同输入和初始权重下至少重复 3 次；分别报告增量和累计收益，不跨历史窗口拼接加速比。吞吐、同步阶段和 Nsight 核时分开统计，使用 `--no-fingerprints`，保留失败及退化记录。阶段 4C 的首层梯度剪枝已验证，端到端稳定收益仍待复核。

阶段 4D 的[显式可选 Conv2d dInput CUDA gather](docs/experiments/conv-backward-fusion/README.md)已完成 [GPU 补测](docs/experiments/conv-backward-fusion/supplement-01/README.md)：92 项 GPU 回归通过，同一 GPU6 窗口中 F/D 吞吐 +4.69%、F/A +13.14%，F 为 PyTorch 的55.56%。独立 GPU0 的完整 Nsight 验证了 Launch/Copy/Col2im 减少，但四层纯设备 dInput 全部变慢，Backward 核时 +39.87%；端到端正收益主要支持减少主机提交和间隙的解释。不同窗口不混算计时；原型默认关闭，没有进入 4E。全部原始数据及负结果见[补测数据索引](docs/experiments/conv-backward-fusion/supplement-01/data-index.md)。

阶段 4E 的[通用 Shape-Aware 调度实验](docs/experiments/conv-shape-dispatch/README.md)已完成：100种卷积形状、最终188项GPU回归和两个CNN的匹配测量；仅小型1×1规则通过全新24形状验证，较宽规则失败记录完整保留。Auto默认关闭；本轮DonkeyNet G/D吞吐−5.81%，独立CNN−0.52%，没有证实端到端增量收益。相对当前代码复现的无Native基线G/A分别+0.67%/+6.77%，不可归为Auto独立贡献。详见[完整原始数据与23份Nsight索引](docs/experiments/conv-shape-dispatch/data-index.md)，等待人工审查后再决定后续工作。

第一次先回传：

1. `nvidia-smi`、`nvcc --version`、CuPy 配置信息。
2. `hytorch/backends/cuda/build/build.log` 和 `build.json`。
3. `runs/gpu_check_01/` 的 `environment.json`、`tests.log`、`tests.xml`。
4. `runs/gpu_smoke_01/` 的 `config.json`、`summary.json`、`steps.jsonl`、`hardware.jsonl`。

完整实验再回传 GEMM/ops CSV、训练 JSON/JSONL 与图表；Nsight 报告和最终权重可单独传递。若失败，保留完整报错与对应命令，先定位失败关卡。

## 源码范围与主机侧自检

包含完整 `hytorch/` Python/C/CUDA 源码、DonkeyNet/数据读取器、CUDA benchmarks、构建/验证/报告脚本、迁移训练示例与相关测试。不包含数据集、模型权重、编译库、历史 runs/outputs、本地备份或 IDE 缓存。

```bash
python -m pytest tests/test_cuda_host.py tests/test_conv2d_gemm_cpu.py tests/test_gemm_blocked_packed.py -q
```

未安装 PyTorch 时，两项跨框架对齐测试会跳过；这不影响其余主机契约测试，但不能声称完成 PyTorch 对齐检查。GPU 验收仍必须运行 `validate_cuda.py`。

- [完整开发与 GPU 验收指南](docs/CUDA开发与GPU验收指南.md)
- [Mac 上的验证记录](docs/CUDA验证记录.md)
- [核心修改清单](docs/CUDA修改清单.json)
- `UPLOAD_MANIFEST.json`：本次上传文件及 SHA256，用于确认源码包内容。

文档中提到的 `.cuda_baseline_backup/` 和既有 CPU 烟测输出保留在开发者本机，未随仓库发布；验证记录描述的是本机已做的检查。当前 A100 实测、失败与权限限制统一以最新实验报告为准，历史实现文档不代表最新验收状态。
