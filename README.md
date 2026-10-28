# HyTorch CUDA 开发与 GPU 验收

供队友在 NVIDIA GPU 上编译、验证和运行课程实验的源码包。

**当前状态：代码已在 Mac 完成主机侧检查，CUDA 编译和 GPU 正确性/性能尚未验收。不要把本仓库当作已获得加速结果的实验。**

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

```bash
python benchmarks/bench_cuda_gemm.py --sizes 1024 2048 4096 --tiles 16 32 --repeats 3 --output runs/gemm_01
python benchmarks/bench_cuda_ops.py --batches 1 4 16 --repeats 3 --output runs/ops_01
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
python scripts/convert_imagenet_weights.py --model alexnet --output weights_imagenet/alexnet.npz --verify --verify-device cuda
python scripts/convert_imagenet_weights.py --model vgg11 --output weights_imagenet/vgg11.npz --verify --verify-device cuda
python scripts/convert_imagenet_weights.py --model vgg16 --output weights_imagenet/vgg16.npz --verify --verify-device cuda
```

分类数据使用 `train/类别名/图片` 和 `val/类别名/图片`，例如：

```bash
python examples/finetune_imagenet.py --model alexnet --weights weights_imagenet/alexnet.npz --data /path/to/classification_dataset --device cuda --head-only --batch 1 --epochs 3 --output runs/transfer_alexnet_01
```

另外两个模型可使用相同入口。默认冻结特征并训练分类器；`--head-only` 只更新最后一层；`--train-all` 全量训练。使用新模型库的 RGB/ImageNet 预处理，不要混用原 VGG16 的 BGR 预处理。

## 需要回传哪些结果

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

文档中提到的 `.cuda_baseline_backup/` 和既有 CPU 烟测输出保留在开发者本机，未随仓库发布；验证记录描述的是本机已做的检查。当前仓库只声明实现状态，不包含实测 GPU 加速结论。
