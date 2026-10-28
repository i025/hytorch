# HyTorch CUDA 开发与 GPU 验收指南

这次是在 Mac 上完成的源码开发和主机侧验证。**CUDA 编译、真实 GPU 正确性、性能提升、Nsight 指标、真实 ImageNet 权重对齐与真实自动驾驶训练尚未验证。** 所有新输出目录要求不存在，避免把旧实验和新实验混在一起。

## 1. 截图要求 → 实现 → 后续验证

| 要求 | 本次实现 | GPU 上要做的事 |
|---|---|---|
| CUDA C/C++ 卷积、池化 | `kernels.cu` 的直接卷积、im2col、矩阵乘、布局转换与偏置融合、最大池化 | 编译；前向与输入/权重梯度对齐；检查越界 |
| 五对象矩阵乘对比 | NumPy、CuPy、CUDA basic/blocked/shared | 1024/2048/4096，各至少 3 次；tile 16/32 |
| 正确性与性能同报 | 平均/标准差/中位数、原始耗时、相对 NumPy/CuPy 加速比、最大绝对误差、相对 L2 误差 | 先看 `correct`，再讨论加速比 |
| 传输与计算分开 | GEMM 的 CUDA Event 计算计时，独立 H2D/D2H 主机墙钟计时 | 分析设备常驻与含传输场景 |
| 接回框架 | `MatMul`、`Linear`、`Conv2d`、`Pooling` 的 forward 路由 | 看 `dispatch_counts`，证明真实进入 CUDA C |
| 沿用原反向 | 卷积/池化反向仍走原实现；矩阵乘反向显式走原 dot | 检查梯度与最终权重漂移 |
| GPU/CPU 监控 | 利用率、已用/总显存、采样峰值、温度、功耗、CPU、进程 RSS | 整个训练跑足够长，检查是否采到足够样本 |
| 训练六项 | step、forward（含 loss）、backward（含清梯度）、update、data、samples/sec；另列 transfer 与 other | 确认各阶段占比的共同分母 |
| 定位瓶颈 | JSON 观察与下一步建议、NVTX、Nsight 操作步骤 | 先 Systems 定位，再 Compute 看访存/调度 |
| 自动驾驶端到端对比 | 同一 DonkeyNet、数据预处理、初始权重、样本顺序、SGD、FP32；PyTorch 对照 | 用真实数据跑 1 个完整 epoch，再延长多轮 |
| 至少三种预训练 CNN | 新模型库 AlexNet、VGG11、VGG16，torchvision ImageNet V1 转换、校验和、logits 对齐入口 | 下载/转换真实权重；每个模型执行 `--verify` |
| 迁移训练 | 加载特征与分类器权重、替换分类头、冻结特征或只训练最后一层、验证集准确率 | 用实际分类数据训练，检查冻结权重不变 |

截图中的 softmax/交叉熵融合、Adam 多张量融合、混合精度是**后续优化候选**，本次不在没有 GPU 瓶颈证据的情况下同时引入。已有一项可单独对比的融合：卷积 GEMM 输出布局转换与偏置相加。

截图中“端到端收益一定小于单算子收益”在实验中不能直接当成事实；两者的计时范围、内存行为与调用形状不同。报告实际测量值，并解释差异，不能预填加速比。

## 2. 修改边界与接口

原文件已保存在 `.cuda_baseline_backup/`。原 CPU/CuPy 路径仍是默认选项，开启方式：

```python
from hytorch.backends.cuda import backend, dispatch_stats

with backend(matmul="shared", conv="im2col", pool="native", tile=16):
    y = model(x)  # x 和模型参数应位于同一个 GPU，采用 float32
    loss = loss_fn(y, target)
    model.cleargrads()
    loss.backward()
    optimizer.update()
print(dispatch_stats())
```

支持的选择：

- `matmul`: `cupy`（原路径）/ `basic` / `blocked` / `shared`。
- `conv`: `cupy`（原 im2col+tensordot）/ `direct` / `im2col`（CUDA im2col + shared GEMM）。
- `pool`: `cupy`（原路径）/ `native`。
- `tile`: 16 或 32。blocked 使用输出分块与 K 轴分段，仍从 global memory 读取；shared 才把输入块缓存进 shared memory。这是对照实验，不保证每一级都加速。

CUDA C 后端目前只支持 FP32、二维矩阵乘、NCHW 普通卷积（无 groups/dilation）、普通最大池化；不改 Transformer BMM。CPU 或非 FP32 输入沿用原路径，并增加 fallback 计数。显式选中 CUDA C 后，如果动态库缺失、维度非法、输入跨设备或权重 dtype 不匹配，会报错，不伪装成 CUDA 成功。

`ctypes` 明确声明所有 `argtypes/restype`，传递 CuPy 设备指针及**当前 stream**，不经 NumPy 中转。ABI 内不分配内存、不做 H2D/D2H、不全局同步。Python wrapper 负责 contiguous 整理与 workspace 分配，因此真实框架计时包含这些开销。

最大池化保留原 HyTorch 的 **zero padding** 语义、相等值取第一个索引，避免改变现有网络行为。它与某些框架负无穷 padding 的定义不同；本次预训练模型中的 pooling 均无 padding。im2col 会额外占用 `N*OH*OW*C*KH*KW*4` 字节，实测显存不足时先减 batch。

另修复两个与任务直接相关的问题：

1. `Linear/Conv2d` 初始化的缩放标量显式保持目标 dtype，避免 NumPy 2 将 FP32 提升成 FP64。
2. `Optimizer.update()` 在 hooks 后再次过滤无梯度参数，使 `FreezeParam` 真正排除更新。

## 3. 目标机器准备

支持目标是带 NVIDIA GPU 的 Linux/WSL2，或 Windows x64。Mac 可运行主机测试，但不能执行这些 CUDA kernel。

在复制到 GPU 机器的项目根目录执行。先确认驱动、Toolkit、GPU：

```bash
nvidia-smi
nvcc --version
python --version
python -m pip install -r requirements-cuda.txt
```

CuPy 按服务器实际 CUDA 安装环境选择 **一个** wheel。例如使用 CUDA 12.x 的环境：

```bash
python -m pip install cupy-cuda12x
python -c "import cupy as cp; cp.show_config(); print(cp.cuda.runtime.getDeviceCount())"
```

不要同时安装 `cupy` 与多个 `cupy-cuda*` 包。CUDA 13 或其他环境按 [CuPy 安装文档](https://docs.cupy.dev/en/stable/install.html) 选包；不要仅凭 `nvidia-smi` 的 CUDA Version 字段判断已安装的 Toolkit。PyTorch/torchvision 是可选对照与权重转换依赖，按目标机器匹配安装；其官方入口为 [PyTorch 安装页](https://pytorch.org/get-started/locally/)。本地并未替你安装 GPU 软件。

Windows 原生编译需在 x64 Native Tools Command Prompt 中可用 `cl` 与 `nvcc`，`CUDA_PATH` 指向匹配的 Toolkit；动态库名称为 `hytorch_cuda.dll`。Linux/WSL2 输出 `libhytorch_cuda.so`。以下 Python 命令通用，带反斜线换行的 bash 命令在 Windows 请写成一行。

## 4. 必须按顺序走的验收关卡

### A. 主机侧回归

```bash
python -m pytest tests/test_cuda_host.py tests/test_conv2d_gemm_cpu.py tests/test_gemm_blocked_packed.py -q
```

### B. 编译

```bash
python scripts/build_cuda.py --device 0
```

自动从 GPU 读取 compute capability。也可显式指定目标，例如 `--arch 89`，不要拿示例架构覆盖真实设备。若 nvcc 不支持目标架构，需要匹配的 Toolkit。

编译命令、编译器版本与输出保存在 `hytorch/backends/cuda/build/build.json` 和 `build.log`。先查编译错误，再继续。Mac 只能查看构建命令：

```bash
python scripts/build_cuda.py --dry-run --arch 89
```

### C. GPU 正确性关卡

```bash
python scripts/validate_cuda.py --device 0 --output runs/gpu_acceptance_01
```

覆盖非整块/矩形 GEMM、tile 16/32、非默认 stream、非 contiguous 输入、输出别名拒绝、卷积 stride/padding/bias、前向与梯度、池化负数/重复值/NaN 等。

**只有 `environment.json` 同时为 `status=PASSED`、`gpu_status=EXECUTED` 才进入性能测试。** 没 GPU、没 CuPy、没动态库会失败；普通 pytest 在 Mac 上的 skipped 不算通过。

有 Compute Sanitizer 的机器还可检查非法访存：

```bash
compute-sanitizer --tool memcheck --error-exitcode 1 python scripts/validate_cuda.py --output runs/gpu_memcheck_01
```

### D. 训练烟测

```bash
python benchmarks/bench_cuda_training.py --device cuda --model tiny --samples 8 --batch 2 --matmul shared --conv im2col --pool native --output runs/gpu_tiny_01
```

查看 `summary.json` 的 `dispatch_counts` 是否有 `gemm_shared`、`conv_im2col`、`pool_native`，loss 是否有限。合成数据只用于验证执行链路，不能用于自动驾驶任务精度报告。

### E. 算子正式对比

```bash
python benchmarks/bench_cuda_gemm.py --sizes 1024 2048 4096 --tiles 16 32 --repeats 3 --output runs/gemm_full_01
python benchmarks/bench_cuda_ops.py --batches 1 4 16 --repeats 3 --output runs/ops_full_01
python scripts/report_cuda.py --gemm runs/gemm_full_01 --output outputs/gemm_full_01
```

需要更稳健的统计时把 repeats 提高到 10。笔记本先用 GEMM 128/256 做小测，再上 4096；basic kernel 可能很慢。

GEMM 表含 H2D/D2H（pageable memory 的同步墙钟耗时）和 `transfer_plus_compute_ms`。后者是三部分之和，**不是额外测得的端到端应用时间**。CuPy/CUDA C 使用预分配输出；CUDA event 不含首次编译/上下文初始化。卷积/池化表测完整前向函数，包括 workspace 与 layout 整理，无主机传输。

误差定义：最大绝对误差 `max(abs(y-ref))`；相对 L2 误差 `||y-ref||₂ / max(||ref||₂, 1e-30)`；正确性采用逐元素 `atol + rtol*abs(ref)`。默认 `atol=rtol=1e-4` 只是显式 FP32 验收设置；若老师要求更严，GEMM 可传 `--atol 1e-5 --rtol 1e-5`，并报告是否通过，不能隐瞒阈值。大 K 累加顺序改变可能带来不同舍入误差。

`summary.json` 给出每个规模的实测最快实现，以及通过误差门槛且相对 CuPy 至少 1.5 倍的**首个已测规模**。没达到就写 null；不能推断成一个连续区间或通用临界点。截图的 1.5 倍是判定示例，可通过 `--benefit-speedup` 修改。

### F. 自动驾驶、单项消融与框架对照

数据沿用现有 `{index}_{angle}.jpg` 格式，默认种子划分 90% 训练、10% 验证。

```bash
python scripts/run_cuda_comparison.py --data /path/to/donkey_images --model donkey --batch 4 --epochs 1 --repeats 3 --include-torch --mode throughput --output runs/driving_comparison_01
python scripts/report_cuda.py --comparison runs/driving_comparison_01 --output outputs/driving_comparison_01
```

每个配置使用独立进程：原 CuPy、direct 卷积、im2col 卷积、shared GEMM、组合版本，以及可选 PyTorch。tiny 模型另外比较 native pool；原 DonkeyNet 没有 pooling 层，因此不把“切换未使用的 pool”作为自动驾驶优化。顺序按重复轮次轮换。默认 3 次，每次新建相同初始化与固定样本顺序。

这里为了可比性**显式使用 SGD、关闭 dropout、采用 FP32**；并非直接复现旧脚本的 AdamW/dropout 设置。PyTorch 也使用相同 HyTorch 数据加载器，所以结果描述的是固定数据管线下的训练后端比较，不能泛化成两个框架各自最优配置之争。保留库原来的训练脚本。

报告器检查初始权重摘要、数据摘要、batch、epoch、workers、seed、lr、计时模式、设备等是否匹配，不匹配会拒绝拼表；同时比较最终权重是否接近。真实数据摘要覆盖文件名/标签/文件大小而不是图像全部内容，跨机器复制后需自行确认数据未变。

`train_wall_s` 排除验证，包含训练循环、数据加载器启动/结束、标量读取与日志开销；`samples/sec` 的分母为对应训练范围。验证另记 `val_wall_s/val_loss`。监控覆盖训练+验证、排除预热；需要精确对齐训练区域时按 `steps.jsonl` 的 timestamp 对齐。

### G. 瓶颈诊断与观察开销

```bash
python benchmarks/bench_cuda_training.py --data /path/to/donkey_images --conv im2col --mode phases --epochs 1 --output runs/driving_phases_01
python benchmarks/bench_cuda_training.py --data /path/to/donkey_images --conv im2col --mode throughput --epochs 1 --no-monitor --output runs/driving_unmonitored_01
```

其他参数必须与被对照的原实验一致。`phases` 在每阶段边界同步，阶段时间可相加，但会干扰异步执行；`throughput` 仅在 step 边界同步，phase 字段为 null，不假造分段结果。两者都测有边界同步的训练；需要极限无同步流水吞吐，需另行设计批量计时实验。

`hardware.jsonl` 的显存是 **整卡采样值**，含其他进程；峰值是采样最大值，不是精确的本进程分配峰值。GPU 指标缺失时为 null 并记录错误；无数据不填 0。CPU 为 psutil 系统利用率，另有进程 CPU/RSS。短烟测可能来不及采到硬件样本。温度高本身不证明已经降频。

通过 `--monitor-id` 可指定 `nvidia-smi` 的物理 GPU ID/UUID/PCI bus；默认按 CuPy 的 PCI bus 映射，避免 `CUDA_VISIBLE_DEVICES` 逻辑编号与物理编号混淆。

### H. Nsight：时间线 → kernel 指标

先采时间线，检查 kernel 间空洞、拷贝与 launch 数量：

```bash
nsys profile --trace=cuda,nvtx --sample=none -o runs/hytorch_timeline python benchmarks/bench_cuda_training.py --model donkey --samples 64 --batch 4 --conv im2col --mode throughput --nvtx --output runs/nsys_training_01
```

再查看可用的 section，采集实际热点 kernel：

```bash
ncu --list-sections
ncu --section SpeedOfLight --section MemoryWorkloadAnalysis --section SchedulerStats --kernel-name regex:gemm_shared --launch-count 2 -o runs/gemm_kernel_01 python benchmarks/bench_cuda_gemm.py --sizes 1024 --tiles 16 --output runs/ncu_gemm_01
```

同样再采 tile 32 并改输出目录。比较实际显存字节/带宽、shared memory 行为、occupancy、eligible warps/issue 情况；不要把理论访存减少量冒充实测指标。Nsight 带插桩/重放开销，其运行中的 benchmark 耗时不进入普通性能表。section 名称/可用指标以目标机器 `ncu --list-sections/--query-metrics` 为准。

## 5. 三个预训练模型与迁移训练

使用 torch/torchvision 兼容版本，转换器首次可能下载数百 MB 权重。逐模型执行，验证失败即停止：

```bash
python scripts/convert_imagenet_weights.py --model alexnet --output weights_imagenet/alexnet.npz --verify --verify-device cuda
python scripts/convert_imagenet_weights.py --model vgg11 --output weights_imagenet/vgg11.npz --verify --verify-device cuda
python scripts/convert_imagenet_weights.py --model vgg16 --output weights_imagenet/vgg16.npz --verify --verify-device cuda
```

转换关系：卷积 OIHW 原样；Linear 的 PyTorch `[out,in]` 转为 HyTorch `[in,out]`。每份 NPZ 旁保存来源 URL、版本、参数映射、SHA256 与 logits 验证结果。模型实际计算使用 HyTorch；训练不依赖 PyTorch autograd。新模型库仅接受 RGB/NCHW/224×224，未实现任意尺寸 adaptive pooling。

预处理为短边 256、中心裁剪 224、RGB/255、ImageNet mean/std。它不同于原 `hytorch.models.VGG16.preprocess` 的 BGR 方案，**不能混用旧权重或旧预处理**。

迁移数据目录：`dataset/train/class_a/*.jpg`、`dataset/train/class_b/*.jpg`、对应的 `dataset/val/...`。

```bash
python examples/finetune_imagenet.py --model alexnet --weights weights_imagenet/alexnet.npz --data /path/to/classification_dataset --device cuda --head-only --batch 1 --epochs 3 --output runs/transfer_alexnet_01
```

默认冻结卷积特征、训练分类器；`--head-only` 只更新最后一层；`--train-all` 全量微调。结束时校验冻结参数哈希未变，保存 epoch loss/val accuracy 与最终权重。三个模型都可用同一入口。VGG 分类器体积较大，先 batch=1。

冻结迁移示例是分类任务，不把 ImageNet 分类 head 直接当成 DonkeyCar 转向角回归 head。自动驾驶性能对比仍使用原 DonkeyNet。

## 6. 当前结果与交付文件

- Mac 已验证的范围、具体测试数和未验证项见 `CUDA验证记录.md`。
- `runs/cuda_host_smoke`、`runs/cuda_host_torch_smoke`、`runs/cuda_host_gemm` 只含 CPU/合成数据检查；字段明确标记 `NOT_EXECUTED`。
- `outputs/cuda_host_validation` 是日志/绘图链路的主机侧示例，不是 GPU 性能结论。
- 文件与入口变更见 `CUDA修改清单.json`。

等 GPU 测试完成后，提交 `build.log/build.json`、`environment.json/tests.log`、GEMM/ops CSV、训练 config/steps/hardware/summary/final_weights、Nsight 报告和官方权重转换 JSON。根据这些实测结果再决定默认路由；目前保持原路径，不宣称手写 CUDA 比 cuBLAS 更快。

## 官方接口依据

- [CuPy 性能计时说明](https://docs.cupy.dev/en/stable/user_guide/performance.html)：CUDA 异步执行、CUDA Event、预热。
- [CuPy Stream 接口](https://docs.cupy.dev/en/stable/reference/generated/cupy.cuda.Stream.html)：当前 stream 与指针。
- [torchvision AlexNet 源码](https://docs.pytorch.org/vision/main/_modules/torchvision/models/alexnet.html)、[VGG 源码](https://docs.pytorch.org/vision/main/_modules/torchvision/models/vgg.html)：层顺序和官方权重枚举。
- [Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/)、[Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)：时间线和 kernel 指标采集。
