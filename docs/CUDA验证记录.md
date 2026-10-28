# CUDA 开发验证记录

日期：2026-10-28。开发环境：Mac，Python 3.14.3、NumPy 2.4.4、PyTorch 2.11.0。**本记录不包含任何 NVIDIA GPU 性能结果。**

## 已执行

| 验证 | 结果 | 证据 |
|---|---|---|
| 现有 `tests/` 全量主机回归（含新测试） | 75 passed，38 skipped，3 subtests passed；4 条原有 ONNX 弃用警告 | `runs/cuda_host_regression.xml` |
| 补充 DonkeyNet 对齐后的主机专项 | 15 passed | `runs/cuda_host_contracts.xml` |
| TinyNet 的 HyTorch/PyTorch 前向、梯度、SGD 更新 | 通过 | `tests/test_cuda_host.py` |
| DonkeyNet 的 HyTorch/PyTorch 前向、梯度、SGD 更新 | 通过；这是补充专项中的新增一项 | `tests/test_cuda_host.py` |
| SGD/Adam/AdamW 冻结 hook | 冻结层不变，分类头实际更新 | `tests/test_cuda_host.py` |
| FP32 初始化 | Linear/Conv2d 保持 float32 | `tests/test_cuda_host.py` |
| 后端 context、CPU fallback、缺库报错 | 通过 | `tests/test_cuda_host.py` |
| profiler 分母、throughput 空阶段、监控解析与缺失值 | 通过 | `tests/test_cuda_host.py` |
| 三个预训练模型参数映射结构、224 输入限制、预处理 | 通过；不等于真实权重验证 | `tests/test_cuda_host.py` |
| CPU 合成训练烟测 | HyTorch/PyTorch 各完成 2 个 step | `runs/cuda_host_smoke`、`runs/cuda_host_torch_smoke` |
| 两框架烟测最终权重 | 最大绝对差 `1.4901161193847656e-08`，报告阈值内一致 | `outputs/cuda_host_validation/training_comparison.csv` |
| 异步 DataLoader + 截断 epoch + throughput | workers=2 的 CPU 烟测通过 | `runs/cuda_host_async_smoke` |
| NumPy GEMM 记录与图表链路 | 17/32 规模，3 次重复，误差为 0；GPU 明确未执行 | `runs/cuda_host_gemm` |
| 构建脚本 | dry-run 命令生成通过；没有运行 nvcc | `scripts/build_cuda.py --dry-run --arch 89` |
| 新 Python 文件 | 字节码编译检查通过 | `compileall` |
| 图表布局 | 已打开检查，无文字截断 | `outputs/cuda_host_validation` |

全量回归中的 38 个 skipped：37 个真实 CUDA 测试因为 CuPy/GPU 不可用；1 个原有 Windows spawn 专用回归因为当前是 Mac。之后新增的 DonkeyNet 对齐测试通过专项运行验证，未把两次测试数简单相加冒充一次全量运行。

回归最初遇到两类环境问题：沙箱阻止绑定本机 gRPC 端口；解除该限制后，环境代理又影响 localhost gRPC 连接。最终在允许本地端口的环境中，仅为测试进程设置 `NO_PROXY/no_proxy/no_grpc_proxy=127.0.0.1,localhost` 后通过。没有改动分布式源码。

CPU 小规模耗时只用于检查采集、存储、配对和绘图，不用于评价 HyTorch/PyTorch 性能，不外推 GPU 加速比。短训练可能没有硬件采样点；结果保留 null/0 个样本，没有编造 GPU 指标。

## 尚未执行，交付时仍为 NOT_EXECUTED

1. nvcc 编译与 Linux/Windows 动态库加载。
2. 37 项真实 CUDA kernel/stream/梯度测试与 Compute Sanitizer。
3. 1024/2048/4096 五对象 GPU 性能比较、tile 16/32、H2D/D2H、临界点。
4. 真实 DonkeyCar 数据的 GPU 训练、验证准确性/回归损失与资源占用对照。
5. Nsight Systems 时间线与 Nsight Compute 带宽/访存/调度指标。
6. 官方 AlexNet/VGG11/VGG16 权重下载、转换后 logits 验证和真实分类数据迁移训练。

GPU 上请从 `CUDA开发与GPU验收指南.md` 的 A→B→C→D 关卡开始，C/D 通过后再跑完整实验。代码准备完成不代表上述项目已经验收通过。
