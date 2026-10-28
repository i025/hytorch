# 官方预训练 CNN GPU 对齐

**PASS：AlexNet、VGG11、VGG16 全部真实在 A100 上完成参数转换与 logits 对齐。** 物理 GPU 0、逻辑 0；torchvision IMAGENET1K_V1 官方权重，eval 模式，固定 seed=42、输入 [1,3,224,224]，FP32、禁用 Torch matmul/cuDNN TF32，atol=rtol=1e-4。

| 模型 | 最大绝对误差 | 相对 L2 误差 | 参数映射最大误差 | GPU 验证 |
|---|---:|---:|---:|---|
| alexnet | 2.74181366e-06 | 4.21161892e-07 | 0.0 | PASSED |
| vgg11 | 2.86102295e-06 | 3.63579178e-07 | 0.0 | PASSED |
| vgg16 | 2.62260437e-06 | 3.90308233e-07 | 0.0 | PASSED |

各模型 weights.json 包含官方源 URL、版本、全部参数名/形状映射和逐参数最大绝对误差（均为 0），logits.csv 保留全部 1000 个原始输出值及误差。各 -monitor 目录包含完整终端输出、命令、起止时间和显存/时钟/功耗采样，已去除 PID。全部不使用 hash/checksum 校验。

权重保留在服务器 /home/xymo/projects/hytorch/.venv/torch-cache/hub/checkpoints/ 和 runs/pretrained_{alexnet,vgg11,vgg16}_01/weights.npz，不上传 Git。原始 logits NPZ 同目录。没有取得真实分类数据集，Head-only 迁移训练 **BLOCKED**；本验证不是 ImageNet 准确率评估。

scripts/convert_imagenet_weights.py 新增 --no-checksums，下载显式 check_hash=False、跳过本地 checksum，同时保留名称/形状验证并逐数组严格比较转换结果；hytorch/model_zoo.py 仅给当前加载路径增加 verify_checksum=False 入口，其他调用默认行为不变。推理前向不涉及后续发现的训练反向标量类型提升问题。

```bash
TORCH_HOME="$PWD/.venv/torch-cache" python scripts/convert_imagenet_weights.py --model alexnet --output runs/pretrained_alexnet_01/weights.npz --verify --verify-device cuda --no-checksums
```

其余模型及完整 UUID 包装命令见各 -monitor/launch.json。下载原始日志位于 ../environment/pretrained-download-01.log。
