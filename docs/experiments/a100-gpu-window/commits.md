# 本轮阶段提交

分支 experiment/a100-gpu-complete；父提交 f5cf9c8ffc7c039c2e99d473ba17b6e6e4696012。最终索引/报告提交见该分支最新提交，本文件不自引用自己的提交 ID。

- [dcd43c2e529991f779dec83658907ecc7f720f70](https://github.com/i025/hytorch/commit/dcd43c2e529991f779dec83658907ecc7f720f70) — experiment: install PyTorch and execute TinyNet and DonkeyNet parity checks
- [05c752918987358b1ecece576ae815874b6a99db](https://github.com/i025/hytorch/commit/05c752918987358b1ecece576ae815874b6a99db) — experiment: retain A100 GEMM baseline, numerical failures and CuPy API fix
- [c5a49d7cbe0666d311f5bd2108c03d40ea0a45ef](https://github.com/i025/hytorch/commit/c5a49d7cbe0666d311f5bd2108c03d40ea0a45ef) — experiment: record passing A100 Conv2d and pooling baselines
- [4f471e47519568c2e620cca615cbee91b71f47b6](https://github.com/i025/hytorch/commit/4f471e47519568c2e620cca615cbee91b71f47b6) — experiment: complete 39 matched synthetic GPU training runs with PyTorch
- [9fbfb887edde451a63be1fb0ddd55b167f681b6d](https://github.com/i025/hytorch/commit/9fbfb887edde451a63be1fb0ddd55b167f681b6d) — experiment: preserve Nsight timeline, dtype finding and counter permission failures
- [0874938023b8d447c2e3e817386558b63623ab07](https://github.com/i025/hytorch/commit/0874938023b8d447c2e3e817386558b63623ab07) — experiment: verify official AlexNet VGG11 VGG16 logits on A100 without checksums
- [9f1d4ff4566e60843ae369feed275e4a72bfd746](https://github.com/i025/hytorch/commit/9f1d4ff4566e60843ae369feed275e4a72bfd746) — experiment: add Torch GEMM and reproduce FP32 accumulation tolerance failures
- [bca7af42ceb4ce9659cb5edd07abe4292221b89c](https://github.com/i025/hytorch/commit/bca7af42ceb4ce9659cb5edd07abe4292221b89c) — fix: preserve MSE and tanh gradient dtype; repeat matched A100 FP32 training
