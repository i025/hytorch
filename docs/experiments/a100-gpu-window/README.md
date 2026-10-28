# HyTorch A100 空闲窗口实验：最终报告

**本轮可执行实验已完成并保留成功/失败证据，整体不能判为全部 PASS。** GEMM 2048/4096 严格数值容差未通过；Nsight Compute 受权限阻止；真实数据训练和迁移训练缺少数据。主机/GPU 回归、修复后合成训练、三个官方预训练模型验证通过。没有自动开始 Kernel 优化。

[全部数据与命令索引](files.md) · [GEMM](../a100-stage2a-gemm/README.md) · [Conv/Pooling](../a100-stage2b-ops/README.md) · [训练对照](training-comparison/README.md) · [问题与修复](analysis/README.md) · [Profiling](profiling/README.md) · [预训练模型](pretrained/README.md)

## 设备、监控窗口与环境

用户授权最长八小时等待空闲 GPU：2026-10-28 16:41:09 UTC 至 2026-10-29 00:41:09 UTC（北京时间 00:41 至 08:41）。在截止前获得空闲设备并执行完本轮可执行项目。每 30 秒检查八张卡，显存/利用率/计算进程数均为 0 后再等 10 秒确认，启动前再次检查；未强占、未杀其他进程。监控原始数据在 environment/idle-watch-*/。

- 物理 GPU 5（GPU-a33323b7-f2ac-2735-70f3-fd1de1ce547a）：第一次空闲窗口被其他作业占用，启动被拒绝；第二次只执行到 CuPy API 错误，未取得有效基线。
- 物理 GPU 0（GPU-f099fdbd-adf2-6157-3462-4145bc2a8359）：16:59 UTC 获得空闲窗口，完成 GEMM、Ops、初始训练、Nsight/Sanitizer 和三模型验证。
- GPU 0 再次忙碌后停止新增工作并恢复监控；17:09 UTC 获得物理 GPU 2（GPU-158426a4-6684-7863-8cbe-ed6595f97683），完成 dtype 诊断、修复后 GPU 回归、完整训练和 Nsight 复测。
- 始终只使用一张卡，按 UUID 设置 CUDA_VISIBLE_DEVICES，内部均为逻辑 GPU 0。修复前后用卡/时段不同，不将二者耗时差归因为单一修改。最终训练各模型内部对比均来自同一卡、同版本、同条件。

Ubuntu 20.04.6，2×AMD EPYC 7763，8×A100-PCIE-40GB；驱动 535.247.01，系统 CUDA Toolkit 12.2.2 / nvcc 12.2.140，GCC 9.4.0。独立 `.venv` Python 3.8.10、NumPy 1.24.4、CuPy 12.3.0、Torch 2.4.1+cu121、torchvision 0.19.1+cu121。CuPy 单独路径实测 Runtime 12020；Torch 构建 CUDA 12.1，Torch 先导入的联合 smoke 中 CuPy 实测 Runtime 12010，未把不同加载路径混称一个 Runtime。pip check 无冲突；Torch/CuPy 均真实运行了 CUDA tensor。完整包列表、安装/导入日志、系统工具与阶段命令均在 environment/。

## 验收状态

| 项目 | 状态 | 实测证据 |
|---|---|---|
| PyTorch 安装/依赖/CUDA 运算 | PASS | environment/*、torch-cuda-check-01 |
| 主机回归 | PASS：25 passed / 0 failed / 0 skipped | pytorch-parity/host_after_dtype_fix_01.log + XML |
| TinyNet、DonkeyNet PyTorch 主机对齐 | PASS：两个测试实际执行，修复后再次验证 | numeric-comparisons-02.jsonl、summary-after-fix.csv |
| GEMM 128/256 预检、1024 正式 | PASS | stage2a pilot-03/formal-01 |
| GEMM 2048/4096 正式及 Torch 补充 | FAIL：超出原定 atol=rtol=1e-4 | 全部原始 CSV 与 analysis/gemm-diagnosis-01 |
| Conv/Pooling | PASS：24 行数值检查，含 18 组 GPU 配置 | stage2b results-01 |
| GPU 回归（修复后） | PASS：39 passed / 0 failed / 0 skipped | analysis/validation-after-dtype-fix-01 |
| 修复后合成 GPU 训练对照 | PASS：39 throughput + 13 phases 运行 | training-comparison/*-fixed-* |
| Nsight Systems | PASS：原始 trace 与 CSV | profiling/；原始二进制留服务器 |
| Nsight Compute Tile 16/32 | BLOCKED：ERR_NVGPUCTRPERM | profiling/ncu-tile*-monitor-01 |
| Compute Sanitizer | PASS：37 GPU 测试，0 memory errors（修复前） | profiling/sanitizer-results-01 |
| AlexNet/VGG11/VGG16 GPU logits | PASS：3/3 | pretrained/ |
| 真实 DonkeyCar 训练 | BLOCKED：未取得数据集路径 | 常用目录检查记录在 environment/tool-data-inventory.json |
| 真实分类迁移训练 | BLOCKED：未取得分类数据集 | 未执行、不伪造任务准确率 |
| Kernel 优化 / Unity 适配 | NOT_RUN：本轮没有启动 | 保留供共同分析后的后续任务 |

## 主要结论

1. **GEMM 最快者**：七配置同场 baseline 的有效 N=1024 结果中 CuPy 最快，0.295526 ms；自研最快 Shared 32 为 0.740966 ms，耗时约 CuPy 的 2.51 倍。其他大规模失败行不能评为有效最快实现。独立 Torch/CuPy 补充的 N=1024 为 0.246579 / 0.326861 ms；不同场次不可拼接为严格匹配榜单。
2. **Tile 与 Shared Memory**：N=1024 Shared 32 比 Shared 16 快约 1.094×，比 Blocked 32 快约 1.869×，支持该配置下使用 Shared Memory 的实际收益。2048/4096 虽有相应原始计时，因未通过正确性不作验收收益；没有 NCU 计数器，不能推断 occupancy 或带宽原因。
3. **Conv / Pooling**：所测 Conv 输入下 Direct 比 Im2col 快；batch 16 Direct vs CuPy 仅约 1.015×，差异很小。Native Pooling 三个 batch 均更快，完整均值、标准差和逐次结果见 Ops 表。算子计时包括分配与布局处理，不能与 GEMM 纯计算时间混用。
4. **PyTorch 数值对齐**：两个主机模型初始参数误差为 0。TinyNet Forward/Loss/梯度/更新最大绝对差 1.4901e-7 / 0 / 9.5367e-7 / 7.4506e-9；DonkeyNet 为 8.1956e-8 / 4.7684e-7 / 4.4703e-7 / 1.4901e-8。所有逐参数相对 L2 与原定容差在原始 JSONL/CSV。
5. **修复后真实 GPU 训练吞吐量**：TinyNet 原 HyTorch/CuPy 993.964、Combined 1006.085、Torch 1260.166 samples/s；DonkeyNet 分别 525.299、529.029、801.052 samples/s。Torch 相对原路径约 1.268× / 1.525×。每模型 64 synthetic 样本、batch 4、epoch 1、三次重复；不能外推真实驾驶任务、长期训练或不同 batch。
6. **主要耗时线索**：修复后 DonkeyNet Combined 的独立 phases 平均 forward 1.999、backward 4.576、update 0.491 ms/step；Torch 对应 0.992、3.206、0.288 ms。Nsight trace 训练 NVTX 包络内 2368 个 kernel，kernel+memcpy 并集覆盖率 14.657%（不是 SM 利用率），反向阶段与大量短 kernel/主机提交值得调查。无法给出受限计数器的 memory bandwidth/occupancy/warp 数据。
7. **预训练验证**：AlexNet/VGG11/VGG16 logits 最大绝对差分别 2.7418e-6 / 2.8610e-6 / 2.6226e-6，相对 L2 为 4.2116e-7 / 3.6358e-7 / 3.9031e-7，全部通过；参数映射逐元素误差均为 0。没有进行真实分类准确率或迁移训练评估。
8. **下一轮优先事项**：共同确定 GEMM 大 K 累加精度与高精度参考验收方案，再评估 Shared GEMM 调优；训练侧优先调查 backward、col2im/布局转换及小 kernel 提交成本。需管理员提供 NCU 计数器权限、用户提供真实数据路径。当前没有提前实施这些优化。

## 修复与限制

- CuPy 12.3 不支持 `ndarray.get(blocking=True)`，删除不支持的参数并保持同步复制语义；修复前日志与修复后预检均保留。
- MSE/Tanh 反向常数引发 GPU 梯度 FP64 提升。只在相关运算中按输入 dtype 创建常量，未改 CUDA kernel、未放宽容差。修复后 dtype 实测、25 项主机与 39 项 GPU 回归及完整训练均通过。初始训练虽数值接近，不能当作严格 FP32 基线；历史结果完整保留。
- GEMM 2048/4096 的全部 GPU 变体各有 10 / 2418 个超差元素。最差点 FP64 dot 与按顺序 FP32 FMA 诊断支持累加舍入/取消误差判断；所测大规模 FMA 采样与 GPU 精确匹配。仍保持 FAIL，未改验收。
- 同模型训练输入/初始权重实际数组完全相同，最终参数最大绝对差 TinyNet 7.45e-9、DonkeyNet 4.15e-8；DonkeyNet 某小参数相对 L2 约 8.27e-4，完整披露而不只给 PASS。
- 每秒整卡硬件采样常比短训练周期还长，不能当成本进程精确峰值或持续利用率。修复后各训练外层监控采样最多观察到一个计算进程，仍不能排除采样间隙干扰。
- 未使用 sudo，未修改共享系统配置；未运行任何哈希验证；未上传模型权重、数据集、密钥或其他用户进程 ID。

## 原始数据、图与提交

服务器根目录 `/home/xymo/projects/hytorch`；公开数据副本在本目录及 stage2a/stage2b，全部文件路径与原始 runs 索引见 [files.md](files.md)。正式原始命令保存在每阶段 launch.json/commands.json/日志 command.json，附实际起止时间。所有计时、Loss、数值误差图由真实 CSV/JSON/SQLite 生成，源码在 scripts/，没有伪造终端截图。

代表图：[GEMM 六图](../a100-stage2a-gemm/figures-formal-01/gemm-latency.png)、[Conv/Pooling](../a100-stage2b-ops/figures/pooling.png)、[DonkeyNet 实际 Loss](training-comparison/donkey-throughput-fixed-details-01/loss.png)、[训练阶段](training-comparison/donkey-phases-fixed-details-01/phases.png)、[Nsight 实际时间线](profiling/nsys-analysis-after-fix/timeline.png)、[主机误差](figures/host-parity-after-fix.png)。

分支 [experiment/a100-gpu-complete](https://github.com/i025/hytorch/tree/experiment/a100-gpu-complete)，各阶段提交见 [commits.md](commits.md)。没有合并 main 或 force push。GitHub CLI 未安装，未创建 PR；可使用 [创建 PR 链接](https://github.com/i025/hytorch/pull/new/experiment/a100-gpu-complete)。
