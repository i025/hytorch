# 对 experiment/gpu0-autonomous-7h 的选择性移植与 A100 验证

## 结论

有两项适合用于当前优化分支：**原生 ReLU backward 融合**和**显式不计算训练数据输入梯度**。两者组合相对修改前精确提交，在 10 对独立进程实验中均更快：192-step DonkeyNet 训练均值 **0.737014s → 0.705258s，耗时下降 4.31%，吞吐提高 4.50%**。loss、参数梯度和最终权重哈希逐位一致。

Heap 调度适合宽图，保留可选接口；DonkeyNet 单独开启 heap 的差异不显著，不默认启用。没有整体 merge 实验分支，没有替换当前 GEMM、Col2im、packed dWeight、F1/O1、SGD 或 TrainingMonitor。

## 源码与证据审查

- 目标分支：a100-optimizations-and-monitoring-20261108，基线 `4317bf55b22e362c9106fe40ec37ec3b62f2327d`。
- 来源分支：experiment/gpu0-autonomous-7h，审查提交 `d58c581aa3602e450cd736fd14cce5ceda039de3`。
- 共同祖先：`edb6fbe3bfaa9eed461addd29e6943b22e28216e`。
- 来源分支自己的最终报告明确 GPU 窗口为 **NOT_RUN**，没有完成 GPU ReLU/Sum 或训练实验。219 项 CPU 回归与宽图调度收益不能当作 A100 验收。
- 两分支独立发展，整体合并会替换/冲突现有优化与监控，因此只移植可隔离验证的机制。来源报告快照在 baseline/source_*.md，来源核心文件由 detached worktree 读取。

| 候选 | 处理与理由 |
|---|---|
| ReLU backward | 移植现有 FP32 mask-multiply kernel，验证后保留显式开关；减少布尔 mask 分配及一次 launch |
| 输入梯度需求 | 引入 requires_grad=False 的显式停止语义；Conv2d/Conv2DGradW 跳过不需要的 dInput，不剪裁参数梯度、不按层名硬编码 |
| Heap 调度 | 以 (-generation,-serial) 保持旧排序和 LIFO 平局顺序；通过高阶/非默认 stream/宽图测试，默认仍 list |
| Sum 归约 | 不移植：来源报告存在强消去误差负样本，GPU 验收缺失；本轮没有声称其 GPU 性能或通用精度已验证 |
| fused dInput/dWeight/SGD、conv auto dispatch | 不移植：当前分支已有独立验证过的路径，不能用来源未完成的新窗口替代；本轮未逐个 benchmark 这些候选 |
| CPU 预取/DataLoader | 本轮没有移植；当前验证是合成数据，不可用来判断真实 JPEG I/O 收益。需要独立真实数据协议 |

## 实现与接口

ReLU 使用现有来源的 `__fmul_rn(gy, x>0 ? 1 : 0)`，保留 NaN/Inf 和 signed-zero 的原 mask multiply 行为，不启用 fast-math/TF32。加入 C ABI 空指针/size 保护，沿用当前 runtime 的 FP32、device、size、连续化和 current stream 检查。非连续输入显式物化，费用纳入完整调用测量。FP64、CPU、空输入和 mixed-dtype 梯度保留原路径；create_graph 时保留可微分的原操作。

`Variable(data, requires_grad=False)` 表示不接收梯度、不继续遍历该输入；操作输出仍默认 True，**不是 PyTorch 风格的自动全图 requires-grad 推断**。默认 True 保留现有行为；不要在构图后随意修改该标志。只有调用者明确不需要训练数据梯度时才启用。中间激活仍反传，参数梯度和高阶权重导数通过验证。顺便支持返回 None 的梯度与 leaf.backward()；没有修改 loss、优化器、权重梯度计算方法。

CLI 新增 --relu-backward original/fused、--no-input-grad、--backward-schedule list/heap，均保持原行为为默认。已有 --final-config 的含义不变。推荐训练显式启用前两项，heap 仅按计算图实验选择。详见 REPRODUCE.md。

## 实验设计

实际执行物理 GPU 1（A100-PCIE-40GB，UUID GPU-0bc09ee6-967a-b53d-bf80-38379585cf08），CUDA_VISIBLE_DEVICES=1 / 逻辑0。开始时查询两张卡均无进程；从未使用 GPU0，结束快照中 GPU0 的 VLLM 为其他任务，未干预。GPU1 最终0MiB，实验进程已退出。

FP32、256个合成样本、batch4、3epochs/192steps、seed42、固定初始化、legacy SGD .001、dropout0；shared GEMM 128-CTA dispatch、native im2col/pool/col2im、packed dWeight、F1+O1 不变。相同数据顺序、10step warmup，恢复初始化后训练，末尾同步。正式 wall 不包含 Python进程启动/初始化/结果传回CPU，不开 profiler、监控或逐phase同步。模型输出/参数/所有192个loss的读取在计时外。

12轮同进程六模式平衡交替；TinyNet另外6轮三模式检查通用性。再以修改前完整源码 worktree+原动态库对照新组合，做10对独立进程、交替顺序验证净收益，排除新增开关基础设施对 baseline 的影响。后者不是同进程实验，所有逐对值完整保留。没有删除异常值；95% CI 使用配对 t（df11或9），共享服务器、未锁频和运行时段仍是限制。

## 同进程消融（每模式12轮）

| 模式 | 192-step wall均值/s | 相对关闭候选的耗时下降 | 配对 Δwall 95%CI/ms |
|---|---:|---:|---:|
| 候选均关闭 | 0.742186 | — | — |
| 仅ReLU融合 | 0.719195 | 3.10% | [-25.50,-20.48] |
| 仅不计算输入梯度 | 0.722578 | 2.64% | [-21.69,-17.53] |
| 仅heap | 0.740471 | 0.23% | [-3.63,0.20] |
| ReLU + 输入梯度跳过 | 0.703332 | 5.24% | [-40.97,-36.74] |
| 三项全部 | 0.701473 | 5.49% | [-43.04,-38.39] |

组合改善不能简单相加。三项全部对 baseline 的显著收益不能用于证明 heap 的独立收益，因此推荐组合不包含 heap。TinyNet baseline0.284582s，ReLU+输入梯度跳过0.257519s，耗时下降9.51%，配对差CI[-28.21,-25.92]ms；加入heap为0.258334s，略慢于前两项组合。

## 相对原始精确提交的净收益

| 对照 | 10次均值/s | samples/s |
|---|---:|---:|
| 原始4317bf5源码/动态库 | 0.737014 | 1042.04 |
| 新ReLU+输入梯度跳过 | 0.705258 | 1088.96 |

10/10对更快，平均差 **-31.756ms/run**，95%CI **[-42.389,-21.123]ms**。耗时下降4.31%、速度比1.0450×。该结果比同进程消融更保守，作为本轮最终净收益结论。两组每对的参数、梯度及全部loss哈希均相同。训练最终loss为0.00025566318072378635；仅为合成回归，不代表自动驾驶精度。

## 单算子与 CPU heap

ReLU 完整调用覆盖标量、小数组、DonkeyNet七个激活shape、大数组、转置/负stride。每组固定输入、预热20次、20轮交替；每轮30次调用，记录CUDA Event区间、CPU提交时间、末尾同步wall。两组都包括输出分配；Native额外布局物化不隐藏。Event区间包含主机提交间隙，不能称纯kernel时间。七个真实形状的Event中位数改善约1.49–1.53×；原始明细及不利case全部见 RELU_BENCHMARK.csv、RELU_SUMMARY.csv。

CPU heap独立运行在GPU试验结束后：平衡二叉聚合图、每宽度20对，只测backward（构图不计时）。宽度1约1.00×、8约1.01×、64约1.15×、256约1.63×；256宽度的中位数35.59ms→21.77ms，梯度逐位相等。不是CNN训练加速，也不表示所有大节点数图都受益。

## Nsight：收益来源

独立20-step trace，正式性能不取profiler wall；主机同线程NVTX包含关系+CUDA correlationId归因。

| 阶段 | 原kernel数/step | 新kernel数/step | 原纯kernel μs/step | 新纯kernel μs/step |
|---|---:|---:|---:|---:|
| Forward | 34 | 34 | 545.18 | 544.23 |
| Loss | 4 | 4 | 13.11 | 12.98 |
| Backward | 72 | 62 | 658.93 | 581.09 |
| Optimizer | 32 | 32 | 114.87 | 114.74 |
| 合计 | 142 | 132 | 1332.09 | 1253.04 |

原每step七个mask kernel消失，新增七个relu_backward_kernel，且少了七个独立乘法launch；col2im从5次到4次，另少第一层dInput的矩阵/布局操作，合计减少10个kernel。Forward、dWeight的算法与SGD未变。Backward NVTX主机区间2584.85→2416.11μs，launch API312.61→291.69μs；这些是独立profiler诊断，不能与纯kernel时间相加解释正式wall。没有采集NCU，不声称提高occupancy。

pool_reserved_end_bytes只表示每次训练结束时CuPy池保留量，不是峰值。本轮未测全GPU显存峰值，不据此宣称显存峰值降低。

## 正确性与负结果

完整回归402 passed、2个原有CPU skip；新增候选GPU专门检查49 passed（随后新增6个GPU heap case纳入完整回归）；CPU定向重跑32 passed。共2538条loss/梯度/权重对比全部逐位相等。独立原版对照10对的三类哈希全部一致。Compute Sanitizer memcheck26 passed/0 errors；racecheck26 passed/0 errors/0 warnings。实际优化CLI完整训练通过。

原始失败保留：移植测试中SGD枚举名称需从来源接口适配为当前native；新测试曾错误要求既有pow高阶路径FP32，将其改为与原路径dtype/数值直接比较，没有修改生产pow或放宽数值门限。系统/usr/bin旧Sanitizer缺少注入库，改用已安装/usr/local/cuda/bin工具后通过，未安装软件或改权限。详见CORRECTNESS.md。

本轮只证明两个组合在上述合成DonkeyNet/TinyNet上的收益；没有跑真实图片完整训练，没有声称来源整个分支都更快。保留原有路径和显式开关，不合并尚缺GPU证据的其他原型。
