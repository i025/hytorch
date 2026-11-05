# HyTorch 六小时自主优化研究

**本轮部分完成，研究与GPU任务已停止。** 北京时间2026-11-05 01:36:35开始，
六小时截止07:36:35（UTC分别为11-04 17:36:35/23:36:35）。下一次恢复处理时已是
北京时间10:37:53，超过截止，故未再启动任何实验。**没有在六小时墙钟内完成交付。**
实际监测记录止于北京时间02:44:50；此后至恢复之间无记录，不声称连续监测六小时。
基线`b7446c6046c084b4a28d826b605df4a8761a7152`，分支
`experiment/hytorch-autonomous-optimization`。本轮不修改4H历史结论或数值容差。

## 当前完成状态

| 项目 | 状态与证据 |
|---|---|
| 源码/历史性能分析 | 完成，见[source-analysis](source-analysis.md)、[bottleneck-analysis](bottleneck-analysis.md) |
| CPU回归 | 103 passed，0 failed，0 skipped（最终完整主机集合，保留此前各次输出） |
| CPU共同状态及FP64参考 | 510+1020组全部通过 |
| CPU1600步漂移/共同状态重放 | 长轨迹出现严格失败，612组同状态一步全部通过；保留负结果 |
| 预取CPU原型 | 12次真实图像对照逐元素一致；吞吐数据见下文 |
| Native ReLU反向原型 | 已实现、SM80编译和CPU动态库加载通过，默认关闭；GPU验收NOT_RUN |
| GPU重放/738项测试/算子/训练/Nsight | 两次资源检查中断，0项任务完成；全部NOT_RUN，不宣称通过或加速 |
| GitHub | 已本地分批提交；推送因缺少凭据阻塞，远程分支未发布 |

## 数值结论：长期超差不是CUDA Bug的充分证据

新增完全不使用CUDA的1600步真实数据实验中，CPU HyTorch FP32与CPU PyTorch FP32
从第391步起有361个Loss超过原严格容差，最大差1.56238675e-5；但18个共同状态
检查点的612组一步Forward、Loss、梯度、更新比较全部通过。初始及四个4H最终状态
的510组CPU对照、1020组独立FP64全网络对照也全部通过。

第一次更新后已经有319个参数值不同，最大7.45058060e-9，当时Loss相同。
因此“首次严格Loss失败”远晚于“微小差异首次出现”。这些实测支持不同FP32计算
舍入经连续非线性SGD迭代逐渐放大的解释，不再追求长期逐位一致。CPU SGD在完全
相同参数/梯度下也有2/251749个值与分离乘减不同，差9.31e-10，全部在容差内。

这不是对任意CUDA状态的无Bug证明；GPU共同状态重放未执行。保留4H119组失败，
不放宽容差，不做无依据精度修补。详见[numerical-drift.md](numerical-drift.md)。

## 两个显式可选原型

1. **P：既定顺序CPU预取。** 复用现有AsyncDataLoader，仅增加`indices`，保持
   原图像、标签、重复样本、epoch顺序与末批。GPU主线程负责传输和训练。
2. **R：通用Native ReLU Backward。** 一个CUDA C/C++ Kernel替代掩码+乘法，
   `backend(relu_backward='fused')`显式启用。FP32、任意非空形状、当前Stream；
   非连续布局的准备计入完整算子；高阶求导保留原路径。默认不变。

**J=I+P+R**。P/R分别消融，不开发第三个优化。没有修改Conv、GEMM、dWeight、
dInput或SGD Kernel。设计、原理和未覆盖范围见[design-decisions.md](design-decisions.md)。

显式接口（GPU正确性尚未验收，不改变默认）：

```python
from hytorch.backends.cuda import backend
from hytorch.dataloaders import DataLoader

loader = DataLoader(dataset, 4, shuffle=False, indices=existing_order, num_workers=2)
with backend(relu_backward='fused'):
    loss.backward()
```

该示例仅展示新增参数；正式J使用I的全部既定后端参数，见`common.py::Engine.context`。

CPU1024张真实图像、三次轮换重复、包含完整加载器启动/结束：

| 线程数 | 图像/s均值±样本SD |
|---:|---:|
| 0 | 1096.82±147.93 |
| 1 | 1092.18±55.00 |
| 2 | 1681.71±45.54 |
| 4 | 1774.12±145.80 |

保留第一次较慢结果。选择2线程以控制并发和波动。这是CPU预处理结果，**不是训练
吞吐或CUDA加速比**。J/I、J/A、J/E及完整训练加速本轮均为NOT_RUN。
另外准备E-prefetch对照，防止把通用数据预取误称为自研CUDA独有收益。

## 实验条件与GPU生命周期

Ubuntu20.04.6，双EPYC7763，Python3.8.10，NumPy1.24.4，CuPy12.3.0，
PyTorch2.4.1+cu121，CUDA Toolkit12.2.140，GCC9.4，驱动535.247.01。
本项目既有独立`.venv`，没有安装新依赖、使用sudo或改变共享环境。

拟测A/F/I/P/R/J/E，真实Udacity相同初始权重/行顺序、DonkeyNet及SpatialNet、
Batch4、FP32、SGD .001、seed42、无Dropout/TF32；10步预热后重置，256步×3次独立
进程，轮换顺序。驻留输入、完整I/O、独立同步阶段、Nsight分别计时，不交叉拼接。

GPU0 UUID：`GPU-f099fdbd-adf2-6157-3462-4145bc2a8359`，映射逻辑0。
首次观察到空闲后于UTC17:54:57启动，17:55:02检测到其他用户新PID3427844而停止，
仅结束我们的PID3428466。没有完整数值或性能结果，未Reset、未切卡，Context随进程
退出释放。第二次监测固定同UUID，于UTC18:44:50达到持续空闲门槛，但进入控制器
时前置空闲断言失败，没有启动GPU子进程。该次前检快照未写入，无法确认具体是
进程列表还是显存字段触发；不能补造进程状态。监测随后结束，并未持续到六小时截止。
最终只读检查GPU0仅见其他用户PID3729603，没有本轮GPU/监测Python进程。
没有切卡、Reset、显存占位或修改系统配置。GPU使用前后数据见
[首次中断](results/resource-interruption-01/file-index.csv)、[最终完整索引](results/final-raw-01/file-index.csv)。

## 长期性能表与未完成验收

本轮统一条件的GPU性能实验均未执行，保留空值，不引用旧窗口成绩补齐：

| 版本 | 本轮吞吐 | 相对前阶段 | 相对A | 对PyTorch比值 |
|---|---|---|---|---|
| A 无Native | NOT_RUN | — | NOT_RUN | NOT_RUN |
| F 原Gather优化 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| I Split-K dWeight | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| P I+预取 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| R I+Native ReLU | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| J I+P+R | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| E PyTorch / E-prefetch | NOT_RUN | — | — | NOT_RUN |

没有本轮Nsight、GPU微基准或匹配吞吐结果，不能提供其性能曲线或稳定性结论。
SM80编译资源报告为新ReLU12寄存器、0shared、0spill，不等于实测Occupancy。
实验原型保留默认关闭，不能推荐为已验收的加速版本。未完成的epoch-only计时草稿
已恢复到提交版本，补丁仅留服务器`unfinished-epoch-timing.patch`，未用于任何实测。

## 模型质量独立观察

仅分析4H原始验证预测，不改数据划分或训练策略。I第三epoch预测均值+.0148615，
验证标签均值−.0364025；预测标准差.0145984，标签标准差.1238619，Pearson .3081。
验证MSE .01706862高于零预测.01666691，差.00040171。均值偏差平方.00262800，
协方差贡献−.00111426，均可从原始预测重算。存在明显分布/输出范围差异，但没有
由此证明某个训练Bug或自动驾驶能力。见[quality-analysis](results/cpu-raw-01/quality-analysis)。

## 原始数据、复现和图

- [命令](commands.md)、[时间线](timeline.md)、[完整数据索引](data-index.md)。
- [CPU原始记录及服务器NPZ索引](results/cpu-raw-01/file-index.csv)。
- [CPU初始化复查与加载Profiling索引](results/cpu-additional-01/file-index.csv)：初始化整理后1530个已保存数组精确复现；加载cProfile单独记录。
- 服务器根目录：`/home/xymo/projects/hytorch/runs/autonomous_optimization_01/`。
- 原图、完整NPZ参数/梯度/输入、动态库、Nsight二进制均不上传GitHub。
- [最终状态JSON](results/final-status.json)及[最终198个服务器文件/81份可移植记录索引](results/final-raw-01/file-index.csv)。
- 下列均为真实数据可视化，不是终端截图：
  [CPU长期漂移/同状态一步](figures/cpu-01/01-cpu-long-drift-and-local-replay.png)、
  [4H历史GPU长期漂移](figures/cpu-01/02-historical-gpu-long-drift.png)、
  [CPU参数漂移](figures/cpu-01/03-parameter-drift.png)、
  [CPU预取吞吐](figures/cpu-01/04-cpu-prefetch.png)、
  [模型质量独立观察](figures/cpu-01/05-quality-observations.png)。

## 后续优先级与交付限制

1. 先获得可靠单卡窗口，完成同状态CUDA重放和P/R/J消融；优先判断数据预取的真实
   端到端收益，并与E-prefetch公平对照。CPU提升不能替代GPU验收。
2. 再依据完整流水线数据，研究读取/预处理与H2D重叠及缓冲成本；不更改数据划分、
   预处理数学或训练策略来提高数字。
3. 根据新Trace再决定是否优化剩余点算子/自动求导提交（历史ReLU、MSE、Tanh均有
   多次短Kernel），不凭历史Kernel核时直接重写第三个Kernel。

本地阶段提交`1dcfa82`、`a8aa5a6`、`80ead58`保存了研究过程；最终归档提交另见
本分支HEAD。GitHub预检多次返回`could not read Username for https://github.com`，
没有GitHub CLI，也没有可用推送凭据。提交/推送授权已经收到，阻碍是凭据而非权限。
恢复凭据脚本在服务器`runs/autonomous_publication_01/cache_credentials.sh`，Token
不应发到聊天中。恢复后仅需推送已有提交，不应自动重启GPU研究或进入下一阶段。
