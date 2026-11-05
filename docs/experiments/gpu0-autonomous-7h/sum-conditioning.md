# CPU FP32 Sum 条件数诊断

本节使用已有 Sum 算术模型，没有修改 Kernel 或验收容差。GPU NOT_RUN。
长度为 1024、4097、65537、1048577，固定种子 0–19，输入由 NumPy default_rng 生成 FP32 正态分布。
每个输入再构造一个强消去版本：减去 FP64 mean 转成的 FP32 标量。
没有选择性筛选种子，也没有调整输入以获得更好通过率。

独立参考为固定 FP32 输入的 FP64 sum，保留 FP64 参考精度；
比较使用已有 `atol=1e-5, rtol=1e-4`，不将相对误差大自动当作 Bug，也不豁免绝对容差超差。
条件数为 `sum(abs(x)) / abs(sum(x))`，用于分析输入的消去程度。

## 实测结果

| 长度 | 普通输入：CPU模型/NumPy 失败数 | 强消去：CPU模型失败数 | 强消去：NumPy失败数 | 强消去条件数中位数 |
|---|---|---:|---:|---:|
| 1024 | 0/20、0/20 | 0/20 | 0/20 | 1.1940e8 |
| 4097 | 0/20、0/20 | 0/20 | 2/20 | 1.1171e8 |
| 65537 | 0/20、0/20 | 13/20 | 16/20 | 8.3961e7 |
| 1048577 | 0/20、0/20 | 18/20 | 19/20 | 8.4536e7 |

共 320 项比较：普通输入 160 项通过；强消去输入 160 项中 68 项失败，全部保留。
长度 1048577 时，强消去输入 CPU 模型最大绝对误差为 0.000267759722，NumPy 为 0.000374176190。
同长度普通输入的 CPU 模型最大绝对误差为 0.000457858759，反而更大；但其参考输出量级较大，
在原相对容差下仍通过。不能只看最大绝对误差，或只看“是否更接近 NumPy”来判断精度。

对 1048577 长度强消去输入，CPU 模型绝对误差中位数 8.74527e-5，
NumPy 为 6.88400e-5；尽管模型的超差项更少、最大值更小，误差中位数却更大。
因此没有证据宣称此归约顺序在所有输入上都优于 NumPy，也没有通过拟合 NumPy 顺序来改 Kernel。

## 结论的边界

这组 CPU 结果支持“强消去会使多种 FP32 归约顺序难以满足固定容差”的归因。
它不能排除 GPU 地址计算、并发、Stream 或同步 Bug；仍须执行已准备的 GPU–CPU 模型逐元素比较。
CPU 模型一致也不将原失败改成通过；GPU 正确性、性能与训练收益仍分别为 NOT_RUN。
已有两个独立强消去负输入及其 keepdims 重复测试继续保留，不以本节数据取代原日志。

## 原始记录

- [逐输入、逐方法误差](correctness/sum-conditioning-01/sum-conditioning-01/samples.csv)
- [分组汇总](correctness/sum-conditioning-01/sum-conditioning-01/summary.csv)
- [生成规则及种子](correctness/sum-conditioning-01/sum-conditioning-01/config.json)
- [完整服务器文件索引](correctness/sum-conditioning-01/file-index.csv)
- [运行日志](logs/sum-conditioning-01.log)
- [实际数据图](figures/cpu-conditioning-01/cpu-sum-conditioning.png)

服务器原始目录 `runs/gpu0_autonomous_7h_01/sum-conditioning-01/`，约 86MiB，
失败输入 NPZ 全部只留服务器，GitHub 保存全部 CSV/JSON 和文件索引。
图为 CPU 数据可视化，不是 GPU 数据或终端截图。
