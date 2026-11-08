# 六页PPT材料（所有图从本轮CSV生成）

## 1. 问题：单算子改善不等于完整训练收益
- 已有FP32/GEMM/Col2im优化作为当前可靠基线，不累计历史百分比。
- 本轮同时评估dWeight布局与多张量SGD，候选必须同时满足数值与端到端条件。
- 用图：figures/data_path.svg；历史性能仅作为背景，不与当前柱图相加。

## 2. 分析：布局与启动成本
- trace显示原dWeight每step5次im2col、10次copy、9次GEMM/归约。
- 原SGD16参数32kernel/step；直接做批量更新有潜力但仍须数值验收。
- 用图：figures/dweight_kernels.svg、figures/operator_wrappers.svg；证据PROFILE.csv/FULL_WRAPPERS.csv。

## 3. 实现：accepted D / rejected S
- D：fresh native(T,D)→gy packing→CuPy/cuBLAS→OIHW，不缓存forward数值。
- S原型：32描述符一组、当前指针/stream、别名回退、显式两次舍入，24regs/0local。
- S因CuPy默认FTZ与禁止FTZ的原型在subnormal上不一致而禁用；图中需红色注明“原型未通过”。
- 用图：figures/data_path.svg。明确库贡献边界。

## 4. 正确性
- 90旧+78留出+15真实梯度，B/R全部通过；B/A差异完整披露。
- 原有高阶weakref错误已单独修复，所有配置共同使用。
- 最终272passed+2个原CPU主动skip；Sanitizer各28passed/0error。S原型失败记录不能隐藏。
- 用图：figures/numerics.svg；表引用CORRECTNESS/REPORT及原始日志。

## 5. 性能
- B0 0.851160s → D 0.820923s，3.552%减少。
- 配对节省30.237ms，95%CI[27.827,32.647]ms。
- 四个请求配置均展示，但S*=fallback、DS*=D+fallback，不能称真正两项优化组合收益。
- 用图：training.svg、paired.svg、frameworks.svg。主图零起点，SD与CI标明；同窗口PyTorch仍更快。

## 6. 边界
- batch1/16及TinyNet各3轮范围检查，不夸大为普遍结论。
- 内存仅allocator峰值；batch1没有同等内存下降，连续轮数有限。
- 历史严格GEMM失败仍在；真实驾驶数据准确率NOT_RUN；S生产加速NOT_RUN。
- 用图：memory.svg；推荐只启用D（其余backend按报告显式设置），SGD保持legacy。

可直接引用C1–C5，但C5必须同时附“数值未通过，未启用”。所有SVG可编辑、PNG可直接放PPT，绘图脚本diagnostics/plots.py。
