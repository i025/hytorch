# PPT 五页建议

本文件中的所有表格和图片来自本轮最终评测；未使用历史时窗的性能数值。

1. **平台与三种实现**：A100-PCIE-40GB、CPU OpenBLAS 4线程、FP32/TF32关闭、F1/O1显式开启。用 SOURCE_STATUS.md 与 WORKLOAD.csv。说明 NumPy是CPU，CuPy/cuBLAS和Native是同GPU；dWeight与Pool backward是混合实现。每case18轮、GEMM array30轮，短操作5次循环摊销；主口径为驻留数据同步完成时间。
2. **正确性**：319 passed、2 CPU跳过；804组独立FP64诊断0失败，4096³完整参考；Native GEMM MaxAbs 1.34e-4 / RelativeL2 2.09e-6。放 figures/06_accuracy.png 与 CORRECTNESS.md，保留“近似一致、并非逐位一致”和历史严格失败未撤销。
3. **GEMM与规模**：放 figures/03_gemm_size_curve.png、05_optimization_ab.png 左图。240×64×432 / 72×64×576 最终相对旧register约3.10×/3.30×；1024³–4096³相对旧Shared约1.81–2.47×，但仍慢于cuBLAS。用 DISPATCH_DECISIONS.csv 标注实际CTA。
4. **Im2col / Col2im / Conv**：放 figures/01_threeway_latency.png 左图、04_im2col_col2im_scale.png。Im2col 6D→matrix copy 必须计入；Col2im holes kernel更慢但完整接口更快；Conv2回退必须保留。不要把 kernel 时间与正常 wall time 混在同一加速比。
5. **优化价值与适用范围**：放 figures/02_native_vs_cupy.png、07_advantage_regions.png，搭配 OPERATOR_AGGREGATE.csv、KERNEL_DETAILS.csv。dWeight五层少一个Copy与分配，conv2请求字节13.10MB→6.78MB。强调使用cuBLAS的是dWeight乘法；本轮不跑完整训练，不能宣称新的端到端训练加速。

所有图均有SVG可无损编辑；全case的退化、负stride、tiny、holes见CSV，不建议仅截取最快shape。图1/2聚合是等case权重均值，不是DonkeyNet训练权重；图3/5给shape级例子更适合讲算法机制。
