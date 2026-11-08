# 正确性状态

- 原主机回归：33 passed（24+9）。
- 原GPU回归：204 passed、12 deselected CPU项目、0 skipped。
- 列矩阵layout：90/90逐位相等，包含负stride和非单位slice。
- dWeight B/A：54/90逐位相等，36组非逐位一致。
- 原Donkey数值门限atol2e-5/rtol2e-4用于新旧一致性：12/90组失败（12个元素、4个独立shape/seed，三种layout重复）。
- 同一门限对FP64：旧A9/90失败，新B0/90失败。不能称为新算法普遍精度恶化。
- Conv host门限atol1e-5/rtol1e-4用于B/A：21/90失败。更严格小型dtype门限仅辅助记录。
- 独立CPU long-double参考7个失败记录与GPU FP64参考最大差1.04e-13。
- Compute Sanitizer memcheck：90组完成，0 errors。
- Compute Sanitizer racecheck：90组完成，0 hazards、0 errors、0 warnings。

Sanitizer与重复诊断的数值结果见raw/memcheck.csv、raw/racecheck.csv；它们的成功退出不代表数值准入通过。未增加生产reuse，不存在“新路径完整训练通过”结论。未执行新reuse高阶梯度、原地修改、生命周期、跨step stale数据和训练对齐测试。

原测试源码与门限未修改；大形状N(0,1)数据的新旧比较是独立实验，原单元测试仍通过。门限原本的参考对象为其指定PyTorch测试，本轮将数值阈值保守用于替换一致性，而未擅自改为只看FP64参考。
