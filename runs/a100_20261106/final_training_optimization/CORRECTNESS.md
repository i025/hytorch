# 正确性验收汇总

- 最终完整回归272 passed、2 skipped；CPU35、GPU237，skip是原CPU DonkeyNet两项显式由GPU覆盖。raw/full_tests_guard.log列明原因。
- 最终新增专项35 passed（raw/guard_tests.log）。
- 新原生调用/SGD原型memcheck、racecheck各28 passed、6 deselected、0错误/0hazards。原型special rounding不因sanitizer通过而获得数值准入。
- 初次SGD原型1 failed、31 passed：subnormal逐位差异，保留raw/special_initial.log。生产sgd_multi现返回False，禁用候选并继续D。
- D168组固定/留出+15组真实训练B/FP64参考均通过；A/R、B/A和其他严格门限全部在NUMERICS.csv；失败坐标未删除。
- 单步、完整192步、batch1/16、TinyNet、CuPy和Torch参考的逐参数及loss轨迹对齐全部通过。参考不同框架不是逐位相等要求。
- 新旧D相对误差按冻结混合门限；位相等与数值相等分列，跨精度位相等不适用。
- 原Conv2DGradW二阶导数错误单独CPU复现并修复；默认/packed两组在CPU/GPU与PyTorch独立公式比较通过。
- 历史大GEMM严格失败未解决；原有测试源码、门限SHA逐项保持不变。
- int32几何保护后补跑35专项+完整回归，性能完整重测。原正常数值集合算术与kernel未变。
