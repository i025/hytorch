# 本阶段修改说明

没有生产或测试源码修改，没有新增reuse开关，没有重编译或替换CUDA库。git_diff.patch为空，baseline/git_diff.patch是进入阶段时已有修改，二者不能混淆。

核对并保存：functions_conv.py Conv2d/Conv2DGradW；runtime.py conv2d/_arrays/_geometry；kernels.cu hy_im2col/im2col_kernel；core.py Function/Variable.backward/Config；functions.py FP32修复。

只新增runs/a100_20261106/dweight_im2col_reuse下诊断脚本和实验交付：
- math_gate.py：原tensordot、真实native列矩阵、新矩阵布局、FP64参考与未放宽门限比较。
- check_reference.py：CPU独立展开与long-double复核失败点。
- validate_baseline.py：完整既有host/GPU基线和独立实验Sanitizer。
- report.py、package.py：报告、清单与压缩包。

未保存或跨step复用任何生产workspace。只在独立实验局部变量持有col，不能把其生命周期当作新生产方案。raw/scope_checks.json含文件与动态库SHA一致性证明。
