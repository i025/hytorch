# 正确性

- 完整GPU回归：200 passed，11 deselected，无skipped。此前102项全部保留，新增98项。
- 主机：23项原主机/新backend契约 + 9项dtype回归 = 32 passed。
- Compute Sanitizer memcheck：67 passed，ERROR SUMMARY: 0 errors。
- Compute Sanitizer racecheck：67 passed，0 hazards（0 errors, 0 warnings）。

新增测试含KH/KW=1/3/5、stride1/2/3、padding0/1/2的27组合以及3个非对称组合；连续、训练转置、负stride三布局共90项。另测试零stride广播/步长切片、预分配out、非默认stream、连续3次调用、无贡献输出位置、输出别名/几何拒绝、旧ABI库fallback。

每组直接与原CuPy结果**逐位相等**，与独立CPU scatter参考逐位相等；输入前后逐位不变。CUDA Graph实际kernel节点检查native路径确实运行col2im_strided。CPU scatter是按kh/kw升序、逐oh/ow散射，不复用native合法范围推导。

FP32/FP64、to_matrix true/false另与PyTorch Fold的一阶和二阶梯度对照（rtol2e-6,atol2e-7）。显式FP64时无native调用，输出及一阶/二阶梯度保持FP64。自动求导类不变。

36个benchmark case（12种几何×3布局），每种15轮事件配对，全部与原生产CuPy函数逐位一致、输入不变；预分配CuPy基准的AST kernel也逐项与生产函数验证。7轮训练的全部最终参数梯度、权重和loss逐位一致，最大绝对差0；参数及梯度保持FP32。

既有GEMM的严格门限和此前失败记录未修改；本轮没有重新进行大型GEMM门限实验，也不把col2im测试通过描述成旧GEMM门限问题已解决。旧GEMM源码字节前缀保留，完整旧GPU回归执行。

约束：合法非空6D col、连续非重叠输出、当前设备；非法geometry和别名明确报错。超出支持索引metadata时生产路径fallback。未测试不受binding约束的外部任意C ABI调用。
