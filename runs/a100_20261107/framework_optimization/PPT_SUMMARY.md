# PPT 六页汇报提纲

## 1. 算子速度不等于完整调用速度

图：figures/01_operators.png（同名SVG可编辑）。kernel层激活/pool接近PyTorch，完整调用仍明显落后；SGD32kernel对1kernel，当前实现存在FTZ约束，不能直接启用旧融合原型。类别比为样例中位数，不是网络加权贡献。

## 2. 五层卷积逐项对照

图：figures/02_conv_layers.png。展示forward/dWeight/dInput/dBias/full backward，数学操作相同。强调首层孤立input梯度与实际PyTorch训练需求不同；不把孤立层GPU优势宣传为训练优势。Linear功能接口与Layer路径亦有差异，见OPERATOR_REPORT。

## 3. 框架调度是可验证机会

图：figures/03_gpu_cpu.png。实际142 vs72 kernel/step；冻结trace GPU活动1.313 vs1.065ms，kernel跨度内间隙3.585 vs1.027ms。CPU/GPU相互重叠，不相加。38次/step纯shape验证是本轮具体可消除工作。

## 4. 生产代码的小范围修改

图：figures/04_training.png。256项LRU只缓存已验证整数shape，不缓存地址、stream、设备、数值；每次buffer安全校验保留。正式B0 0.851833s → F1 0.786649s，wall减少7.65%。正常PyTorch仍0.312231s。

## 5. 平衡配对验收

图：figures/05_ablation.png。六种三配置排列各2次，12/12胜出；平均节省65.18ms，95%配对区间[61.68,68.69]ms。只有一个候选，不声称双优化叠加。完整291pass、2既有CPUskip，Sanitizer零错误；F1/B0 loss/梯度/权重数值一致。

## 6. 成本与结论边界

图：figures/06_memory.png。GPU内存峰值、分配数相同；CPU cache最多256项，约52–70KB traced分配。batch1/16/TinyNet补充均3/3胜出。推荐显式开启F1、保留默认关闭；未优化CUDA算法。保留时钟波动trace与最后分配诊断共享任务说明，不用其计时宣称收益。

所有图注明A100、batch、方法、单位、数据源。图片为本轮新窗口数据，未将历史2.67×当作本轮性能分母。
