# 单一方案：Cout 四路独立 FP32 累加链

基线 `e2d60c26cdb28facd4a41dd3b443c3148e397b66`。决定在新文件保留一个通用 Gather 版本：输出元素/线程映射、空间循环、256线程block和现有访存方向不变，仅将 Cout 的一条串行 FMA 链分成四条独立链，最后用 FP32 树形加法合并，余下不足4个通道按原FMA处理。不是四种Kernel或运行时调优。

## 已有实测依据

4E的100形状Nsight原始导出按CUDA API correlation关联。历史c31 `[4,32,33,31]`, W`[64,32,5,5]`, stride2/pad1：旧Gather306.711 μs、原路径89.772 μs；c15 `[16,64,17,19]`, W`[64,64,3,3]`：253.292 μs与81.292 μs；c26 `[1,3,31,35]`, W`[64,3,5,5]`：168.513 μs与35.339 μs。原路径各有2或3个Kernel，而旧Gather仅1个。完整100行留在 results/history-shapes.csv，来源是4E results/profiles-01/shape-kernels.csv。

这说明减少Launch没有消除旧Gather的GPU计算成本；不能据此把高Cout或FMA依赖断言为唯一瓶颈，因为形状、输出并行度和访存量也同时变化。

## 从源码确认的事实与待验证假设

- 旧Kernel每线程处理一个NCHW dInput元素，按Kh/Kw/Cout顺序计算；Cout每一步依赖前一步partial，最长链为Cout。
- 同一warp多数相邻线程沿W展开，访问同一通道gy的相邻空间；stride2会使部分lane在不同kh/kw分支活跃。相同Cin/kh/kw时Weight是warp广播式重复读取；不同Cin间gy存在复用机会，缓存是否有效须硬件数据验证。
- old Weight/gy都从Global地址空间读取，不等于每个load都访问DRAM。循环内地址计算、除余/边界分支和串行FMA均有成本，当前没有NCU计数器证据将其中某项定性为主瓶颈。
- 选定的假设是四路独立链提高每线程指令并行度，给load/FMA更多调度余地；不改变现有较合理的空间合并方向。四路结果树形归约会改变FP32舍入顺序，必须沿用原容差实测。
- 风险是更多寄存器、额外终结加法和代码体积可能抵消收益，尤其小Cout/小输出。不另加形状开关掩盖退化，不写模型/层号或固定张量尺寸分支。

没有选择Shared Memory或warp沿Cout拆分：前者需要新的tile/同步布局，后者会改变gy合并访问；本轮先隔离一个小而可解释的计算改动。未声称这个选择已经得到GPU加速。

## 冻结与支持范围

在新Kernel实现和新验证集计时之前固定四路方案，不依据后续验证结果增加其他方案或调整Auto。支持连续FP32 NCHW/OIHW、groups1/dilation1，任意合法N/Cin/Cout、正Kh/Kw，stride每轴1/2、padding每轴0/1，沿用原ABI尺寸约束。Cout尾数有通用余数循环。

新的24个独立形状包含不同Batch/Cin/Cout比例、未见空间、非4倍数Cout和矩形kernel/混合stride/pad；只用于验证，不参与机制选择。所有结果（包括退化）保留。旧 original/fused/auto和默认模式不变，新模式只显式 `fused_opt`，Auto不会选择新Kernel。
