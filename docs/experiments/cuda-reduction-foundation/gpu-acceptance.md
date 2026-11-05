# 下一轮 GPU 验收计划（本轮全部 NOT_RUN）

本轮未选择物理GPU、未执行 nvidia-smi、未创建设备 Context、未运行任何 GPU 计算。
以下是预备脚本与命令；CPU仅生成计划和收集测试，不代表这些任务已执行。

## 单卡窗口和正确性门槛

先获准使用一张空闲 A100，记录 index/UUID/显存/其他进程，将 UUID 映射逻辑0。
用[计划生成器](scripts/gpu_plan.py)和[显式执行入口](scripts/run_gpu_plan.py)，连续顺序运行，
任一失败立即停止且保留日志/NPZ/XML，不换卡、不抢占，也不使用显存占位。
子进程退出释放各自 Context，所有测试属于一个连续实验窗口；控制器本身无CUDA Context。
后续 CPU 汇总、SQLite 导出和绘图在 GPU 工作结束后做。

顺序：

1. 新 Sum 103项GPU用例 + 已有 ReLU GPU测试；包含24种shape/axis、keepdims、默认/非默认流、
   scalar、轴tuple、非连续物化、unsupported原路径、重复/一阶/二阶梯度、Mean/SumTo。
2. **原样重放本轮两份消去失败输入**，与CuPy及独立FP64比较。
   FP32不承诺普遍满足近零的绝对容差；本轮没有解决此限制，不能删除失败用例或放宽阈值。
   若真实GPU也失败，先保存与CPU模型的差异，停止后续性能任务，等待审查。
3. DonkeyNet/SpatialNet，真实4H输入，6组后端共同初始状态的3个单步 + 连续3步：
   Forward、Loss、全部参数梯度、更新后权重；I/S、I/R、I/SR与PyTorch分别比较。
4. 通过后，Sum微基准：CuPy、native完整操作、native预分配ABI、PyTorch。
   24形状×4实现×3重复×30次，5次预热，轮换顺序，保存逐次耗时与workspace。
5. 已有 ReLU 12形状、3重复、30次，复用旧脚本，不修改其Kernel。
6. 匹配训练：compute、io、phases分别运行；DonkeyNet/SpatialNet；10步预热后重置，
   每配置256step、3个独立进程重复，轮换顺序。
7. 最后单独采集Nsight Systems，先微算子再各模型I/S/R/SR/E的16step NVTX阶段。

沿用既有数值阈值：算子/Forward/梯度 `atol=1e-5, rtol=1e-4`；
同状态 Loss/权重 `atol=1e-6, rtol=1e-5`。逐项保存有限值、最大绝对误差、相对L2、超差数。
测试失败时仍保留原始数组，不用 skip 替代GPU执行。启动fixture需 `HYTORCH_REQUIRE_CUDA=1`。

## 公平对照

| 名称 | 后端 | Sum | ReLU Backward | 数据加载 |
|---|---|---|---|---|
| A | 无自研CUDA C、普通SGD、未剪枝的当前代码路径 | original | original | 同一有界预取2worker |
| I | Combined + FusedSGD + 首层剪枝 + 原Gather + Split-K dWeight | original | original | 相同 |
| S | I | native | original | 相同 |
| R | I | original | 既有native原型 | 相同 |
| SR | I | native | 既有native原型 | 相同 |
| E | PyTorch CUDA | PyTorch | PyTorch | 相同 |

不用历史 P/J 名称重命名本轮收益，避免把现有DataLoader或ReLU记作本轮新增Kernel。
全组FP32、TF32关闭、Batch4、SGD .001、seed42、Dropout0、相同输入顺序/初始权重。
窗口控制器设置`NVIDIA_TF32_OVERRIDE=0`，PyTorch同时显式关闭matmul/cuDNN TF32。
复用 `runs/udacity_real_prepared_01/protocol.json`、`throughput-inputs.npz` 以及同一
UdacityDataset/open_stream=True，未改划分/预处理。
未来分别报告 S/I、SR/I、S/A、SR/A 与 S/E、SR/E。不能引用历史窗口百分比拼接。

## 计时和显存口径

- **CUDA Event span**：包含提交空隙的设备事件间隔，不能命名为纯Kernel时间。
- **预分配ABI**：在计时外准备连续输入、SumLayout、workspace、输出；仍有ctypes和提交开销。
- **完整算子同步墙钟**：包含描述符、布局准备、必要分配、调用、最后同步；事件对象创建在计时外。
  微基准当前连续输入为主，非连续数值用例另有测试；要报告非连续性能需另存独立配置。
- **纯Kernel时间和launch数**：单独Nsight采集，通过CUDA API correlationId映射同线程NVTX父范围，
  求和实际CUPTI Kernel duration。不能按GPU执行时段与主机NVTX重叠归因。
- compute是GPU驻留Batch、逐step边界同步；io含原图读取/预处理/H2D/训练/日志与loader关闭；
  phases在阶段边界同步用于诊断。三种计时不同，不混算。epoch wall还包含step之外写日志和Loss提取。
- 显存：本轮脚本保存理论Sum workspace bytes、CuPy活跃增量/池保留量、训练PyTorch peak。
  CuPy池保留量与活跃增量**不是精确峰值**；精确临时分配/跨框架峰值需后续额外监控/Trace，
  本轮不能报告已经测得的GPU峰值。

## 离线 Nsight 解析示例

```bash
nsys export --type sqlite --output runs/cuda_reduction_gpu_future_01/micro-trace.sqlite \
  runs/cuda_reduction_gpu_future_01/micro-trace.nsys-rep
.venv/bin/python docs/experiments/conv-dweight-fusion/scripts/analyze_micro_trace.py \
  --input runs/cuda_reduction_gpu_future_01/micro-trace.sqlite \
  --output runs/cuda_reduction_gpu_future_01/micro-correlated --expected-cases 288
```

微基准标签沿用该解析器格式，每(shape,repeat,implementation)一个case，30次范围，共288case。
其 `kernel_us/kernels/launch_api_*` 与 `kernel-events.csv` 适用；旧分类列 tiles/reduce/Im2col
仅识别旧dWeight名字，不应用来解释Sum。新Sum须按`reduce_sum_kernel`名称过滤，预期每调用
1/2个Kernel。缺少API匹配要报错，不能补0。Nsight插桩结果不与普通训练吞吐混用。
若NCU权限仍受限，使用现有ptxas资源报告和Systems；不修改系统权限，不声称测得Occupancy。
