# DataLoader 与真实数据路径审查

基线：`9909e546d9314e342663d75c4a10d1f46b4b7351`，来自
`experiment/hytorch-autonomous-optimization`。本轮所有计算仅用 CPU。

## 实际调用与顺序

`DataLoader(num_workers=0)` → `_SyncDataLoader.__next__()` → 按 batch_index
逐项 `dataset[i]` → 两次 `np.array` 组装图像与标签。默认同步行为不变。

`num_workers>0` 复用现有 `AsyncDataLoader`，不是新增异步实现。每个 Epoch
`__iter__()` 关闭上一批线程，创建任务队列、结果队列及指定数量的 daemon 线程。
每个任务是一整个 Batch，包含 `(seq, batch_index)`；一个 Worker 在该 Batch 内
顺序读取样本、预处理并组装两个 NumPy 数组，返回 `(seq, x, t)`。

消费端以 `_next_seq` 和 `_pending` 字典恢复任务顺序。`indices` 是前一分支已有
接口，支持重复样本、指定顺序和不足一个 Batch 的末批；本轮没有重新实现。
`shuffle=True` 对提供的索引排列，`False` 保留它。Worker 不进行 H2D 或模型计算。

`prefetch_factor * num_workers` 仅限制 **result_queue** 的 Batch 容量。任务队列
一次放入整个 Epoch 的索引任务；顺序缓冲 `_pending` 不受该容量约束。可复现的
首 Batch 延迟用例在容量2时积累99个小 Batch，见
[原始记录](results/pending-observation.json)。这是原有顺序缓冲语义，不是新的
严格内存上界。本轮不更换调度架构，也不宣称已经修复该限制。

## 已复现的生命周期 Bug 与修复

原 `_shutdown_workers()` 仅排空任务队列、加入结束任务、每个 Worker 最多等待5秒。
当结果队列已满时，Worker 阻塞在 `result_queue.put()`，无法领取结束任务。
关闭函数随后清空 `_workers` 并标为未启动，两个线程依然存活且引用已丢失；
重新迭代会建立新队列和线程，旧线程仍滞留。这个问题同样影响提前退出和错误清理。

原始代码保存在服务器 `runs/cpu_pipeline_01/baseline_dataloaders.py`，内容来自上述
Git基线，未进行文件哈希。两线程满队列复现：关闭10.001204秒后仍有2个活线程。
同一脚本对修复版测得0.000108秒、0个存活线程。这是单次合成复现，不是训练加速。

最小修复位于 `hytorch/dataloaders.py`：

- 每个 Epoch 一个 stop Event；Worker 提交结果使用可检查退出信号的限时 put。
- 关闭时先设置 Event、排空结果队列以唤醒已阻塞提交，再 join 全部当前 Worker。
- 清除 pending、队列与 Worker 引用；自然耗尽、重新迭代、异常处理均使用该路径。
- 新增 `close()` 与公开 DataLoader 上下文管理，用于明确提前结束 Epoch。

```python
with DataLoader(dataset, 4, shuffle=False, num_workers=2,
                prefetch_factor=2, indices=existing_order) as loader:
    for x, t in loader:
        if finished:
            break
```

单独 `break` 不会自动销毁仍被引用的 Python 迭代器，因此提前退出应使用上面的
上下文或显式 `close()`。无法强行终止正在执行的任意用户 `dataset[i]`；关闭会
等待该调用返回，然后真实 join，而不是在超时后把存活线程当作已关闭。
没有添加 CUDA 调用、持久线程池、异常吞并或生产 Runtime 改动。

## 复制与预处理

`UdacityDataset.__getitem__` → `preprocess_udacity`：PIL打开 JPEG，转换 RGB
（触发懒解码），裁掉顶部70及底部25像素，双线性 Resize 到200×66，FP32除127.5
再减1，HWC→连续CHW；标签为独立FP32单元素数组。

图像转为FP32、归一化表达式、中间结果及 CHW 连续化都会产生数组。
Batch组装再复制到连续 `(N,3,66,200)` 数组，为当前接口提供拥有自身内存的 Batch。
异步消费端 CPU 模式直接返回 Worker 的数组，没有额外复制整个 Batch；pending
只保存数组引用。标签 copy 很小，本轮不改变其所有权语义。

独立候选 `pipeline_common.inplace_preprocess` 把除法和减法作用于新建FP32数组，
两次舍入仍保持FP32。1024张实际图像逐元素一致，但吞吐没有跨配置稳定提升，
所以 **没有修改正式归一化实现**。

## 依据cProfile保留的文件句柄选项

Pillow路径打开在当前版本会执行`realpath`，正式1024图profile中累计0.089223秒，
10240次lstat。只改变打开入口：`UdacityDataset(open_stream=True)`先用Path.open，
将句柄交给原`preprocess_udacity`。上下文在处理成功或失败时关闭文件。
默认False不变，没有修改任何预处理运算，也不保留图像缓存。

原型30次独立进程后，集成API再做18次独立进程匹配测量。两批分别保存，
对比基线Dataset来自Git原始源码快照，避免集成后脚本误测新路径为旧基线。
两批2线程分别平均+9.09%/+13.33%，每次重复均更快；同步和4线程存在负重复，
不做普遍稳定提升声明。新路径cProfile无`realpath`调用，支持路径解析成本的归因，
但不能把全部墙钟增益精确等同于该嵌套函数时间。

## 计时边界与归因

正式性能使用未插桩 DataLoader，独立子进程逐批丢弃输出，不保留整套图像，
包括迭代器创建和自然耗尽。每 Batch 的 `next()` 墙钟包含同步加载或队列等待，
不是纯 Queue 成本。CPU占比是进程 CPU秒/墙钟，100%表示一个核；RSS是该子进程
从启动到测量结束的生命周期高水位，包含Python/CuPy导入和数据集元数据。

分阶段诊断显式 `read_bytes` 后在内存中解码，以区分文件读取与JPEG计算；此时
文件路径与正式PIL懒加载不同。逐图输出均与正式预处理比较；不能把这些阶段相加
当成另一个训练墙钟。另保留正式同步路径的 cProfile，累计时间存在嵌套。

Queue诊断包装 put/get、NumPy组装和样本调用，仅用于分析。Worker间的耗时会
重叠，Queue耗时也包含锁、唤醒和调度，不能全部相加当成独占CPU执行时间。
该诊断不参与90次正式吞吐结果。

## 下一次 GPU 对照

复用上一分支的 Engine、初始化、SGD和后端配置；I/P/R/J分别保持既定定义。
新增统一入口只选择共享 CPU DataLoader 参数：E与I/R同步，E-prefetch与P/J使用
相同2线程、prefetch=2，所有路径统一显式open_stream=True。额外保留A作为无Native基线。原生ReLU是前一分支已准备、
尚未GPU验收的原型，本轮没有开发或修改任何CUDA代码。

`future_training.py --plan` 仅创建CPU元数据。只有显式 `--run-gpu` 才导入训练
Engine并执行设备代码，本轮未运行此分支。每个版本使用同一Batch4、seed42、
SGD .001、关闭Dropout/TF32、相同真实图像顺序、10步预热后恢复权重。
普通吞吐采用相同 StepProfiler 边界同步/标量日志口径，完整Epoch包含数据加载及
线程清理；phases单独测量，不与普通吞吐混用。126项未来任务仅为计划，不是GPU结果。
