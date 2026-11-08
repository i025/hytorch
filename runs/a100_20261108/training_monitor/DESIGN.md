# TrainingMonitor 设计与 API

仓库现有 hytorch/profiling.py 是模块文件，因此新增 hytorch/training_monitor.py，并由 profiling.py 导出 TrainingMonitor，避免破坏已有 import。

```python
from hytorch.profiling import TrainingMonitor
m = TrainingMonitor(enabled=True, gpu_index=0, sample_interval=.2,
                    output_dir="runs/my_monitor", cuda_backend="cupy")
with m:
    it = iter(loader)
    for i in range(len(loader)):
        with m.step(i, epoch=0):
            with m.phase("data_load", gpu=False):
                batch = next(it)
            with m.phase("forward"):
                prediction = model(batch)
            # 用 phase 分别包裹准备、传输、loss、backward、update。
print(m.run_dir, m.summary())
```

可直接运行 examples/training_monitor.py；通用 monitored_epochs 位于 hytorch/monitor_loop.py，显式包裹 next(iterator)，通过 transfer/predict/loss_fn/backward/update/retain_loss 回调接入任意模型。retain_loss 应返回 detached 数组，避免保留计算图。支持自定义 prepare_batch；此次合成输入准备为 identity 加 metadata，不冒充图片解码。

构造默认 enabled=False；禁用时不导入 NVML/psutil/GPU 库、不建采样线程、不输出文件。timing/resources 可分别开关；CPU 使用 cuda_backend=None、gpu_index=None；Torch 使用 cuda_backend='torch'。每次 start 创建唯一 run 子目录，异常退出也 stop/export；同一实例可再次运行。step/phase 由创建运行的线程操作，拒绝嵌套、重复 start 和错误生命周期。提供 start/stop、step、phase、set_step_metadata、flush_events、summary、export。

GPU 设备用当前 runtime 的 UUID/PCI 对应 NVML handle，记录逻辑编号、物理编号、CUDA_VISIBLE_DEVICES；不根据掩码字符串猜编号。纯资源模式可显式 UUID/PCI，避免与 runtime selector 冲突。后台线程只选择其自身逻辑设备（非零时），不修改主线程设备/stream。NVML 或 psutil 不可用时继续训练、记录错误和 null。

采样线程 priming psutil，第一条 percent=null。单次读取 NVML、psutil、allocator 计数，不创建 nvidia-smi 子进程、不主动复制 GPU 数据或同步。样本保存实际时钟、计划时钟、延迟和采集耗时；CPU phase 标签只是采样开始时的 host 标签。停止时通知、join 并关闭 NVML。

phase 在当前 CUDA stream 上记录起止 events，退出 phase 不同步；stop/flush_events 在明确完成边界同步相关 stream 后批量读事件。phase 首尾 stream 改变时该 event 区间标为 unsupported；不能捕获所有跨流依赖或并发 GPU 工作，阶段间区间可能重叠，禁止直接求和当总计算量。throughput 保存 host_iteration_s，CUDA step_wall_s=null；latency 在 step 末同步并记录完成延迟，仅用于诊断。

事件及明细保留量为 O(steps×phases)，长训练应在安全 epoch 边界 flush_events；明细仍保存在内存，极长任务需要后续分块导出设计。allocator_peaks 默认关闭。CuPy hook 跟踪主线程默认 pool 分配瞬间，记录 latest-equal peak 的 CPU 标签；其他线程/custom allocator 不在该“精确”范围。Torch opt-in 会重置目标设备峰值计数，影响共享这些计数的同进程代码，应仅诊断使用。

CLI --monitor 启用新循环；--monitor-components disabled/timing/resources/all；--monitor-mode throughput/latency；--monitor-interval；--monitor-allocator-peaks；--output。--final-config 选取已接受配置，不改 kernel。没有 --monitor 时保留已有 StepProfiler 训练分支，资源采样默认关闭。--nvtx 可独立配合 helper 用于外部 trace。stop 导出耗时计入完整生命周期开销，但不计入 training_window_s。
