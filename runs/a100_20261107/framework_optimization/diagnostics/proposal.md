# 待实施候选 F1：有界不可变 shape 校验缓存

阶段A CPU cProfile：30step 中 _shape_ok 1140次（38/step），累计约13ms；其中np.prod约9ms。此数字仅用于定位，不能作为正式wall收益。backward排序与add_func并未进入主要热点前列，暂不改自动求导调度。

计划在CUDA backend增加 metadata_cache=False。仅缓存精确tuple内精确Python int形成的shape校验结果，maxsize256；其它输入类型走原路径，避免float/int相等导致绕过类型语义。miss执行原有完整校验，不改变np.int64乘积、报错类型和门限；缓存不保存array、指针、device、stream或数值，不分配GPUworkspace。所有_buffers/dtype/device/alias检查保持逐调用执行。cache安全依据是纯shape检查只依赖不可变值，与GPU设备无关。

不引入第二优化；没有证据支持为排序/对象系统风险付费。完整受影响调用路径和10轮训练通过后决定保留，默认关闭。
