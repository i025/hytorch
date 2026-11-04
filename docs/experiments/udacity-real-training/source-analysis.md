# 阶段4H源码路径与计时口径

基线`a5a50159f6217f10eee8028b31503a444212b1ac`。本轮不开发CUDA Kernel，不改变Conv2d、自动求导或优化器。复用4G的SM80动态库`runs/conv_dweight_build_01/libhytorch_cuda.so`，离线ABI1和dWeight符号检查通过，不进行内容哈希验证。

## 数据与模型

新增`donkeycar_hy/udacity_dataset.py`，从driving_log.csv读取center路径和steering，显式使用CSV标签，不解析DonkeyCar文件名标签。CSV中来源机器路径统一按本地`IMG/basename`解析。左右摄像头仅检查文件存在和图像格式，不参与训练，不构造修正标签。

训练、验证及未来单图预测共用`preprocess_udacity`：Pillow RGB→裁去顶部70行、底部25行→bilinear resize至200×66→float32像素/127.5−1→连续CHW。没有随机增强或颜色fallback。采用RGB，不声称与其他项目的YUV配置等价。

模型复用`donkeycar_hy/model.py:DonkeyNet(dropout_rate=0)`；PyTorch直接复用`benchmarks/bench_cuda_training.py:torch_model`。均为五层Conv/ReLU、Flatten、Linear100/ReLU、Linear50/ReLU、Linear1/Tanh、MSE，SGD lr=.001，无momentum/weight decay。初始化复用`initialize(seed=42)`，按参数数组直接比较初始状态，不调用state_hash。FP32，TF32关闭，batch4，训练末批2张保留，不丢样本。

## 四种执行路径

| 版本 | 实际开关 |
|---|---|
| A | backend默认CuPy，普通SGD，未剪枝；性能子进程指向不存在的Native库，dispatch必须为空 |
| F | matmul shared、conv im2col、pool CuPy、tile16、普通4D Gather dInput、Fused SGD、首层输入梯度剪枝 |
| I | F加`conv_dweight=fused`，4G Split-K Implicit GEMM |
| E | 匹配PyTorch CUDA模型/SGD，标准cuDNN，关闭TF32和benchmark，deterministic开启 |

A是当前代码复现的无本项目Native CUDA C基线，CuPy/cuBLAS内部仍使用CUDA；不是历史最初commit的原始性能。本轮不使用4F fused_opt或Auto。`scripts/engine.py`只复用模型、初始化、设备转换和训练操作，实际计算留在原模块。

## 独立的计时实验

`scripts/train.py`与既有合成benchmark分开，不改变旧数据处理或默认CLI。

以下是已准备的执行协议，不代表均已实测；实际PASS/NOT_RUN/中断状态以本阶段README和原始窗口记录为准。

- compute：1024张固定真实center图像预处理后一次传入GPU，10步预热并恢复初始参数，正式256步；epoch墙钟包含Python、模型、同步、日志，排除磁盘、预处理和H2D，不能当作纯Kernel时间。
- io：同1024图像、同顺序，每步实际读JPEG/预处理/传输，10预热+256步；保留逐步read_preprocess_s及完整墙钟。操作系统页缓存不清理，因此不是冷盘测试。
- phases：同样真实I/O，独立同步阶段实验，记录data/transfer/forward/backward/update毫秒，不参与未插桩compute/io吞吐比。
- train：6330张训练图像完整3个epoch，逐epoch固定seed42+epoch的顺序，验证1608张全部图像；含阶段同步以记录各epoch的实际F/B/U，明确与未插桩吞吐分开。保存epoch0及每轮验证预测、MSE/MAE/RMSE、每步Loss和最终权重。
- trace：F/I各16个真实图像step，单独Nsight Systems；复用4G WeightRanges和4A correlationId分析，10步预热不算正式phase。原始Trace留服务器，GPU释放后才导出SQLite。

Loss有限检查、参数实际变化、FP32数值对照保留。单步梯度/forward容差1e-5+1e-4相对，Loss/更新权重1e-6+1e-5相对；正式分析不修改标准，长训练若有误差也照实保存。

## 共享GPU生命周期

统一入口`scripts/run_window.py`为纯CPU控制器，完成离线准备后只读查询物理GPU1，取实际UUID写入CUDA_VISIBLE_DEVICES；子进程仅逻辑0，按47任务串行执行。原任务PID、显存、GPU利用率/温度/功耗/时钟以及pmon原始输出均记录。基线PID只观测，不发信号。监控只可能终止本轮新建session/process group。

控制器不创建CUDA context，不执行cudaDeviceReset或GPU Reset，不设置电源、时钟、MIG/MPS，不用显存/忙循环占位。CUDA资源由本轮子进程退出释放。预计进程显存约2GiB，基于4G实测并计入163MB驻留输入；启动需至少3GiB空闲。剩余显存低于1GiB、基线原进程消失、整卡利用率持续≥95%达15秒或监控/子任务失败时停止后续任务并保存原始进度。

原进程退出可能是正常结束，也可能异常；没有其应用延迟/日志时无法判定原因或承诺零影响。所有成绩标记SHARED_GPU，不作为独占A100基准，不把整卡利用率或间隙当作本进程SM利用率。

实际首次窗口中一个原PID退出，监控按约定停止。用户确认这是正常结束并批准同GPU1从头重测；第二窗口47项全部完成，现有原PID始终可见。所有正式性能、全数据训练和Nsight均来自第二个连续窗口，首次数据保留不混算。真实长训练的严格末态/Loss比较失败，局部短程通过不替代长期验收，详见README及drift-01。
