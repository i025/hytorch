# GPU0七小时研究：阶段性汇总

GPU窗口未执行。实际监控样本44条，首尾跨度0.7176小时；
不是把任务预算直接当实测时长。最大采样间隔60.100065秒，最高连续空闲计数0。
物理GPU0 / `GPU-f099fdbd-adf2-6157-3462-4145bc2a8359`，未切卡。

CPU回归210通过，0失败，0跳过；heap调度CPU188项逐元素一致；CPU Sum算术模型训练结果见JSON。
CPU合成图收益不当作GPU或自动驾驶训练收益。原始失败和修正记录保留。

GPU Sum：{'status': 'NOT_RUN'}。
GPU ReLU：{'status': 'NOT_RUN'}。
GPU基线回归：{'status': 'NOT_RUN'}。
新调度GPU：{'status': 'NOT_RUN'}。

正式训练完成0次，只有至少3次独立重复的组才计算正式比值。
GPU未执行或组不完整时，不填造Kernel时间、吞吐、累计收益、与PyTorch差距或Nsight结果。
统计口径、协议复核和窗口每项完整命令见本目录summary.json、training CSV和服务器window.json。
Sum强消去超差即使符合CPU舍入模型也保留FAIL；不改容差，不能声称通用数值验收通过。

原始目录：`/srv/lab-disks/disk2/users/xymo/projects/hytorch/runs/gpu0_autonomous_7h_01`。本次图像为原始CSV数据可视化，不是终端截图。
