# 原始数据索引

完整服务器根目录：`/home/xymo/projects/hytorch/runs/cuda_reduction_01/`。
`/home/xymo`对应本服务器`/srv/lab-disks/disk2/users/xymo`，索引记录实际规范路径。
没有GPU原始数据；所有GPU项NOT_RUN。

| 内容 | GitHub可移植记录 | 服务器完整原始文件 |
|---|---|---|
| 系统/包/线程限制 | [environment](environment) | 同目录 |
| SM80构建、完整nvcc参数/资源报告 | [build-01](results/raw-01/cuda_reduction_01/build-01)、[包装日志](logs/build-01.log) | `runs/cuda_reduction_01/build-01/`（含`.so`） |
| 主机57项、完整196项 | [日志](logs)、[XML](results/regression-01.xml) | 同目录 |
| CPU算术首次/完整保存/最终失败返回码 | [cpu-reference-01](results/raw-01/cuda_reduction_01/cpu-reference-01)、[-02](results/raw-01/cuda_reduction_01/cpu-reference-02)、[-03](results/raw-01/cuda_reduction_01/cpu-reference-03) | 对应目录含全部输入/输出及失败NPZ |
| CPU两模型三步比较312项 | [check-host-01](results/raw-01/cuda_reduction_01/check-host-01) | `runs/cuda_reduction_01/check-host-01/*-values.npz` |
| 24形状微算子预备配置 | [micro-plan-01](results/raw-01/cuda_reduction_01/micro-plan-01) | 同名runs子目录 |
| 12组训练预备配置 | [training-plans-01](results/raw-01/cuda_reduction_01/training-plans-01)、[CPU执行命令](results/training-plans-01.json) | 同名runs子目录 |
| 单卡窗口123条顺序命令 | [plan.json](results/raw-01/cuda_reduction_01/gpu-plan-01/plan.json) | `runs/cuda_reduction_01/gpu-plan-01/plan.json` |
| 所有服务器原始文件逐文件清单 | [file-index.csv](results/raw-01/file-index.csv) | 339个文件、36份可移植复制；不计算哈希 |
| 数值图 | [完整误差](figures/cpu-01/cpu-errors.png)、[消去负样本](figures/cpu-01/cpu-cancellation.png)、[图中数值](figures/cpu-01/cancellation-values.csv) | 同目录 |

大型输入/权重NPZ和动态库未上传。339个文件指本轮runs归档快照，不包含docs里的日志、测试XML与图。
所有原始数据/旧失败日志保留；PNG为CSV/NPZ数据可视化，不是终端截图。
新CUDA源码在仓库 [reduction.cu](../../../hytorch/backends/cuda/reduction.cu)，
CPU参考在 [reference.py](scripts/reference.py)；不保留重复的源码副本。
