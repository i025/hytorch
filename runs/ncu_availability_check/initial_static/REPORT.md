# Nsight Compute 可用性检查

检查时间：2026-11-08 07:58 UTC。项目：`/home/xieminhui_pubuse/hytorch`。

## 结论

**NCU 已安装且静态环境看起来满足 A100 profiling 的基本条件，但本轮无法确认实际 profiling 可用性：两张 GPU 均被占用，因此按要求停止在静态检查。** 这不是一次 NCU profiling 失败，也不能归因为计数器权限不足。

| 问题 | 结论 |
|---|---|
| 服务器能否使用 NCU？ | CLI可运行；实际kernel profiling/计数器采集尚未验证。 |
| 当前用户能采哪些指标？ | 本轮成功采集0项。已确认basic及相关section在安装包中存在，不等于设备采集成功。 |
| 具体错误？ | 默认PATH下`ncu --version`、`nvcc --version`退出127，原因是PATH缺少目录；绝对路径可运行。没有启动profiling，没有NCU profiling错误码。 |
| 安装、版本、权限还是其他问题？ | 安装存在，版本无已发现的静态冲突；当前阻断是GPU占用。不能确认或排除实际运行时的其他问题。 |
| 是否需要管理员？ | 当前无证据支持修改权限/驱动。优先协调GPU1空闲时间。仅当未来实测出现权限错误等，再请管理员核查。 |
| 能否分析HyTorch GEMM？ | 设计上可以尝试已有gemm_register；本轮未执行，不能宣称成功。 |
| 推荐哪些section？ | basic中的SpeedOfLight、Occupancy、LaunchStats；basic成功后再少量增加SchedulerStats、WarpStateStats。均已在本机列表核对。 |

## 安装和版本

| 项目 | 结果 |
|---|---|
| `which ncu` / `which nvcc` | 均退出1，PATH中没有CUDA bin目录 |
| 默认`ncu --version` / `nvcc --version` | 均退出127，command not found |
| 推荐NCU绝对路径 | `/opt/nvidia/nsight-compute/2025.2.1/ncu` |
| NCU实际版本 | 2025.2.1.0，build35987062；版本命令退出0 |
| `/usr/local/cuda/bin/ncu` | 同样报告2025.2.1.0，路径解析为cuda-12.4/bin/ncu |
| 默认CUDA Toolkit绝对路径 | `/usr/local/cuda/bin/nvcc`，12.4.99 |
| NVIDIA驱动 | 580.95.05 |
| nvidia-smi显示CUDA | 13.0，代表驱动兼容能力，不代表当前nvcc版本 |
| Python / Torch | Python3.10.12，Torch2.9.1+cu128，编译CUDA12.8 |
| CuPy发行包 | cupy-cuda12x 14.2.0；本轮未初始化CUDA来查询运行时 |
| glibc | 2.35；安装包要求Linux glibc≥2.27 |

安装包自带 `docs/ReleaseNotes/topics/gpu-support.html` 将 NVIDIA GA100 标为Yes，A100硬件在支持范围；系统要求和摘录保存于compatibility_notes.txt。Python环境仅CPU导入Torch，`torch.cuda.is_initialized()`为False。CLI可运行和静态版本信息不能代替完整运行时兼容性验证。

## 权限

当前用户 `xieminhui_pubuse`，UID/GID1291600100，非root，CapEff=0。实际读取：

```text
RmProfilingAdminOnly: 0
```

这说明当前读取到的驱动管理员专用限制关闭；不能因用户不是root就断言无法采集。NVIDIA说明限制开启时普通用户可能遇到ERR_NVGPUCTRPERM，而限制关闭允许普通用户访问该机制：[官方权限说明](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters)。本机R580应结合该说明的legacy kernel regkey部分理解，不套用文档中R610新机制。

**本轮没有观察到ERR_NVGPUCTRPERM，但原因是未运行profiling，不是实测证明权限正常。** 没有sudo、setcap、模块参数修改、驱动重载、重启或系统PATH变更。

## GPU资源与停止原因

| 物理GPU | 显存占用 | 进程 | 利用率 |
|---|---:|---|---|
| 0 A100-PCIE-40GB | 20,543MiB（进程20,534MiB） | PID570250，VLLM::EngineCore | 快照0%，仍属占用 |
| 1 A100-PCIE-40GB | 3,009MiB（进程3,000MiB） | PID575118，python | 连续查询分别4%与92% |

GPU1 UUID：`GPU-0bc09ee6-967a-b53d-bf80-38379585cf08`。没有其他空闲GPU。因此没有启动PyTorch GPU算子、NCU basic或GEMM测试。没有接触/终止GPU0的VLLM或GPU1的任务，没有持续等待或抢占设备。

## 测试记录的含义

- `ncu_basic.log`：明确记录NOT_RUN_GPU_BUSY；profiling退出码为N/A，不伪造为0；stdout/stderr无profiling内容。
- kernel识别：未测试；计数器采集：未测试；有效.ncu-rep：未生成。
- `ncu_gemm.log`与`metrics.csv`未创建，因为没有执行GEMM profiling或采到指标。
- `ncu_sets.txt` / `ncu_sections.txt` / `ncu_help.txt`均为实际CLI静态查询，退出0。section查询不是profiling。
- `metric_candidates.csv`是安装包section文件中的候选名称及来源行号，不是采集结果，没有设备级query验证。

## 后续最小测试建议（本轮未执行）

空闲且授权后，用绝对路径执行basic、launch-count=1。先确认报告中出现kernel名称和实际metric值，不能以程序打印2.0作为成功依据。复现命令使用`--clock-control none`，避免NCU默认base时钟控制；不更改系统环境。

PyTorch basic确实成功后，再执行一个1024³ FP32自研GEMM。源码hy_gemm会得到16×16=256个CTA，满足128CTA阈值，选择gemm_register<16>，block16×16；这是静态源码推导，需用NCU实际kernel名称确认。`gemm_minimal.py`已准备，但没有执行、没有构建或修改生产库。

| 目标 | 已存在的section | 安装包中的候选metric（未采集） |
|---|---|---|
| SM Throughput | SpeedOfLight | sm__throughput.avg.pct_of_peak_sustained_elapsed |
| Memory Throughput | SpeedOfLight | gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed |
| Achieved Occupancy | Occupancy | sm__warps_active.avg.pct_of_peak_sustained_active |
| Registers/Thread | LaunchStats | launch__registers_per_thread |
| Warp stall | WarpStateStats | long_scoreboard/barrier对应ratio，完整名称见metric_candidates.csv |
| Warp调度 | SchedulerStats | 使用版本自带section，由工具解析设备支持的指标 |

优先section而不是未经设备查询的手写metric列表。Memory Throughput不是单纯DRAM带宽；需要详细访存分析再考虑MemoryWorkloadAnalysis。basic若成功而扩展section失败，应归类为部分可用，并保留具体错误；不建议直接使用包含数千指标的full set。

若未来确认计数器被拒绝，请管理员核查当前宿主机/容器策略并安排授权测试；本轮不执行任何绕过。无需改共享服务器配置的替代是使用既有Nsight Systems CUDA时间线、CUDA events和CPU profiling，分析kernel时长、launch与同步间隙；它们不能替代实际occupancy/warp stall硬件计数器。

## 完整性

仅新增runs/ncu_availability_check中的诊断文件；生产源码、测试、benchmark和动态库按SHA核对未改变。没有Commit、Push、PR、安装工具或大规模benchmark。本轮静态检查已完成并停止。
