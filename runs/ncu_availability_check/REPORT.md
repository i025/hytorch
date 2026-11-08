# Nsight Compute 可用性检查：实际测试后的结论

更新时间：2026-11-08 08:21 UTC。本轮用户要求“继续”后，重新检查设备并执行了一个最小NCU测试。初次因GPU占用而停止的报告和日志保留在initial_static/。

## 七个问题的回答

| 问题 | 当前实际结论 |
|---|---|
| A100服务器能否使用NCU？ | 已安装、CLI能运行、能连接CUDA程序并识别kernel；但当前默认临时目录下profiling失败，尚不能确认硬件指标采集可用。 |
| 当前用户能采集哪些指标？ | 本轮0项。无有效性能指标报告，不将识别kernel视为采集成功。 |
| 具体错误？ | `InterprocessLockFailed`：无法打开/创建`/tmp/nsight-compute-lock`。NCU退出码9。 |
| 安装、版本、权限还是其他问题？ | 明确阻断为NCU跨进程锁文件访问失败，属于用户态文件系统/安全策略问题；不是本轮观察到的Performance Counter权限错误。 |
| 是否需要管理员？ | 解决共享锁的访问策略可能需要文件所有者/管理员协调；目前无依据修改NVIDIA驱动计数器权限。另有官方文档规定的、满足无并发NCU前提的进程级TMPDIR替代方案，本轮只记录，未执行。 |
| 能否用NCU分析HyTorch自研GEMM？ | 当前受同一阻断影响，未执行GEMM profiling；只有basic确实成功后才继续。 |
| 后续推荐哪些section？ | basic成功后：SpeedOfLight、LaunchStats、Occupancy；随后按需增加SchedulerStats、WarpStateStats。这些section已在本机确认存在，具体计数器尚未实测。 |

## 安装与静态环境

- `/opt/nvidia/nsight-compute/2025.2.1/ncu`：2025.2.1.0，build35987062。
- `/usr/local/cuda/bin/ncu`同版本；默认PATH未包含它，因此裸`ncu`报command not found，这不是未安装。
- nvcc绝对路径版本12.4.99；驱动580.95.05；nvidia-smi显示CUDA13.0为驱动兼容能力。
- Python3.10.12，Torch2.9.1+cu128，CuPy发行包14.2.0；安装包支持GA100，glibc2.35满足其最低要求。
- 当前用户xieminhui_pubuse，UID1291600100，非root；`RmProfilingAdminOnly: 0`，CapEff=0。
- 没有发现静态版本冲突；实际能连接进程和识别kernel，但不足以认证完整profiling兼容性。

完整版本、PATH、命令退出码在environment.txt、permissions.txt、static_commands.json；候选metric及其安装包来源在metric_candidates.csv，后者不是实际采集数据。

## GPU资源与最小测试

08:21 UTC重新检查时，GPU1为0MiB、0%且没有计算进程，物理UUID为`GPU-0bc09ee6-967a-b53d-bf80-38379585cf08`。GPU0继续运行VLLM PID570250，约20GB显存，未触碰。

在物理GPU1运行512×512 PyTorch ones/add测试，NCU只允许采集1个kernel；使用basic set，clock-control=none、pipeline-boost-state=dynamic、cache-control=none。没有训练或大规模benchmark，没有锁定时钟。

结果：

```text
==PROF== Connected to process 645591 (/usr/bin/python3.10)
==ERROR== Failed to open/create lock file /tmp/nsight-compute-lock.
==ERROR== InterprocessLockFailed
==ERROR== Failed to profile "vectorized_elementwise_kernel" in process 645591
==ERROR== The application returned an error code (9).
```

- NCU实际退出码：**9**。
- kernel识别：**是**，vectorized_elementwise_kernel。
- 性能计数器采集：**否**；有效metrics：**0**；没有生成可用.ncu-rep。
- 程序被NCU错误路径终止，没有以“计算成功”冒充profiling成功。
- 本轮未观察到ERR_NVGPUCTRPERM；由于阻断发生在锁阶段，硬件计数器权限仍未实际验证。
- 按失败时停止的要求，没有重试profiling、没有执行GEMM或更深指标采集。

原始stdout、stderr、完整命令、时间和退出码位于`recheck_20261108T082100Z/`。顶层ncu_basic.log为本次真实失败日志；初次NOT_RUN日志保留在initial_static/ncu_basic.log。

## 锁文件诊断：事实与推断

只读检查确认：

| 项目 | 结果 |
|---|---|
| /tmp | root所有，权限1777，sticky bit开启 |
| /tmp/nsight-compute-lock | 普通空文件，所有者tzj（UID1291600034），权限0666 |
| 当前用户 | UID1291600100，与锁文件所有者不同 |
| ACL | owner/group/other均rw，无额外限制条目 |
| os.access(W_OK) | True；普通权限检查允许写入，不代表NCU的open/create一定允许 |
| fs.protected_regular | 读取被拒绝，**实际值未知** |
| fs.protected_fifos | 读取被拒绝；不是本轮已确认根因 |

因此不能简单归因为文件mode缺少写权限。安装的NCU 2025.2.1文档明确指出：sticky临时目录下，fs.protected_regular可能导致其他用户无法访问该锁文件，即使传统权限看起来允许。当前现象与此情况吻合，但由于系统参数不可读，**这只是可能原因，未证实该值为1或2**。确证的结论仅为NCU无法打开/创建共享锁。

权威来源：本机`/opt/nvidia/nsight-compute/2025.2.1/docs/ProfilingGuide/index.html`第2805–2811行，摘录保存在本次recheck的installed_faq_excerpt.html。锁用于序列化NCU进程，不能当作普通无用临时文件删除。当前在线[Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#faq)可能随版本变化，本报告具体机制依据已安装2025.2.1文档。

未删除、chmod、chown共享锁；未使用sudo/setcap/sysctl；未改NVIDIA驱动配置、模块参数或重启。

## 管理员协助与无需系统修改的替代

1. 共享锁的常规解决：请文件所有者/管理员在确认无NCU使用者后核查锁文件及临时目录安全策略。不要据此直接开放或关闭驱动计数器限制。
2. 安装版官方FAQ允许：**确认同一系统没有并发NCU实例时**，为本次进程设置一个当前用户可写的不同TMPDIR。它不需要改共享/tmp锁或系统设置，但会改变锁的协调路径，因此不能在未确认无并发NCU时直接使用。本轮遵守失败后停止，仅提供建议，没有执行该替代路径。
3. CUDA events、CPU profiling和既有Nsight Systems时间线可用于kernel耗时/launch/同步分析，不提供相同的SM吞吐、实际occupancy和warp stall计数器，不能冒充NCU替代指标。

## 后续自研GEMM与指标范围

待basic成功后，再使用现有gemm_minimal.py测试1024³ FP32 GEMM。根据当前hy_gemm源码，256个输出CTA满足128CTA门限，应选gemm_register<16>；该选择需由将来的NCU实际kernel名称验证。本轮未执行脚本，没有修改CUDA代码或重建库。

建议先basic中SpeedOfLight（SM/Memory Throughput）、LaunchStats（Registers/Thread）、Occupancy（Achieved Occupancy）；再按需增加SchedulerStats、WarpStateStats。具体候选名称和来源见metric_candidates.csv，设备级query和实际采集尚未完成。basic若能成功而深入section失败，应报告部分可用，不因深度采集失败否定basic结果。

## 完整性与结束状态

失败后的资源查询显示GPU1为0MiB、0%，无本次进程残留；GPU0 VLLM继续运行。75个源码/测试/benchmark/动态库文件SHA均未变化，Git状态与检查前一致。

本轮没有生成ncu_gemm.log或metrics.csv，因为GEMM未执行、没有成功指标。初次报告与日志未丢失。没有Commit、Push、PR或进入下一轮优化，检查至此停止。
