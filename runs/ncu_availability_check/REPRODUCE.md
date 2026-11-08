# 复现命令

首次只执行静态检查；续检已执行一次basic，退出9，错误为InterprocessLockFailed。原始执行命令在recheck_20261108T082100Z/basic.json，真实日志见ncu_basic.log。以下GEMM/扩展步骤仍未执行。初次NOT_RUN记录保存在initial_static/。

## 静态检查（已执行）

```bash
cd /home/xieminhui_pubuse/hytorch
which ncu
ncu --version
which nvcc
nvcc --version
/opt/nvidia/nsight-compute/2025.2.1/ncu --version
/usr/local/cuda/bin/nvcc --version
nvidia-smi
nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total,driver_version,compute_mode --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
cat /proc/driver/nvidia/params | grep -i RmProfilingAdminOnly
id
/opt/nvidia/nsight-compute/2025.2.1/ncu --list-sets
/opt/nvidia/nsight-compute/2025.2.1/ncu --list-sections
```

CLI不在PATH时用绝对路径即可，无需修改系统配置。完整执行结果见environment.txt、permissions.txt和static_commands.json。

## 安全前提

每次profiling前重新查询GPU1，确认无计算进程、无占用任务，且有使用权限。即使利用率暂时0%，已有VLLM/Python进程也不能视为空闲。GPU1仍占用则停止；不自动改用GPU0，不终止任何进程。下述CUDA_VISIBLE_DEVICES=1仅适用于确认空闲后的物理GPU1。

## 最小basic（已尝试并失败）

在bash中运行并单独保留退出码：

```bash
NCU=/opt/nvidia/nsight-compute/2025.2.1/ncu
PY=/home/xieminhui_pubuse/hytorch-env/bin/python
NCU_OUT=runs/ncu_availability_check/recheck_$(date -u +%Y%m%dT%H%M%SZ)
mkdir "$NCU_OUT"
env CUDA_VISIBLE_DEVICES=1 NVIDIA_TF32_OVERRIDE=0 \
  "$NCU" --set basic --launch-count 1 \
  --clock-control none --pipeline-boost-state dynamic --cache-control none \
  --export "$NCU_OUT/torch_basic" \
  "$PY" -c 'import torch; a=torch.ones((512,512),device="cuda"); b=a+a; torch.cuda.synchronize(); print(b[0,0].item())' \
  > "$NCU_OUT/ncu_basic.log" 2>&1
NCU_RC=$?
printf '%s\n' "$NCU_RC" > "$NCU_OUT/ncu_basic.exit_code"
```

launch-count=1可能采集首次ones填充kernel，而非后面的加法，二者都是足够小的可用性探针；应以实际报告kernel名称为准。NCU可能为一个kernel进行多pass重放，launch-count不是硬件重放次数上限。

只有确认退出码、日志及报告有效后才离线导出：

```bash
"$NCU" --import "$NCU_OUT/torch_basic.ncu-rep" --page raw --csv > "$NCU_OUT/torch_metrics.csv"
```

必须看到实际kernel及有效指标值才判为basic成功。打印2.0、出现文件或退出0均不能单独证明采到计数器。

## 自研GEMM（仅在basic确认成功后，未执行）

再次确认GPU1空闲。现有库无需重建；脚本只执行一次1024³ GEMM，kernel过滤排除初始化/验证：

```bash
env CUDA_VISIBLE_DEVICES=1 NVIDIA_TF32_OVERRIDE=0 \
  LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} \
  "$NCU" --set basic --launch-count 1 \
  --clock-control none --pipeline-boost-state dynamic --cache-control none \
  --kernel-name-base demangled --kernel-name 'regex:.*gemm_register.*' \
  --export "$NCU_OUT/gemm_basic" \
  "$PY" runs/ncu_availability_check/gemm_minimal.py \
  > "$NCU_OUT/ncu_gemm.log" 2>&1
NCU_RC=$?
printf '%s\n' "$NCU_RC" > "$NCU_OUT/ncu_gemm.exit_code"
```

若GEMM basic也成功，只追加一次小范围section采集（事先再次检查资源）：

```bash
env CUDA_VISIBLE_DEVICES=1 NVIDIA_TF32_OVERRIDE=0 \
  LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} \
  "$NCU" --section SpeedOfLight --section LaunchStats --section Occupancy \
  --section SchedulerStats --section WarpStateStats --launch-count 1 \
  --clock-control none --pipeline-boost-state dynamic --cache-control none \
  --kernel-name-base demangled --kernel-name 'regex:.*gemm_register.*' \
  --export "$NCU_OUT/gemm_sections" \
  "$PY" runs/ncu_availability_check/gemm_minimal.py \
  > "$NCU_OUT/ncu_gemm_sections.log" 2>&1
NCU_RC=$?
printf '%s\n' "$NCU_RC" > "$NCU_OUT/ncu_gemm_sections.exit_code"
```

section名称已通过本机--list-sections确认。精确metric候选来自安装包文件；若要指定metric，先在有安全设备的窗口使用--query-metrics确认设备支持和suffix。basic成功、扩展失败时保留两份结果并标为部分可用；不要全归因权限。

权限失败时只保存错误并请管理员核查，不使用sudo/setcap、不改驱动参数、不重载或重启。所有env赋值仅作用于当次子进程。

## 锁失败后的处理边界

本轮实际执行入口：

```bash
/home/xieminhui_pubuse/hytorch-env/bin/python runs/ncu_availability_check/run_minimal_check.py basic
```

该入口每次创建独立recheck目录，并拒绝GPU1已有计算进程、显存或利用率非零的情况。原始NCU退出码9保存在basic.json；诊断runner本身的完成码不是profiling成功判据。不要直接运行gemm/sections模式：必须先人工确认basic有效指标。

失败后只执行了stat/namei/getfacl/os.access及只读资源检查，没有打开写入共享锁。`fs.protected_regular`读取被拒绝，值未知。当前应先解决锁访问阻断，而非调整Performance Counter权限。

安装版官方FAQ给出的无需系统配置修改方案是：在**确认整台服务器没有并发NCU实例**后，只对本次命令设置TMPDIR为用户可写目录。单次进程快照并不提供排他保证，应先协调测试窗口；不要未经协调换锁目录，亦不要删除共享锁。此替代方案本轮未执行。
