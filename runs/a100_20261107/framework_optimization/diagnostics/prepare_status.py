from pathlib import Path
import json,hashlib,datetime,tarfile
R=Path(__file__).resolve().parents[4];O=R/'runs/a100_20261107/framework_optimization'
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
status={'status':'blocked_before_gpu_experiments','recorded_at_Asia_Shanghai':now,'host_tests':'19 passed (three host test modules only)','GPU_tests':'NOT_RUN','stage_A':'baseline_saved; GPU measurement NOT_RUN','stage_B':'NOT_RUN: requires stage A evidence','production_changes':False,'GPU1_pid':1290426,'GPU1_observed_utilization_percent':[0,29,38],'GPU1_last_observed_memory_MiB':3445,'GPU0':'VLLM PID 1115402; untouched'}
(O/'STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2))
(O/'REPORT.md').write_text('''# 本轮准备记录：尚未完成实验

当前没有空闲且授权的 A100，阶段 A GPU 对比未运行，阶段 B 尚未实施。此包是准备与阻塞记录，不是最终实验交付。历史 0.820923s / 0.307929s 不作为本轮基线或加速比分母。

## 已完成

保存 HEAD、Git 状态、全部 tracked 未提交 diff、untracked 清单，以及 82 个源码/测试/构建文件与 SHA256。baseline/source 包含实际 CUDA 动态库与编译记录。版本来自当前虚拟环境的包元数据；当前 cuDNN runtime 版本待空闲 GPU 窗口采集。

执行 tests/test_cuda_host.py、tests/test_col2im_host.py、tests/test_conv2d_gemm_cpu.py：19 passed。仅代表这三个文件，不代表完整 host/GPU 回归通过。

未修改生产源码、数值门限、默认配置或已有实验记录；没有提交、推送或创建 PR。

## 资源阻塞

只读查询发现 GPU1 上 PID1290426 正在运行 GRCache extract_v9.py --panel panel_90_150。连续查询显存从2419MiB增长至3463MiB，最后3445MiB，GPU利用率最高观察到38%。即使瞬间利用率为0%，也不把已占用设备当作空闲。
GPU0 为 VLLM PID1115402，占用20543MiB。未终止、暂停或影响这些进程，没有启动本任务 GPU 工作负载。

## 源码检查所得候选（未经性能验证）

- hytorch/core.py Variable.backward：每次 add_func 排序，以及逐节点上下文管理器；必须保持 generation 和同代 tie 顺序，不能跳过梯度节点。待 cProfile 与无 profiler 完整 backward 配对测量。
- hytorch/core.py Function.__call__：输入包装、输出 Variable 与 weakref 创建。no_grad 下仍记录 inputs/outputs；自定义 Function 可能依赖这些成员，不能盲删。
- hytorch/backends/cuda/runtime.py current_config 返回 dict 副本；selected、布局校验与 ctypes 参数构造需要分项计时。任何缓存不得保存指针或 stream。
- hytorch/dataloaders.py 同步路径逐样本组装 NumPy batch，再由训练循环 H2D；尚无本轮证据证明它占主要耗时。预加载不能只施于优化组。

这些候选没有按实际训练收益排序，尚不满足生产优化的证据门槛。

## 未运行

实际 workload instrumentation、15次逐算子对比、Nsight 时间线、候选优化、完整回归、Sanitizer、10轮训练消融、PyTorch5轮参考、TinyNet/batch补充、显存实验和 PPT 图表均未运行。没有生成伪造/空白性能CSV来充当交付。

待 GPU1 空闲后继续同一新窗口采集 B0 和所有候选数据，不能拼接历史窗口计算配对收益。阶段A测量冻结之后再决定阶段B的1–2项修改。
''')
(O/'EXPERIMENT_PROTOCOL.md').write_text('''# 待执行协议

B0 使用 shared/im2col/native pool/native col2im/packed dweight；SGD legacy。维持已有数值门限，不启用被拒绝的融合 SGD。

固定 seed42、256合成样本、batch4、3epochs/192steps、dropout0、SGD0.001。TF32禁用，PyTorch正常cuDNN/autograd，无compile。双方同样按step传输和保留GPU loss，在最终同步后读取轨迹。

阶段A：从真实step捕获所有Conv/Linear输入、输出、梯度布局、strides、参数、实际GEMM调用和次数；加入通用不规则形状。每组至少15次平衡交替、预热、固定输入。区分CUDA event区间（含可能的提交间隙）与CUPTI kernel累计时间。完整Python提交、同步完成、分配、copy、误差独立记录。卷积dInput/dWeight/dBias/full backward与等价Torch运算分别比。

CPU cProfile独立运行，不用其耗时做正式性能。Nsight短trace分别捕获两个框架，记录实际kernel/launch/API/NVTX/H2D/D2H；区分活动时间并集和kernel累计时间，CPU/GPU重叠不相加。按真实调用频率报告热点。

冻结阶段A数据与SHA后，按实测影响、可消除程度和风险选择1–2候选。每项有默认关闭开关，不改变梯度累积顺序/高阶语义，无跨step数值缓存。先专项正确性再计时；无收益的复杂改动不保留。

正式训练各有效配置至少10轮平衡交替；Torch至少5轮同窗口参考。报告逐对差、mean/median/sample SD/95%配对t区间。补充batch1/16和TinyNet。最终权重/梯度/loss使用既有门限比较。分析内存池指标时说明未包含库内分配；Nsight/CUDA API数据和无profiler正式训练分离。

完整host/GPU回归：不得将GPU skip计为pass。针对涉及ABI/内存路径执行适用memcheck/racecheck。确认资源状态后才开始GPU窗口；整个窗口定期记录共享任务，发生干扰不混成正式配对数据。
''')
(O/'REPRODUCE.md').write_text('''# 当前已执行步骤

仓库 /home/xieminhui_pubuse/hytorch，虚拟环境 /home/xieminhui_pubuse/hytorch-env。

```bash
cd /home/xieminhui_pubuse/hytorch
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 /home/xieminhui_pubuse/hytorch-env/bin/python -m pytest tests/test_cuda_host.py tests/test_col2im_host.py tests/test_conv2d_gemm_cpu.py -q
nvidia-smi
nvidia-smi --query-gpu=index,name,uuid,utilization.gpu,memory.used --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv
```

save_baseline.py 只允许首次创建 baseline，拒绝覆盖。后续恢复工作前核对 baseline/SHA.json 与生产源码；如源码变化，保存独立新基线。EXPERIMENT_PROTOCOL.md 是尚待执行的协议，并非已实现或已验证的 benchmark 脚本。
''')
unchanged=[]
for row in json.loads((O/'baseline/SHA.json').read_text()):
 if hashlib.sha256((R/row['path']).read_bytes()).hexdigest()!=row['sha256']:unchanged.append(row['path'])
assert not unchanged,unchanged
(O/'raw/production_unchanged.json').write_text(json.dumps({'all_82_files_unchanged':True,'checked_at':now},indent=2))
with tarfile.open(O.parent/'framework_optimization_preparation.tar.gz','w:gz') as t:t.add(O,arcname=O.name)
print(status)
