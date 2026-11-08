# 当前已执行步骤

仓库 /home/xieminhui_pubuse/hytorch，虚拟环境 /home/xieminhui_pubuse/hytorch-env。

```bash
cd /home/xieminhui_pubuse/hytorch
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 /home/xieminhui_pubuse/hytorch-env/bin/python -m pytest tests/test_cuda_host.py tests/test_col2im_host.py tests/test_conv2d_gemm_cpu.py -q
nvidia-smi
nvidia-smi --query-gpu=index,name,uuid,utilization.gpu,memory.used --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv
```

save_baseline.py 只允许首次创建 baseline，拒绝覆盖。后续恢复工作前核对 baseline/SHA.json 与生产源码；如源码变化，保存独立新基线。EXPERIMENT_PROTOCOL.md 是尚待执行的协议，并非已实现或已验证的 benchmark 脚本。
