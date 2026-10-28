# 阶段 1 完整已记录命令索引

工作目录为本项目；时间均为 UTC。GPU 验收额外环境见 `validation/audit-config.json`，其余精确环境见同名 `.command.json`。早期记录器未保存 CuPy/CUDA 缓存变量，GPU 激活脚本记录对应设置。重复运行请换新输出路径。

## environment/install-pip.log.command.json

UTC: 2026-10-28T09:15:41.441966+00:00；退出码 0；墙钟耗时 32.679822206962854 秒。

```bash
.venv/bin/python /home/xymo/projects/udacity-hytorch/.bootstrap/get-pip.py --no-cache-dir pip==25.0.1 setuptools==75.3.2 wheel==0.45.1
```

## environment/system-info.txt.command.json

UTC: 2026-10-28T09:15:44.815173+00:00；退出码 0；墙钟耗时 3.6063213308807462 秒。

```bash
bash -c 'date -Is; hostname; cat /etc/os-release; lscpu; gcc --version; /usr/local/cuda-12.2/bin/nvcc --version; cat /usr/local/cuda-12.2/version.json; nvidia-smi; nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,driver_version,memory.total,memory.used,utilization.gpu,compute_mode --format=csv; nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv; python3 --version; id; command -v gh || true'
```

## environment/install-requirements.log.command.json

UTC: 2026-10-28T09:16:41.143000+00:00；退出码 1；墙钟耗时 193.18625916005112 秒。

```bash
.venv/bin/python -m pip install --no-cache-dir -r requirements-cuda.txt cupy-cuda12x==12.3.0
```

## environment/install-requirements-02-wheels.log.command.json

UTC: 2026-10-28T09:20:55.832019+00:00；退出码 0；墙钟耗时 144.16799242491834 秒。

```bash
.venv/bin/python -m pip install --cache-dir .venv/pip-cache --only-binary=:all: -r requirements-cuda.txt cupy-cuda12x==12.3.0
```

## environment/pip-check.log.command.json

UTC: 2026-10-28T09:27:05.596367+00:00；退出码 0；墙钟耗时 0.3778373228851706 秒。

```bash
python -m pip check
```

## host-tests/pytest.log.command.json

UTC: 2026-10-28T09:27:05.596893+00:00；退出码 0；墙钟耗时 2.507005630992353 秒。

```bash
python -m pytest tests/test_cuda_host.py tests/test_conv2d_gemm_cpu.py tests/test_gemm_blocked_packed.py -q -ra --junitxml=docs/experiments/a100-stage1/host-tests/tests.xml
```

## environment/python-packages.txt.command.json

UTC: 2026-10-28T09:27:06.022415+00:00；退出码 0；墙钟耗时 0.2869850310962647 秒。

```bash
python -m pip freeze
```

## environment/python-info.txt.command.json

UTC: 2026-10-28T09:27:06.356707+00:00；退出码 0；墙钟耗时 0.6199272649828345 秒。

```bash
python docs/experiments/a100-stage1/environment/python-info.py
```

## environment/gpu-availability-02.txt.command.json

UTC: 2026-10-28T09:27:45.350972+00:00；退出码 0；墙钟耗时 0.1461619979236275 秒。

```bash
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv
```

## environment/gpu-processes-02.txt.command.json

UTC: 2026-10-28T09:27:45.535217+00:00；退出码 0；墙钟耗时 0.12952068401500583 秒。

```bash
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
```

## environment/git-push-preflight.log.command.json

UTC: 2026-10-28T09:28:06.457236+00:00；退出码 128；墙钟耗时 1.8609986410010606 秒。

```bash
git push --dry-run origin HEAD:refs/heads/experiment/a100-stage1
```

## figures/generation.log.command.json

UTC: 2026-10-28T09:28:30.648591+00:00；退出码 0；墙钟耗时 0.7084245670121163 秒。

```bash
python docs/experiments/a100-stage1/make_host_figure.py
```

## environment/gpu-availability-03.txt.command.json

UTC: 2026-10-28T15:12:10.788674+00:00；退出码 0；墙钟耗时 0.1445930921472609 秒。

```bash
nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu --format=csv
```

## environment/gpu-processes-03.txt.command.json

UTC: 2026-10-28T15:12:10.974603+00:00；退出码 0；墙钟耗时 0.13616535486653447 秒。

```bash
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
```

## build/terminal-01.log.command.json

UTC: 2026-10-28T15:12:11.149437+00:00；退出码 0；墙钟耗时 2.4032255690544844 秒。

```bash
python scripts/build_cuda.py --device 0
```

## environment/selected-gpu.txt.command.json

UTC: 2026-10-28T15:13:40.267008+00:00；退出码 0；墙钟耗时 1.3752586250193417 秒。

```bash
python docs/experiments/a100-stage1/environment/gpu-info.py
```

## build/dynamic-dependencies.log.command.json

UTC: 2026-10-28T15:13:41.681316+00:00；退出码 0；墙钟耗时 0.00959265697747469 秒。

```bash
ldd hytorch/backends/cuda/build/libhytorch_cuda.so
```

## validation/terminal-01.log.command.json

UTC: 2026-10-28T15:13:41.731802+00:00；退出码 0；墙钟耗时 6.643339903792366 秒。

```bash
python scripts/validate_cuda.py --device 0 --output runs/gpu_acceptance_01
```

## host-tests/pytest-02.log.command.json

UTC: 2026-10-28T15:14:33.016225+00:00；退出码 0；墙钟耗时 1.1152814109809697 秒。

```bash
python -m pytest tests/test_cuda_host.py tests/test_conv2d_gemm_cpu.py tests/test_gemm_blocked_packed.py -q -ra --junitxml=docs/experiments/a100-stage1/host-tests/tests-02.xml
```

## smoke/terminal-01.log.command.json

UTC: 2026-10-28T15:14:56.678784+00:00；退出码 0；墙钟耗时 4.755440400913358 秒。

```bash
python benchmarks/bench_cuda_training.py --device cuda --model tiny --samples 8 --batch 2 --matmul shared --conv im2col --pool native --no-fingerprints --output runs/gpu_smoke_01
```

## figures/generation-02.log.command.json

UTC: 2026-10-28T15:16:54.582777+00:00；退出码 0；墙钟耗时 1.5183139101136476 秒。

```bash
python docs/experiments/a100-stage1/summarize_results.py
```

## environment/gpu-after.txt.command.json

UTC: 2026-10-28T15:21:27.658992+00:00；退出码 0；墙钟耗时 0.10834824480116367 秒。

```bash
nvidia-smi --id=GPU-c6683db1-d2aa-97b9-c1b4-e497bd7ff0c5 --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv
```

## environment/gpu-after-processes.txt.command.json

UTC: 2026-10-28T15:21:58.403247+00:00；退出码 0；墙钟耗时 0.102566157002002 秒。

```bash
nvidia-smi --id=GPU-c6683db1-d2aa-97b9-c1b4-e497bd7ff0c5 --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
```

## environment/gpu-after-owners.txt.command.json

UTC: 2026-10-28T15:21:58.675328+00:00；退出码 0；墙钟耗时 0.09748811600729823 秒。

```bash
ps -o pid,user,etime,comm -p 1336263
```
