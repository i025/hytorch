# 实际运行命令

工作目录均为本项目；精确环境变量见同名 `.command.json`。首次建环境命令为 `python3 -m venv --without-pip .venv`；该短命令未单独捕获 stdout/stderr，后续安装和测试均有完整日志。

## environment/install-pip.log.command.json

UTC: 2026-10-28T09:15:41.441966+00:00；退出码 0；记录器墙钟耗时 32.679822206962854 秒。

```bash
.venv/bin/python /home/xymo/projects/udacity-hytorch/.bootstrap/get-pip.py --no-cache-dir pip==25.0.1 setuptools==75.3.2 wheel==0.45.1
```

## environment/system-info.txt.command.json

UTC: 2026-10-28T09:15:44.815173+00:00；退出码 0；记录器墙钟耗时 3.6063213308807462 秒。

```bash
bash -c 'date -Is; hostname; cat /etc/os-release; lscpu; gcc --version; /usr/local/cuda-12.2/bin/nvcc --version; cat /usr/local/cuda-12.2/version.json; nvidia-smi; nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,driver_version,memory.total,memory.used,utilization.gpu,compute_mode --format=csv; nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv; python3 --version; id; command -v gh || true'
```

## environment/install-requirements.log.command.json

UTC: 2026-10-28T09:16:41.143000+00:00；退出码 1；记录器墙钟耗时 193.18625916005112 秒。

```bash
.venv/bin/python -m pip install --no-cache-dir -r requirements-cuda.txt cupy-cuda12x==12.3.0
```

## environment/install-requirements-02-wheels.log.command.json

UTC: 2026-10-28T09:20:55.832019+00:00；退出码 0；记录器墙钟耗时 144.16799242491834 秒。

```bash
.venv/bin/python -m pip install --cache-dir .venv/pip-cache --only-binary=:all: -r requirements-cuda.txt cupy-cuda12x==12.3.0
```

## environment/pip-check.log.command.json

UTC: 2026-10-28T09:27:05.596367+00:00；退出码 0；记录器墙钟耗时 0.3778373228851706 秒。

```bash
python -m pip check
```

## host-tests/pytest.log.command.json

UTC: 2026-10-28T09:27:05.596893+00:00；退出码 0；记录器墙钟耗时 2.507005630992353 秒。

```bash
python -m pytest tests/test_cuda_host.py tests/test_conv2d_gemm_cpu.py tests/test_gemm_blocked_packed.py -q -ra --junitxml=docs/experiments/a100-stage1/host-tests/tests.xml
```

## environment/python-packages.txt.command.json

UTC: 2026-10-28T09:27:06.022415+00:00；退出码 0；记录器墙钟耗时 0.2869850310962647 秒。

```bash
python -m pip freeze
```

## environment/python-info.txt.command.json

UTC: 2026-10-28T09:27:06.356707+00:00；退出码 0；记录器墙钟耗时 0.6199272649828345 秒。

```bash
python docs/experiments/a100-stage1/environment/python-info.py
```

## environment/gpu-availability-02.txt.command.json

UTC: 2026-10-28T09:27:45.350972+00:00；退出码 0；记录器墙钟耗时 0.1461619979236275 秒。

```bash
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv
```

## environment/gpu-processes-02.txt.command.json

UTC: 2026-10-28T09:27:45.535217+00:00；退出码 0；记录器墙钟耗时 0.12952068401500583 秒。

```bash
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
```

## environment/git-push-preflight.log.command.json

UTC: 2026-10-28T09:28:06.457236+00:00；退出码 128；记录器墙钟耗时 1.8609986410010606 秒。

```bash
git push --dry-run origin HEAD:refs/heads/experiment/a100-stage1
```

## figures/generation.log.command.json

UTC: 2026-10-28T09:28:30.648591+00:00；退出码 0；记录器墙钟耗时 0.7084245670121163 秒。

```bash
python docs/experiments/a100-stage1/make_host_figure.py
```
