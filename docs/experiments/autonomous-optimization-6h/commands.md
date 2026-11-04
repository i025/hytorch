# Commands and provenance

Run from `/home/xymo/projects/hytorch`. Existing isolated environment only:

```bash
source docs/experiments/a100-stage2a-gemm/environment/activate.sh
```

This starts CPU-only (`CUDA_VISIBLE_DEVICES=''`). Each `run_logged.py` command
preserves full stdout/stderr and exact argv, timestamps, exit code, selected environment
in its adjacent `.command.json`. Unique output paths refuse overwrites.

## Baseline and CPU preparation

```bash
git status --short --branch
git fetch origin experiment/real-udacity-training
git log -1 --format='%H %s' origin/experiment/real-udacity-training
git switch -c experiment/hytorch-autonomous-optimization
python docs/experiments/a100-stage1/run_logged.py --log runs/autonomous_optimization_01/host-baseline.log -- python -m pytest tests/test_cuda_host.py tests/test_conv2d_gemm_cpu.py tests/test_gemm_blocked_packed.py tests/test_fused_sgd_host.py tests/test_conv_dinput_pruning.py tests/test_conv_dweight_host.py tests/test_conv_dinput_opt_host.py tests/test_udacity_dataset.py tests/test_udacity_experiment.py -q -ra
python docs/experiments/a100-stage1/run_logged.py --log runs/autonomous_optimization_01/cpu-preparation.log -- python docs/experiments/autonomous-optimization-6h/scripts/cpu_prepare.py --prepared runs/udacity_real_prepared_01 --output runs/autonomous_optimization_01/cpu-preparation
python docs/experiments/a100-stage1/run_logged.py --log runs/autonomous_optimization_01/build-prototypes.log -- python scripts/build_cuda.py --arch 80 --resource-usage --output runs/autonomous_optimization_01/build-prototypes
python docs/experiments/a100-stage1/run_logged.py --log runs/autonomous_optimization_01/cpu-state-diagnosis.log -- python docs/experiments/autonomous-optimization-6h/scripts/cpu_state_diagnosis.py --prepared runs/udacity_real_prepared_01 --output runs/autonomous_optimization_01/cpu-state-diagnosis
python docs/experiments/a100-stage1/run_logged.py --log runs/autonomous_optimization_01/cpu-trajectory.log -- python docs/experiments/autonomous-optimization-6h/scripts/cpu_trajectory.py --prepared runs/udacity_real_prepared_01 --cache runs/autonomous_optimization_01/cpu-preparation/epoch-1-inputs.npz --output runs/autonomous_optimization_01/cpu-trajectory
python docs/experiments/a100-stage1/run_logged.py --log runs/autonomous_optimization_01/sgd-rounding.log -- python docs/experiments/autonomous-optimization-6h/scripts/diagnose_sgd_rounding.py --output runs/autonomous_optimization_01/sgd-rounding
python docs/experiments/a100-stage1/run_logged.py --log runs/autonomous_optimization_01/quality-analysis.log -- python docs/experiments/autonomous-optimization-6h/scripts/analyze_quality.py --output runs/autonomous_optimization_01/quality-analysis
```

Final CPU collection (`host-final-02.log`:103 passed,0 failed,0 skipped):

```bash
python -m pytest tests/test_cuda_host.py tests/test_conv2d_gemm_cpu.py tests/test_gemm_blocked_packed.py tests/test_fused_sgd_host.py tests/test_conv_dinput_pruning.py tests/test_conv_dweight_host.py tests/test_conv_dinput_opt_host.py tests/test_udacity_dataset.py tests/test_udacity_experiment.py tests/test_dataloader_indices.py tests/test_relu_backward_host.py tests/test_conv_backward_fusion_host.py tests/test_conv_shape_dispatch.py -q -ra
python -m compileall -q hytorch/backends/cuda/runtime.py hytorch/functions.py hytorch/dataloaders.py docs/experiments/autonomous-optimization-6h/scripts
```

Environment commands `uname -a`, `lscpu`, `gcc --version`, `nvcc --version`,
`nsys --version`, package versions and `python -m pip list --format=freeze` are
retained in `environment.txt` and `python-packages.txt` in the raw root.

## GPU preparation and monitoring

The first `gpu-controller.log.command.json` records the interrupted observed-idle
attempt and full argv. First window: `runs/autonomous_optimization_01/gpu-window/`.
Do not reuse/overwrite that path.

The fixed-device retry (no CUDA context while waiting):

```bash
python docs/experiments/a100-stage1/run_logged.py --log runs/autonomous_optimization_01/gpu-controller-retry.log -- python docs/experiments/autonomous-optimization-6h/scripts/wait_for_gpu.py --uuid GPU-f099fdbd-adf2-6157-3462-4145bc2a8359 --stable-seconds 90 --log runs/autonomous_optimization_01/availability-retry.jsonl -- python docs/experiments/autonomous-optimization-6h/scripts/run_window.py --prepared runs/udacity_real_prepared_01 --cache runs/autonomous_optimization_01/cpu-preparation/epoch-1-inputs.npz --library runs/autonomous_optimization_01/build-prototypes/libhytorch_cuda.so --output runs/autonomous_optimization_01/gpu-window-02
```

The controller writes every actual child command/env UUID/time/exit to `commands.json`,
then runs CUDA numerical replay, explicit manual numerical review, correctness,
operator microbenchmarks, matched A/F/I/P/R/J/E DonkeyNet/SpatialNet compute/I/O/phases
and independent Nsight. A uses a deliberately nonexistent native library path.
The CPU controller never holds CUDA memory, does not Reset, and ends only its own
processes on external resource conflict. The current prepared plan is generated by
`run_window.py --plan UNIQUE.json` without touching CUDA.

GPU collection-only log lists738 tests; collection is not GPU execution or PASS.

## CPU analysis and archival

```bash
python docs/experiments/autonomous-optimization-6h/scripts/plot_cpu.py --raw runs/autonomous_optimization_01 --output docs/experiments/autonomous-optimization-6h/figures/cpu-01
python docs/experiments/fused-sgd-cuda/scripts/archive_results.py --destination docs/experiments/autonomous-optimization-6h/results/cpu-raw-01 runs/autonomous_optimization_01/cpu-preparation runs/autonomous_optimization_01/cpu-state-diagnosis runs/autonomous_optimization_01/cpu-trajectory runs/autonomous_optimization_01/sgd-rounding runs/autonomous_optimization_01/quality-analysis runs/autonomous_optimization_01/build-prototypes
```

No hash/fingerprint procedure is used. Git commit IDs record versions, not a file
integrity verification scheme. Large inputs, arrays and traces stay server-only.
