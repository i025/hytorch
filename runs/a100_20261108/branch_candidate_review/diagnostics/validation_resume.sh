#!/usr/bin/env bash
set -euo pipefail
cd /home/xieminhui_pubuse/hytorch
export CUDA_VISIBLE_DEVICES=1 NVIDIA_TF32_OVERRIDE=0 HYTORCH_REQUIRE_CUDA=1
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=.
PY=/home/xieminhui_pubuse/hytorch-env/bin/python
OUT=runs/a100_20261108/branch_candidate_review
/usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 99 "$PY" -m pytest -q tests/test_relu_fusion.py > "$OUT/raw/memcheck.log" 2>&1
printf '%s\n' 'Memcheck completed'
/usr/local/cuda/bin/compute-sanitizer --tool racecheck --error-exitcode 99 "$PY" -m pytest -q tests/test_relu_fusion.py > "$OUT/raw/racecheck.log" 2>&1
printf '%s\n' 'Racecheck completed'
mkdir -p "$OUT/traces"
HYTORCH_REVIEW_ROOT=/home/xieminhui_pubuse/hytorch-before-candidate-review /opt/nvidia/nsight-systems/2025.1.3/bin/nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --output "$OUT/traces/original" "$PY" "$OUT/diagnostics/experiment.py" --trace baseline > "$OUT/raw/nsys_original.log" 2>&1
/opt/nvidia/nsight-systems/2025.1.3/bin/nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --output "$OUT/traces/candidate" "$PY" "$OUT/diagnostics/experiment.py" --trace relu_prune > "$OUT/raw/nsys_candidate.log" 2>&1
/opt/nvidia/nsight-systems/2025.1.3/bin/nsys export --type sqlite --output "$OUT/traces/original.sqlite" "$OUT/traces/original.nsys-rep" > "$OUT/raw/nsys_export_original.log" 2>&1
/opt/nvidia/nsight-systems/2025.1.3/bin/nsys export --type sqlite --output "$OUT/traces/candidate.sqlite" "$OUT/traces/candidate.nsys-rep" > "$OUT/raw/nsys_export_candidate.log" 2>&1
printf '%s\n' 'Nsight capture completed'
"$PY" benchmarks/bench_cuda_training.py --device cuda --model donkey --samples 256 --epochs 3 --batch 4 --final-config --relu-backward fused --no-input-grad --monitor --monitor-components disabled --output "$OUT/raw/cli_optimized" > "$OUT/raw/cli_optimized.log" 2>&1
printf '%s\n' 'Optimized CLI completed'
nvidia-smi > "$OUT/raw/gpu_after.txt"
