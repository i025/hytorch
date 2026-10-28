# Source from the repository root; only physical GPU 1 is visible as logical 0.
source docs/experiments/a100-stage1/environment/activate-host.sh
export CUDA_VISIBLE_DEVICES=GPU-c6683db1-d2aa-97b9-c1b4-e497bd7ff0c5
export CUPY_CACHE_DIR="$PWD/.venv/cupy-cache"
export CUDA_CACHE_PATH="$PWD/.venv/cuda-cache"
