# Source from the repository root. Select an idle GPU UUID separately before GPU work.
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=''
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_PATH=/usr/local/cuda-12.2
export PATH="$CUDA_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_PATH/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export CUPY_TF32=0
export NVIDIA_TF32_OVERRIDE=0
export CUPY_CACHE_DIR="$PWD/.venv/cupy-cache"
export CUDA_CACHE_PATH="$PWD/.venv/cuda-cache"
export HYTORCH_GEMM_BUILD_DIR="$PWD/.hytorch_build/a100-stage2a-host"
export MPLCONFIGDIR="$PWD/.matplotlib_cache"
