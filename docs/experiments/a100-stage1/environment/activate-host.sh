# Source from the repository root. Settings affect only this shell and children.
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=''
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_PATH=/usr/local/cuda-12.2
export PATH="$CUDA_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_PATH/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export HYTORCH_GEMM_BUILD_DIR="$PWD/.hytorch_build/a100-stage1-host"
export MPLCONFIGDIR="$PWD/.matplotlib_cache"
