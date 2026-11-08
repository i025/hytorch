"""Apply precisely the requested heuristic after baseline GPU tests pass."""
from pathlib import Path
p=Path('hytorch/backends/cuda/kernels.cu');s=p.read_text()
old='''    cudaStream_t s=(cudaStream_t)stream;
    if (variant==0) gemm_basic'''
new='''    // Conservative shape-only heuristic; 128 CTAs is a tuning starting point.
    const int64_t grid_m = ((int64_t)m + 63) / 64;
    const int64_t grid_n = ((int64_t)n + 63) / 64;
    const int64_t register_blocks = grid_m * grid_n;
    constexpr int64_t kMinRegisterBlocks = 128;
    const bool use_register = variant == 2 && m >= 64 && n >= 64 && k >= 64 &&
                              register_blocks >= kMinRegisterBlocks;
    cudaStream_t s=(cudaStream_t)stream;
    if (variant==0) gemm_basic'''
assert s.count(old)==1 and s.count('else if (variant==2 && m>=64 && n>=64 && k>=64)')==1
s=s.replace(old,new).replace('else if (variant==2 && m>=64 && n>=64 && k>=64)','else if (use_register)')
p.write_text(s)
