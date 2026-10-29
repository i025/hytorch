// Standalone experiment: row-major FP32 GEMM, 1 <= K <= 4096.
// No production ABI or default dispatch changes.
#include <cuda_runtime.h>
#include <stdint.h>

__global__ __launch_bounds__(1024)
void gemm_block32_tree(const float* a, const float* b, float* c,
                       int m, int n, int k) {
    __shared__ float sa[32][32], sb[32][32];
    const int r = blockIdx.y * 32 + threadIdx.y;
    const int col = blockIdx.x * 32 + threadIdx.x;
    float levels[8];  // One live subtree per set bit of the processed block count.
    const int blocks = (k + 31) / 32;
    for (int block = 0; block < blocks; ++block) {
        const int base = block * 32;
        sa[threadIdx.y][threadIdx.x] = (r < m && base + threadIdx.x < k)
            ? a[(int64_t)r * k + base + threadIdx.x] : 0.0f;
        sb[threadIdx.y][threadIdx.x] = (base + threadIdx.y < k && col < n)
            ? b[(int64_t)(base + threadIdx.y) * n + col] : 0.0f;
        __syncthreads();
        float partial = 0.0f;
        #pragma unroll
        for (int q = 0; q < 32; ++q) {
            if (base + q < k)
                partial = __fmaf_rn(sa[threadIdx.y][q], sb[q][threadIdx.x], partial);
        }
        __syncthreads();
        // Binary carry: older left subtree + newer right subtree.
        // Unrolling makes level indices compile-time constants.
        #pragma unroll
        for (int level = 0; level < 8; ++level) {
            if (!(block & (1 << level))) {
                levels[level] = partial;
                break;
            }
            partial = __fadd_rn(levels[level], partial);
        }
    }
    // Fold occupied subtrees from right to left (low level to high level).
    // This reproduces adjacent pairwise reduction with an unpaired tail carried.
    float total = 0.0f;
    bool have_total = false;
    #pragma unroll
    for (int level = 0; level < 8; ++level) {
        if (blocks & (1 << level)) {
            total = have_total ? __fadd_rn(levels[level], total) : levels[level];
            have_total = true;
        }
    }
    if (r < m && col < n) c[(int64_t)r * n + col] = total;
}

extern "C" int block32_tree_gemm(const float* a, const float* b, float* c,
                                  int m, int n, int k, void* stream) {
    if (m <= 0 || n <= 0 || k <= 0 || k > 4096) return cudaErrorInvalidValue;
    gemm_block32_tree<<<dim3((n + 31) / 32, (m + 31) / 32), dim3(32, 32), 0,
                         (cudaStream_t)stream>>>(a, b, c, m, n, k);
    return cudaGetLastError();
}

// Static attributes and the runtime occupancy calculator, NOT achieved occupancy.
// A null symbol queries the experimental kernel; a host stub queries a baseline.
extern "C" int block32_tree_resources(const void* symbol, int* values) {
    const void* kernel = symbol ? symbol : (const void*)gemm_block32_tree;
    cudaFuncAttributes attr;
    cudaError_t code = cudaFuncGetAttributes(&attr, kernel);
    if (code != cudaSuccess) return code;
    int active_blocks;
    code = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks, kernel, 1024, 0);
    if (code != cudaSuccess) return code;
    values[0] = attr.numRegs;
    values[1] = (int)attr.sharedSizeBytes;
    values[2] = (int)attr.localSizeBytes;
    values[3] = attr.maxThreadsPerBlock;
    values[4] = active_blocks;
    values[5] = attr.binaryVersion;
    return cudaSuccess;
}
