// One generic FP32 Sum primitive, reused for both partial and final reductions.
#include <cuda_runtime.h>
#include <stdint.h>
#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C" __attribute__((visibility("default")))
#endif

struct SumLayout {
    int out_ndim, reduce_ndim, outputs, reduction;
    int out_shape[8], out_stride[8], reduce_shape[8], reduce_stride[8];
};
static_assert(sizeof(SumLayout) == 144, "Sum descriptor ABI mismatch");

__device__ int offset(int index, int ndim, const int* shape, const int* stride) {
    int result = 0;
    for (int d = ndim - 1; d >= 0; --d) {
        result += (index % shape[d]) * stride[d];
        index /= shape[d];
    }
    return result;
}

__device__ float warp_sum(float value) {
    for (int delta = 16; delta; delta /= 2)
        value = __fadd_rn(value, __shfl_down_sync(0xffffffff, value, delta));
    return value;
}

__global__ void reduce_sum_kernel(const float* x, float* y, SumLayout layout, int parts) {
    if (!layout.reduce_ndim) {
        for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
             i < layout.outputs; i += (int64_t)blockDim.x * gridDim.x)
            y[i] = __fadd_rn(0.f, x[i]);
        return;
    }
    __shared__ float warps[8];
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    for (int64_t job = blockIdx.x; job < (int64_t)layout.outputs * parts; job += gridDim.x) {
        int output = job / parts, part = job % parts;
        int64_t begin = (int64_t)part * 4096;
        int64_t end = parts == 1 ? layout.reduction : min(begin + 4096, (int64_t)layout.reduction);
        int base = offset(output, layout.out_ndim, layout.out_shape, layout.out_stride);
        float value = 0.f;
        for (int64_t r = (int64_t)begin + threadIdx.x; r < end; r += blockDim.x)
            value = __fadd_rn(value, x[base + offset((int)r, layout.reduce_ndim,
                                                   layout.reduce_shape, layout.reduce_stride)]);
        value = warp_sum(value);
        if (!lane) warps[warp] = value;
        __syncthreads();
        if (!warp) {
            value = lane < 8 ? warps[lane] : 0.f;
            value = warp_sum(value);
            if (!lane) y[job] = value;
        }
        __syncthreads();
    }
}

API int hy_reduce_sum(const float* x, float* y, float* workspace,
                      SumLayout layout, int parts, void* stream) {
    int64_t jobs = layout.reduce_ndim ? (int64_t)layout.outputs * parts : (layout.outputs + 255LL) / 256;
    int grid = (int)(jobs < 65535 ? jobs : 65535);
    reduce_sum_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>(x, parts > 1 ? workspace : y, layout, parts);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess || parts == 1) return (int)status;
    SumLayout final_layout = {};
    final_layout.out_ndim = final_layout.reduce_ndim = 1;
    final_layout.outputs = final_layout.out_shape[0] = layout.outputs;
    final_layout.reduction = final_layout.reduce_shape[0] = parts;
    final_layout.out_stride[0] = parts;
    final_layout.reduce_stride[0] = 1;
    grid = layout.outputs < 65535 ? layout.outputs : 65535;
    reduce_sum_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>(workspace, y, final_layout, 1);
    return (int)cudaGetLastError();
}
