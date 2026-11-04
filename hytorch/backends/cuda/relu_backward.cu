// Opt-in FP32 ReLU gradient: avoid the intermediate boolean mask and second launch.
#include <cuda_runtime.h>
#include <stdint.h>
#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C" __attribute__((visibility("default")))
#endif

__global__ void relu_backward_kernel(const float* x, const float* gy, float* gx, int size) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        gx[i] = __fmul_rn(gy[i], x[i] > 0 ? 1.f : 0.f);
}

API int hy_relu_backward(const float* x, const float* gy, float* gx, int size, void* stream) {
    relu_backward_kernel<<<(size + 255LL) / 256, 256, 0, (cudaStream_t)stream>>>(x, gy, gx, size);
    return (int)cudaGetLastError();
}
