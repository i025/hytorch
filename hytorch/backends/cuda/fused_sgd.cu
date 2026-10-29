// Opt-in FP32 multi-tensor SGD. No device allocations, explicit transfers, or sync.
#include <cuda_runtime.h>
#include <stdint.h>
#include <limits.h>
#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C" __attribute__((visibility("default")))
#endif

namespace {
constexpr int kMaxTensors = 64;
constexpr int kBlockSize = 256;

struct SgdTensor {
    float* parameter;
    const float* gradient;
    int size;
    int end_block;
};

struct SgdMetadata {
    SgdTensor tensors[kMaxTensors];
    int count;
    float learning_rate;
};
static_assert(sizeof(SgdMetadata) <= 4096, "SGD metadata must fit kernel arguments");
}

__global__ void multi_tensor_sgd_kernel(SgdMetadata metadata) {
    int tensor = 0;
    while (tensor < metadata.count && blockIdx.x >= metadata.tensors[tensor].end_block)
        ++tensor;
    if (tensor == metadata.count) return;
    const SgdTensor entry = metadata.tensors[tensor];
    const int first_block = tensor ? metadata.tensors[tensor - 1].end_block : 0;
    const int64_t index = (int64_t)(blockIdx.x - first_block) * blockDim.x + threadIdx.x;
    if (index < entry.size) {
        // Match the baseline's separate FP32 multiply and subtract roundings.
        entry.parameter[index] = __fsub_rn(
            entry.parameter[index], __fmul_rn(metadata.learning_rate, entry.gradient[index]));
    }
}

API int hy_multi_tensor_sgd(const uint64_t* parameters, const uint64_t* gradients,
                            const int64_t* sizes, int count, float learning_rate,
                            void* stream) {
    if (count < 0 || count > kMaxTensors ||
        (count && (!parameters || !gradients || !sizes)))
        return (int)cudaErrorInvalidValue;
    if (!count) return (int)cudaSuccess;
    SgdMetadata metadata = {};
    metadata.count = count;
    metadata.learning_rate = learning_rate;
    int blocks = 0;
    for (int i = 0; i < count; ++i) {
        if (!parameters[i] || !gradients[i] || sizes[i] <= 0 || sizes[i] > INT_MAX)
            return (int)cudaErrorInvalidValue;
        blocks += (sizes[i] + kBlockSize - 1) / kBlockSize;
        metadata.tensors[i] = {reinterpret_cast<float*>(parameters[i]),
                              reinterpret_cast<const float*>(gradients[i]),
                              (int)sizes[i], blocks};
    }
    // The launch copies this POD into kernel parameter memory. Host arrays and
    // metadata can expire after this call; parameter/gradient storage stays live
    // under the same stream/lifetime contract as other asynchronous CuPy work.
    multi_tensor_sgd_kernel<<<blocks, kBlockSize, 0, (cudaStream_t)stream>>>(metadata);
    return (int)cudaGetLastError();
}
