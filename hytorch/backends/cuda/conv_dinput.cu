// Experimental FP32 Conv2d input gradient: one gather per NCHW output element.
#include <cuda_runtime.h>
#include <stdint.h>
#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C" __attribute__((visibility("default")))
#endif

__global__ void conv_dinput_gather(const float* gy, const float* weight, float* gx,
    int N, int C, int H, int W, int OC, int KH, int KW,
    int SH, int SW, int PH, int PW, int OH, int OW) {
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=(int64_t)N*C*H*W) return;
    int w=i%W, h=i/W%H, c=i/(W*H)%C, n=i/(W*H*C);
    float value=0;
    for (int kh=0;kh<KH;++kh) {
        int oh=h+PH-kh;
        if (oh<0 || oh>=OH*SH || oh%SH) continue;
        oh/=SH;
        for (int kw=0;kw<KW;++kw) {
            int ow=w+PW-kw;
            if (ow<0 || ow>=OW*SW || ow%SW) continue;
            ow/=SW;
            // Keep the channel dot and spatial sum in FP32. No gcol storage,
            // global atomics, fast-math, tensor cores, or TF32 operations.
            float partial=0;
            for (int oc=0;oc<OC;++oc)
                partial=__fmaf_rn(gy[((int64_t)n*OC+oc)*OH*OW+oh*OW+ow],
                    weight[((int64_t)oc*C+c)*KH*KW+kh*KW+kw],partial);
            value=__fadd_rn(value,partial);
        }
    }
    gx[i]=value;
}

API int hy_conv_dinput(const float* gy, const float* weight, float* gx,
    int N, int C, int H, int W, int OC, int KH, int KW,
    int SH, int SW, int PH, int PW, int OH, int OW, void* stream) {
    conv_dinput_gather<<<((int64_t)N*C*H*W+255)/256,256,0,(cudaStream_t)stream>>>
        (gy,weight,gx,N,C,H,W,OC,KH,KW,SH,SW,PH,PW,OH,OW);
    return (int)cudaGetLastError();
}
