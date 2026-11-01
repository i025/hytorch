// Experimental gather with four independent FP32 channel accumulation chains.
#include <cuda_runtime.h>
#include <stdint.h>
#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C" __attribute__((visibility("default")))
#endif

__global__ void conv_dinput_gather_opt(const float* gy, const float* weight, float* gx,
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
            float p0=0, p1=0, p2=0, p3=0;
            int oc=0;
            for (;oc+3<OC;oc+=4) {
                p0=__fmaf_rn(gy[((int64_t)n*OC+oc)*OH*OW+oh*OW+ow],
                    weight[((int64_t)oc*C+c)*KH*KW+kh*KW+kw],p0);
                p1=__fmaf_rn(gy[((int64_t)n*OC+oc+1)*OH*OW+oh*OW+ow],
                    weight[((int64_t)(oc+1)*C+c)*KH*KW+kh*KW+kw],p1);
                p2=__fmaf_rn(gy[((int64_t)n*OC+oc+2)*OH*OW+oh*OW+ow],
                    weight[((int64_t)(oc+2)*C+c)*KH*KW+kh*KW+kw],p2);
                p3=__fmaf_rn(gy[((int64_t)n*OC+oc+3)*OH*OW+oh*OW+ow],
                    weight[((int64_t)(oc+3)*C+c)*KH*KW+kh*KW+kw],p3);
            }
            float partial=__fadd_rn(__fadd_rn(p0,p1),__fadd_rn(p2,p3));
            for (;oc<OC;++oc)
                partial=__fmaf_rn(gy[((int64_t)n*OC+oc)*OH*OW+oh*OW+ow],
                    weight[((int64_t)oc*C+c)*KH*KW+kh*KW+kw],partial);
            value=__fadd_rn(value,partial);
        }
    }
    gx[i]=value;
}

API int hy_conv_dinput_opt(const float* gy, const float* weight, float* gx,
    int N, int C, int H, int W, int OC, int KH, int KW,
    int SH, int SW, int PH, int PW, int OH, int OW, void* stream) {
    conv_dinput_gather_opt<<<((int64_t)N*C*H*W+255)/256,256,0,(cudaStream_t)stream>>>
        (gy,weight,gx,N,C,H,W,OC,KH,KW,SH,SW,PH,PW,OH,OW);
    return (int)cudaGetLastError();
}
