// Experimental split-K implicit GEMM for FP32 Conv2d weight gradients.
#include <cuda_runtime.h>
#include <stdint.h>
#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C" __attribute__((visibility("default")))
#endif

__global__ void conv_dweight_tiles(const float* x, const float* gy, float* partial,
    int N, int C, int H, int W, int OC, int KH, int KW,
    int SH, int SW, int PH, int PW, int OH, int OW) {
    __shared__ float a[16][16], b[16][16];
    int p=C*KH*KW, tiles=(p+15)/16, rows=(OC+15)/16;
    int split=blockIdx.x/(tiles*rows), tile=blockIdx.x%(tiles*rows);
    int tx=threadIdx.x, ty=threadIdx.y;
    int o=tile/tiles*16+ty, j=tile%tiles*16+tx;
    int c=j/(KH*KW), kh=j/KW%KH, kw=j%KW;
    float value=0;
    for (int k=0;k<128;k+=16) {
        int ra=split*128+k+tx, rb=split*128+k+ty;
        int na=ra/(OH*OW), nb=rb/(OH*OW);
        int ih=rb/OW%OH*SH+kh-PH, iw=rb%OW*SW+kw-PW;
        a[ty][tx]=(o<OC && na<N) ? gy[((int64_t)na*OC+o)*OH*OW+ra%(OH*OW)] : 0;
        b[ty][tx]=(j<p && nb<N && ih>=0 && ih<H && iw>=0 && iw<W)
            ? x[((int64_t)nb*C+c)*H*W+ih*W+iw] : 0;
        __syncthreads();
        #pragma unroll
        for (int q=0;q<16;++q) value=__fmaf_rn(a[ty][q],b[q][tx],value);
        __syncthreads();
    }
    if (o<OC && j<p) partial[((int64_t)split*OC+o)*p+j]=value;
}

__global__ void conv_dweight_reduce(const float* partial, float* dw, int size, int splits) {
    __shared__ float sums[8][32];
    int j=blockIdx.x*32+threadIdx.x, lane=threadIdx.y;
    float value=0;
    if (j<size)
        for (int s=lane;s<splits;s+=8) value=__fadd_rn(value,partial[(int64_t)s*size+j]);
    sums[lane][threadIdx.x]=value;
    __syncthreads();
    for (int step=4;step;step/=2) {
        if (lane<step) sums[lane][threadIdx.x]=__fadd_rn(sums[lane][threadIdx.x],sums[lane+step][threadIdx.x]);
        __syncthreads();
    }
    if (!lane && j<size) dw[j]=sums[0][threadIdx.x];
}

API int hy_conv_dweight(const float* x, const float* gy, float* partial, float* dw,
    int N, int C, int H, int W, int OC, int KH, int KW,
    int SH, int SW, int PH, int PW, int OH, int OW, void* stream) {
    int64_t p=(int64_t)C*KH*KW, r=(int64_t)N*OH*OW;
    int splits=(r+127)/128;
    conv_dweight_tiles<<<splits*((p+15)/16)*((OC+15)/16),dim3(16,16),0,(cudaStream_t)stream>>>
        (x,gy,partial,N,C,H,W,OC,KH,KW,SH,SW,PH,PW,OH,OW);
    int status=(int)cudaGetLastError();
    if (status) return status;
    conv_dweight_reduce<<<(OC*p+31)/32,dim3(32,8),0,(cudaStream_t)stream>>>(partial,dw,OC*p,splits);
    return (int)cudaGetLastError();
}
