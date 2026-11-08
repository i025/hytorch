// FP32, row-major kernels. No allocations, transfers, or synchronization in ABI.
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>
#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C" __attribute__((visibility("default")))
#endif

__global__ void gemm_basic(const float* a, const float* b, float* c,
                           int m, int n, int k) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)m*n) return;
    int r = i/n, col = i%n;
    float v = 0;
    for (int q=0; q<k; ++q) v += a[(int64_t)r*k+q]*b[(int64_t)q*n+col];
    c[i] = v;
}

// Tile the output and K loop, but keep loads in global memory for the ablation.
template<int T> __global__ void gemm_blocked(const float* a, const float* b,
                                            float* c, int m, int n, int k) {
    int r=blockIdx.y*T+threadIdx.y, col=blockIdx.x*T+threadIdx.x;
    if (r>=m || col>=n) return;
    float v=0;
    for (int base=0; base<k; base+=T)
        for (int q=0; q<T && base+q<k; ++q)
            v += a[(int64_t)r*k+base+q]*b[(int64_t)(base+q)*n+col];
    c[(int64_t)r*n+col]=v;
}

template<int T> __global__ void gemm_shared(const float* a, const float* b,
                                           float* c, int m, int n, int k) {
    __shared__ float sa[T][T], sb[T][T];
    int r=blockIdx.y*T+threadIdx.y, col=blockIdx.x*T+threadIdx.x;
    float v=0;
    for (int base=0; base<k; base+=T) {
        int ak=base+threadIdx.x, bk=base+threadIdx.y;
        sa[threadIdx.y][threadIdx.x]=(r<m && ak<k)?a[(int64_t)r*k+ak]:0;
        sb[threadIdx.y][threadIdx.x]=(bk<k && col<n)?b[(int64_t)bk*n+col]:0;
        __syncthreads();
        #pragma unroll
        for (int q=0; q<T; ++q) v+=sa[threadIdx.y][q]*sb[q][threadIdx.x];
        __syncthreads();
    }
    if (r<m && col<n) c[(int64_t)r*n+col]=v;
}

API int hy_cuda_abi_version() { return 1; }
API const char* hy_cuda_error(int code) { return cudaGetErrorString((cudaError_t)code); }
API int hy_gemm(const float* a, const float* b, float* c, int m, int n,
                int k, int variant, int tile, void* stream) {
    if (!a || !b || !c || m<=0 || n<=0 || k<=0 || variant<0 || variant>2 ||
        (tile!=16 && tile!=32)) return (int)cudaErrorInvalidValue;
    cudaStream_t s=(cudaStream_t)stream;
    if (variant==0) gemm_basic<<<((int64_t)m*n+255)/256,256,0,s>>>(a,b,c,m,n,k);
    else {
        dim3 block(tile,tile), grid((n+tile-1)/tile,(m+tile-1)/tile);
        if (variant==1) {
            if (tile==16) gemm_blocked<16><<<grid,block,0,s>>>(a,b,c,m,n,k);
            else gemm_blocked<32><<<grid,block,0,s>>>(a,b,c,m,n,k);
        } else {
            if (tile==16) gemm_shared<16><<<grid,block,0,s>>>(a,b,c,m,n,k);
            else gemm_shared<32><<<grid,block,0,s>>>(a,b,c,m,n,k);
        }
    }
    return (int)cudaGetLastError();
}

// Matrix layout: (N*OH*OW, C*KH*KW), directly consumed by GEMM.
__global__ void im2col_kernel(const float* x, float* col, int N, int C,
    int H, int W, int KH, int KW, int SH, int SW, int PH, int PW, int OH, int OW) {
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x, K=(int64_t)C*KH*KW;
    if (i>=(int64_t)N*OH*OW*K) return;
    int q=i%K, row=i/K, kw=q%KW, kh=q/KW%KH, c=q/(KH*KW);
    int n=row/(OH*OW), h=(row/OW%OH)*SH-PH+kh, w=(row%OW)*SW-PW+kw;
    col[i]=(h>=0 && h<H && w>=0 && w<W)?x[((int64_t)n*C+c)*H*W+h*W+w]:0;
}
API int hy_im2col(const float* x, float* col, int N, int C, int H, int W,
    int KH, int KW, int SH, int SW, int PH, int PW, int OH, int OW, void* stream) {
    int64_t count=(int64_t)N*OH*OW*C*KH*KW;
    im2col_kernel<<<(count+255)/256,256,0,(cudaStream_t)stream>>>
        (x,col,N,C,H,W,KH,KW,SH,SW,PH,PW,OH,OW);
    return (int)cudaGetLastError();
}

__global__ void conv_direct(const float* x, const float* w, const float* b,
    float* y, int N, int C, int H, int W, int OC, int KH, int KW,
    int SH, int SW, int PH, int PW, int OH, int OW) {
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=(int64_t)N*OC*OH*OW) return;
    int ow=i%OW, oh=i/OW%OH, oc=i/(OW*OH)%OC, n=i/(OW*OH*OC);
    float v=b?b[oc]:0;
    for(int c=0;c<C;++c) for(int kh=0;kh<KH;++kh) for(int kw=0;kw<KW;++kw) {
        int h=oh*SH-PH+kh, ww=ow*SW-PW+kw;
        if(h>=0 && h<H && ww>=0 && ww<W)
            v+=x[((int64_t)n*C+c)*H*W+h*W+ww]*w[((int64_t)oc*C+c)*KH*KW+kh*KW+kw];
    }
    y[i]=v;
}
API int hy_conv(const float* x, const float* w, const float* b, float* y,
    int N, int C, int H, int W, int OC, int KH, int KW, int SH, int SW,
    int PH, int PW, int OH, int OW, void* stream) {
    conv_direct<<<((int64_t)N*OC*OH*OW+255)/256,256,0,(cudaStream_t)stream>>>
        (x,w,b,y,N,C,H,W,OC,KH,KW,SH,SW,PH,PW,OH,OW);
    return (int)cudaGetLastError();
}

// Fuse GEMM output layout conversion and channel bias into one kernel.
__global__ void conv_pack(const float* a,const float* b,float* y,int N,int OC,int S) {
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=(int64_t)N*OC*S) return;
    int s=i%S,c=i/S%OC,n=i/(S*OC);
    y[i]=a[((int64_t)n*S+s)*OC+c]+(b?b[c]:0);
}
API int hy_conv_pack(const float* a,const float* b,float* y,int N,int OC,int S,void* stream) {
    conv_pack<<<((int64_t)N*OC*S+255)/256,256,0,(cudaStream_t)stream>>>(a,b,y,N,OC,S);
    return (int)cudaGetLastError();
}

// Match existing HyTorch: zero padding, first index wins ties, NaN propagates.
__global__ void pool_kernel(const float* x,float* y,int64_t* indexes,int N,int C,
    int H,int W,int KH,int KW,int SH,int SW,int PH,int PW,int OH,int OW) {
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=(int64_t)N*C*OH*OW) return;
    int ow=i%OW,oh=i/OW%OH,nc=i/(OW*OH),best=0;
    float v=-INFINITY;
    for(int q=0;q<KH*KW;++q) {
        int h=oh*SH-PH+q/KW,w=ow*SW-PW+q%KW;
        float z=(h>=0 && h<H && w>=0 && w<W)?x[(int64_t)nc*H*W+h*W+w]:0;
        if(q==0 || (!isnan(v) && (z>v || isnan(z)))) {v=z;best=q;}
    }
    y[i]=v; indexes[i]=best;
}
API int hy_pool(const float* x,float* y,int64_t* indexes,int N,int C,int H,int W,
    int KH,int KW,int SH,int SW,int PH,int PW,int OH,int OW,void* stream) {
    pool_kernel<<<((int64_t)N*C*OH*OW+255)/256,256,0,(cudaStream_t)stream>>>
        (x,y,indexes,N,C,H,W,KH,KW,SH,SW,PH,PW,OH,OW);
    return (int)cudaGetLastError();
}
