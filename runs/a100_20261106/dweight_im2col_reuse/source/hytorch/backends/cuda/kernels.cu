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

// A 64x64 output tile with sixteen FP32 accumulators per thread. Reuse each
// shared-memory operand across four outputs instead of loading it per FMA.
// Keep the original kernel for small/narrow matrices to avoid wasted lanes.
template<int K> __global__ void gemm_register(const float* a, const float* b,
                                             float* c, int m, int n, int k) {
    __shared__ float sa[K][65], sb[K][64];
    int tx=threadIdx.x, ty=threadIdx.y, tid=ty*16+tx;
    int row0=blockIdx.y*64, col0=blockIdx.x*64;
    float acc[4][4]={};
    for (int base=0; base<k; base+=K) {
        for (int i=tid; i<64*K; i+=256) {
            int r=i/K, q=i%K;
            sa[q][r]=(row0+r<m && base+q<k)?a[(int64_t)(row0+r)*k+base+q]:0;
            q=i/64; int col=i%64;
            sb[q][col]=(base+q<k && col0+col<n)?b[(int64_t)(base+q)*n+col0+col]:0;
        }
        __syncthreads();
        #pragma unroll
        for (int q=0; q<K; ++q) {
            float av[4], bv[4];
            #pragma unroll
            for (int i=0; i<4; ++i) {
                av[i]=sa[q][ty+i*16];
                bv[i]=sb[q][tx+i*16];
            }
            #pragma unroll
            for (int i=0; i<4; ++i)
                #pragma unroll
                for (int j=0; j<4; ++j) acc[i][j]+=av[i]*bv[j];
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i=0; i<4; ++i)
        #pragma unroll
        for (int j=0; j<4; ++j) {
            int r=row0+ty+i*16, col=col0+tx+j*16;
            if (r<m && col<n) c[(int64_t)r*n+col]=acc[i][j];
        }
}

API int hy_cuda_abi_version() { return 1; }
API const char* hy_cuda_error(int code) { return cudaGetErrorString((cudaError_t)code); }
API int hy_gemm(const float* a, const float* b, float* c, int m, int n,
                int k, int variant, int tile, void* stream) {
    if (!a || !b || !c || m<=0 || n<=0 || k<=0 || variant<0 || variant>2 ||
        (tile!=16 && tile!=32)) return (int)cudaErrorInvalidValue;
    // Conservative shape-only heuristic; 128 CTAs is a tuning starting point.
    const int64_t grid_m = ((int64_t)m + 63) / 64;
    const int64_t grid_n = ((int64_t)n + 63) / 64;
    const int64_t register_blocks = grid_m * grid_n;
    constexpr int64_t kMinRegisterBlocks = 128;
    const bool use_register = variant == 2 && m >= 64 && n >= 64 && k >= 64 &&
                              register_blocks >= kMinRegisterBlocks;
    cudaStream_t s=(cudaStream_t)stream;
    if (variant==0) gemm_basic<<<((int64_t)m*n+255)/256,256,0,s>>>(a,b,c,m,n,k);
    else if (use_register) {
        dim3 block(16,16), grid((n+63)/64,(m+63)/64);
        if (tile==16) gemm_register<16><<<grid,block,0,s>>>(a,b,c,m,n,k);
        else gemm_register<32><<<grid,block,0,s>>>(a,b,c,m,n,k);
    } else {
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

// Strided FP32 col2im: one writer per NCHW output, ascending kh/kw reduction.
__global__ void col2im_strided(const float* col, float* img,
    int64_t n, int64_t c, int64_t h, int64_t w,
    int64_t kh, int64_t kw, int64_t oh, int64_t ow,
    int64_t sh, int64_t sw, int64_t ph, int64_t pw,
    int64_t s0, int64_t s1, int64_t s2, int64_t s3, int64_t s4, int64_t s5) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = n * c * h * w;
    const int64_t step = (int64_t)gridDim.x * blockDim.x;
    for (; i < total; i += step) {
        const int64_t x = i % w, y = (i / w) % h;
        const int64_t ci = (i / (w * h)) % c, ni = i / (w * h * c);
        const int64_t ty = y + ph, tx = x + pw;
        // 0 <= (ty-ky)/sh < oh, and ky == ty (mod sh); analogous for x.
        int64_t ky0 = ty - (oh - 1) * sh;
        int64_t kx0 = tx - (ow - 1) * sw;
        ky0 = ky0 > 0 ? ky0 : 0;
        kx0 = kx0 > 0 ? kx0 : 0;
        const int64_t ky1 = ty < kh - 1 ? ty : kh - 1;
        const int64_t kx1 = tx < kw - 1 ? tx : kw - 1;
        float val = 0.0f;
        if (ky0 <= ky1 && kx0 <= kx1) {
            if (sh != 1) ky0 += (ty - ky0) % sh;
            if (sw != 1) kx0 += (tx - kx0) % sw;
            const int64_t base = ni * s0 + ci * s1;
            // Step directly over valid congruence classes; no inner validity tests.
            const int64_t iy0 = sh == 1 ? ty - ky0 : (ty - ky0) / sh;
            const int64_t ix0 = sw == 1 ? tx - kx0 : (tx - kx0) / sw;
            for (int64_t ky = ky0, iy = iy0; ky <= ky1; ky += sh, --iy) {
                for (int64_t kx = kx0, ix = ix0; kx <= kx1; kx += sw, --ix) {
                    val = val + col[base + ky * s2 + kx * s3 + iy * s4 + ix * s5];
                }
            }
        }
        img[i] = val;
        if (total - i <= step) break;  // Avoid signed overflow of the grid-stride increment.
    }
}

API int hy_col2im(const float* col, float* img,
    int64_t n, int64_t c, int64_t h, int64_t w,
    int64_t kh, int64_t kw, int64_t oh, int64_t ow,
    int64_t sh, int64_t sw, int64_t ph, int64_t pw,
    int64_t s0, int64_t s1, int64_t s2, int64_t s3, int64_t s4, int64_t s5,
    void* stream) {
    if (!col || !img || n <= 0 || c <= 0 || h <= 0 || w <= 0 ||
        kh <= 0 || kw <= 0 || oh <= 0 || ow <= 0 || sh <= 0 || sw <= 0 ||
        ph < 0 || pw < 0) return (int)cudaErrorInvalidValue;
    // Python binding validates products/offsets; cap grid for wide-index outputs.
    const int64_t total = n * c * h * w;
    const int64_t needed = (total - 1) / 256 + 1;
    const unsigned int blocks = (unsigned int)(needed < 65535 ? needed : 65535);
    col2im_strided<<<blocks,256,0,(cudaStream_t)stream>>>(col,img,n,c,h,w,
        kh,kw,oh,ow,sh,sw,ph,pw,s0,s1,s2,s3,s4,s5);
    return (int)cudaGetLastError();
}

// Optional ABI-1 host binding extension. Native-endian signed 64-bit POD fields.
// Byte input is copied to an aligned local object; the caller owns it until return.
struct Col2imParams { int64_t v[18]; };
static_assert(sizeof(Col2imParams) == 144, "col2im POD size");
static_assert(alignof(Col2imParams) == alignof(int64_t), "col2im POD alignment");
API int hy_col2im_fast(const float* col, float* img, const void* params, void* stream) {
    if (!params) return (int)cudaErrorInvalidValue;
    Col2imParams p;
    memcpy(&p, params, sizeof(p));
    return hy_col2im(col,img,p.v[0],p.v[1],p.v[2],p.v[3],p.v[4],p.v[5],
        p.v[6],p.v[7],p.v[8],p.v[9],p.v[10],p.v[11],p.v[12],p.v[13],
        p.v[14],p.v[15],p.v[16],p.v[17],stream);
}
