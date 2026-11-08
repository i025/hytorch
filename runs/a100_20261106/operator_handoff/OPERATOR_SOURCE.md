# 核心源码交付与调用入口

源码为本轮开始时的逐字节快照，保持相对路径。行号均指快照中的真实文件；本轮未修改生产代码。

完整文件均在 `source/`；下方附可直接读取的关键函数完整函数体，不以省略号替代。CuPy cuBLAS 的底层实现为已安装二进制，未伪造其源码。

## 文件与符号索引

| 文件 | 符号及行号 |
|---|---|
| [README.md](source/README.md) | 完整文件 |
| [requirements-cuda.txt](source/requirements-cuda.txt) | 完整文件 |
| [donkeycar_hy/model.py](source/donkeycar_hy/model.py) | `DonkeyNet` L30–84; `build_model` L87–88 |
| [donkeycar_hy/tub_dataset.py](source/donkeycar_hy/tub_dataset.py) | `_preprocess` L37–62; `DonkeyDataset` L65–160 |
| [benchmarks/bench_cuda_training.py](source/benchmarks/bench_cuda_training.py) | `TinyNet` L24–32; `SyntheticDataset` L35–45; `LimitedDataset` L48–56; `parameters` L59–62; `snapshot` L65–66; `state_hash` L69–73; `initialize` L76–87; `torch_model` L90–121; `main` L124–306 |
| [benchmarks/bench_cuda_gemm.py](source/benchmarks/bench_cuda_gemm.py) | `errors` L16–18; `cpu_time` L21–28; `gpu_time` L31–42; `main` L45–134 |
| [benchmarks/bench_cuda_ops.py](source/benchmarks/bench_cuda_ops.py) | `main` L16–65 |
| [benchmarks/bench_cuda_revisions.py](source/benchmarks/bench_cuda_revisions.py) | `main` L23–103 |
| [scripts/build_cuda.py](source/scripts/build_cuda.py) | `build_command` L15–23; `main` L26–54 |
| [scripts/validate_cuda.py](source/scripts/validate_cuda.py) | `main` L13–40 |
| [scripts/report_cuda.py](source/scripts/report_cuda.py) | `read_rows` L11–12; `compare_training` L15–63; `main` L66–133 |
| [scripts/run_cuda_comparison.py](source/scripts/run_cuda_comparison.py) | `main` L11–54 |
| [hytorch/__init__.py](source/hytorch/__init__.py) |  |
| [hytorch/core.py](source/hytorch/core.py) | `Config` L11–13; `using_config` L17–23; `test_mode` L26–27; `no_grad` L30–31; `Variable` L44–163; `Parameter` L166–167; `as_array` L170–173; `as_variable` L176–179; `Function` L182–205; `Add` L229–240; `add` L243–246; `Mul` L249–261; `mul` L264–266; `Neg` L269–274; `neg` L277–278; `Sub` L281–292; `sub` L295–297; `rsub` L300–302; `Div` L305–318; `div` L321–323; `rdiv` L326–328; `Pow` L331–343; `pow` L346–347; `setup_variable` L350–366 |
| [hytorch/functions.py](source/hytorch/functions.py) | `Sin` L15–24; `sin` L27–28; `Cos` L31–40; `cos` L43–44; `Tanh` L47–56; `tanh` L59–60; `Exp` L63–72; `exp` L75–76; `Log` L79–89; `log` L92–93; `Reshape` L99–109; `reshape` L112–115; `Transpose` L119–139; `transpose` L142–143; `GetItem` L146–157; `GetItemGrad` L160–176; `get_item` L179–181; `flatten` L186–188; `Sum` L194–207; `sum` L210–211; `BroadcastTo` L214–226; `broadcast_to` L229–232; `SumTo` L235–246; `sum_to` L249–252; `average` L255–258; `MatMul` L263–279; `matmul` L282–283; `_swap_last2_axes` L289–293; `BatchMatMul` L295–311; `bmm` L313–314; `linear_simple` L317–323; `Linear` L326–341; `linear` L344–345; `sigmoid_simple` L352–354; `Sigmoid` L357–367; `sigmoid` L370–371; `ReLU` L374–384; `relu` L387–388; `softmax_simple` L392–396; `Softmax` L399–415; `softmax` L418–419; `LogSoftmax` L421–433; `log_softmax` L436–437; `LeakyReLU` L440–454; `leaky_relu` L457–458; `mean_square_error_simple` L464–466; `MeanSquareError` L469–480; `mean_square_error` L483–484; `softmax_cross_entropy_simple` L488–498; `softmax_cross_entropy_soft_label` L501–511; `SoftmaxCrossEntropy` L514–533; `softmax_cross_entropy` L536–537; `sigmoid_cross_entropy` L540–549; `binary_cross_entropy` L552–559; `accuracy` L566–572; `dropout` L576–586; `embed_id` L589–590; `BatchNorm` L593–658; `batch_norm` L661–662; `Max` L668–686; `Min` L689–692; `max` L695–696; `min` L699–700; `Clip` L703–717; `clip` L720–721; `Concat` L724–753; `concat` L755–758 |
| [hytorch/functions_conv.py](source/hytorch/functions_conv.py) | `conv2d_simple` L12–27; `pooling_simple` L30–44; `Conv2d` L50–81; `conv2d` L84–85; `Deconv2d` L88–132; `deconv2d` L135–136; `Conv2DGradW` L139–171; `Pooling` L177–199; `pooling` L201–202; `Pooling2DGrad` L204–237; `Pooling2DWithIndexes` L240–257; `pooling` L260–261; `AveragePooling` L264–288; `average_pooling` L291–292; `Im2col` L298–316; `im2col` L319–346; `Col2im` L349–366; `col2im` L369–370; `im2col_array` L376–402; `conv2d_cpu_tensordot_array` L405–413; `conv2d_cpu_im2col_gemm_array` L416–439; `col2im_array` L442–465; `_im2col_gpu` L468–504; `_col2im_gpu` L507–544; `BatchNorm` L552–591; `batch_norm` L594–609 |
| [hytorch/layers.py](source/hytorch/layers.py) | `Layer` L14–87; `Linear` L90–120; `TwoLayerNet` L123–132; `Conv2d` L135–181; `Deconv2d` L184–230; `_get_rng` L235–244; `dropout` L247–263; `Dropout` L266–274; `Dropout2d` L277–301; `BatchNorm2d` L307–340; `EmbedID` L346–353; `BatchNorm` L356–383; `LayerNorm` L388–440; `LinearND` L445–488 |
| [hytorch/cuda.py](source/hytorch/cuda.py) | `get_array_module` L11–27; `as_numpy` L30–46; `as_cupy` L49–63 |
| [hytorch/optimizers.py](source/hytorch/optimizers.py) | `Optimizer` L9–36; `WeightDecay` L42–48; `ClipGrad` L51–64; `FreezeParam` L67–79; `SGD` L86–92; `MomentumSGD` L95–111; `AdaGrad` L114–134; `AdaDelta` L137–163; `Adam` L166–201; `AdamW` L203–245 |
| [hytorch/dataloaders.py](source/hytorch/dataloaders.py) | `_SyncDataLoader` L14–57; `_WorkerError` L66–69; `_DatasetWorker` L72–93; `AsyncDataLoader` L96–208; `DataLoader` L214–262 |
| [hytorch/profiling.py](source/hytorch/profiling.py) | `dump_json` L12–13; `ResourceMonitor` L16–91; `StepProfiler` L94–157; `diagnose` L160–184 |
| [hytorch/utils.py](source/hytorch/utils.py) | `_dot_var` L13–22; `_dot_func` L25–36; `get_dot_graph` L39–78; `plot_dot_graph` L81–101; `sum_to` L107–125; `reshape_sum_backward` L128–157; `logsumexp` L160–168; `max_backward_shape` L171–180; `gradient_check` L186–231; `numerical_grad` L234–281; `array_equal` L284–297; `array_allclose` L300–317; `show_progress` L323–334; `get_file` L340–379; `get_deconv_outsize` L385–386; `get_conv_outsize` L389–390; `pair` L393–400 |
| [hytorch/backends/cuda/kernels.cu](source/hytorch/backends/cuda/kernels.cu) | 完整文件 |
| [hytorch/backends/cuda/runtime.py](source/hytorch/backends/cuda/runtime.py) | `current_config` L17–18; `dispatch_stats` L21–25; `backend` L29–40; `library_path` L43–45; `load_library` L48–72; `_call` L75–80; `_ptr` L83–84; `_arrays` L87–97; `_shape_ok` L100–102; `gemm` L105–128; `_geometry` L131–141; `conv2d` L144–169; `max_pool2d` L172–181; `selected` L184–193 |
| [hytorch/backends/cuda/__init__.py](source/hytorch/backends/cuda/__init__.py) |  |
| [hytorch/backends/cpu/gemm.py](source/hytorch/backends/cpu/gemm.py) | `_candidate_build_roots` L19–26; `_try_compile_with_cl` L29–34; `_try_compile_with_gcc` L37–65; `_load_c_kernel` L68–153; `c_backend_available` L156–157; `_gemm_call_c` L160–197; `cpu_gemm` L200–229; `gemm` L232–233 |
| [tests/test_cuda_device.py](source/tests/test_cuda_device.py) | `cp` L13–24; `test_gemm_rectangular_and_current_stream` L30–43; `test_gemm_views_and_alias_rejection` L46–54; `test_gemm_register_tiles_and_tails` L59–74; `test_conv_forward_backward` L80–95; `test_pool_ties_negative_padding_backward` L99–111; `test_pool_nan_propagates_first_index` L114–118; `test_framework_linear_forward_native_backward_original` L121–134 |
| [tests/test_cuda_host.py](source/tests/test_cuda_host.py) | `test_context_restores_and_cpu_fallback` L17–30; `test_freeze_hook_excludes_update` L34–44; `test_initializers_preserve_float32` L47–51; `test_profiler_denominator_and_throughput` L54–68; `test_monitor_parsing_and_missing_fields` L71–85; `test_pretrained_model_schema` L89–96; `test_preprocess_shapes_and_channels` L99–103; `test_build_arch_validation` L106–109; `test_torch_parity_forward_backward_update` L113–138; `test_missing_library_fails_not_silent_fallback` L141–146 |
| [tests/test_gemm_blocked_packed.py](source/tests/test_gemm_blocked_packed.py) | `TestGemmBlockedPacked` L7–45 |
| [tests/test_conv2d_gemm_cpu.py](source/tests/test_conv2d_gemm_cpu.py) | `TestConv2dGemmCPU` L12–46 |
| [hytorch/backends/cpu/gemm_kernels.c](source/hytorch/backends/cpu/gemm_kernels.c) | 完整文件 |

## 关键调用关系

1. `benchmarks/bench_cuda_training.py:124 main` → `DonkeyNet.forward` → `layers.Conv2d.forward` → `functions_conv.Conv2d.forward`。
2. native 前向：`runtime.selected` → `runtime.conv2d` → `_call(hy_im2col)` → `runtime.gemm` → `_arrays` 连续化权重 → `_call(hy_gemm)` → `_call(hy_conv_pack)`。
3. `Variable.backward` → `Conv2d.backward`，其调用顺序为 dInput、dWeight、dBias。dInput 使用 `Deconv2d.forward` 的 CuPy tensordot/rollaxis/col2im；dWeight 使用 `Conv2DGradW.forward` 的 CuPy im2col/tensordot；dBias 使用 `Sum.forward`。
4. 实际全连接层调用 `F.linear_simple`，不是 `F.Linear`：`MatMul.forward` → native dispatch，随后 Add；`MatMul.backward` 显式用 `MatMul(native=False)` 回到 CuPy dot。两种 Linear API 都提供完整源码，避免混淆。
5. 标量类型路径：`MeanSquareError.backward` → `core.mul` → `as_array` → `cupy.array(2./len(diff))`，实测标量为 FP64；`Tanh.backward` 的 `1-y*y` 经 `rsub/as_array` 生成 int64 标量，实测也把此表达式提升为 FP64。
6. `Pooling.forward` 支持 native，但 backward 的 scatter/col2im 仍为 CuPy；DonkeyNet 本轮实际调用次数为 0。
7. 绑定是 Python ctypes + `extern "C"` CUDA ABI，无额外 C++ binding 层。库本身不分配、不传输、不主动同步；工作区由 CuPy 管理，使用当前 stream 指针。
8. 数据 loader 在 CPU/NumPy 组 batch；每步 H2D；`StepProfiler` 同步边界；每步 scalar(loss) D2H；SGD Python 循环逐参数更新。

## hytorch/backends/cuda/kernels.cu（完整文件）

```cuda
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
    cudaStream_t s=(cudaStream_t)stream;
    if (variant==0) gemm_basic<<<((int64_t)m*n+255)/256,256,0,s>>>(a,b,c,m,n,k);
    else if (variant==2 && m>=64 && n>=64 && k>=64) {
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
```

## hytorch/backends/cuda/runtime.py（完整文件）

```python
"""ctypes CUDA ABI with explicit current-stream and device ownership."""
import ctypes as ct
import os
from pathlib import Path
from contextlib import contextmanager
from contextvars import ContextVar
from collections import Counter
import numpy as np

_config = ContextVar("hytorch_cuda_backend", default={
    "matmul": "cupy", "conv": "cupy", "pool": "cupy", "tile": 16})
_counts = Counter()
_library = None
_dll_directories = []


def current_config():
    return dict(_config.get())


def dispatch_stats(reset=False):
    result = dict(_counts)
    if reset:
        _counts.clear()
    return result


@contextmanager
def backend(matmul="cupy", conv="cupy", pool="cupy", tile=16):
    if matmul not in ("cupy", "basic", "blocked", "shared"):
        raise ValueError("matmul must be cupy/basic/blocked/shared")
    if conv not in ("cupy", "direct", "im2col") or pool not in ("cupy", "native"):
        raise ValueError("conv must be cupy/direct/im2col; pool must be cupy/native")
    if tile not in (16, 32):
        raise ValueError("tile must be 16 or 32")
    token = _config.set(dict(matmul=matmul, conv=conv, pool=pool, tile=tile))
    try:
        yield
    finally:
        _config.reset(token)


def library_path():
    name = "hytorch_cuda.dll" if os.name == "nt" else "libhytorch_cuda.so"
    return Path(os.environ.get("HYTORCH_CUDA_LIBRARY", Path(__file__).parent / "build" / name))


def load_library():
    global _library
    if _library is None:
        path = library_path()
        if not path.is_file():
            raise RuntimeError(f"CUDA library missing: {path}. Run python scripts/build_cuda.py on NVIDIA host.")
        if os.name == "nt" and os.environ.get("CUDA_PATH"):
            _dll_directories.append(os.add_dll_directory(str(Path(os.environ["CUDA_PATH"]) / "bin")))
        lib = ct.CDLL(str(path))
        p, i = ct.c_void_p, ct.c_int
        signatures = {
            "hy_cuda_abi_version": [], "hy_cuda_error": [i],
            "hy_gemm": [p]*3 + [i]*5 + [p],
            "hy_im2col": [p]*2 + [i]*12 + [p],
            "hy_conv": [p]*4 + [i]*13 + [p],
            "hy_conv_pack": [p]*3 + [i]*3 + [p],
            "hy_pool": [p]*3 + [i]*12 + [p],
        }
        for name, args in signatures.items():
            fn = getattr(lib, name)
            fn.argtypes, fn.restype = args, ct.c_char_p if name == "hy_cuda_error" else i
        if lib.hy_cuda_abi_version() != 1:
            raise RuntimeError("Incompatible CUDA library ABI; rebuild library")
        _library = lib
    return _library


def _call(name, *args):
    import cupy as cp
    lib = load_library()
    status = getattr(lib, name)(*args, cp.cuda.get_current_stream().ptr)
    if status:
        raise RuntimeError(f"{name}: {lib.hy_cuda_error(status).decode()}")


def _ptr(x):
    return None if x is None else x.data.ptr


def _arrays(*arrays):
    import cupy as cp
    valid = [x for x in arrays if x is not None]
    if any(not isinstance(x, cp.ndarray) or x.dtype != cp.float32 for x in valid):
        raise TypeError("Native CUDA kernels require CuPy float32 arrays")
    device = cp.cuda.runtime.getDevice()
    if any(x.device.id != device for x in valid):
        raise ValueError("All arrays must belong to the current CUDA device")
    if any(x.size == 0 or x.size > 2**31-1 for x in valid):
        raise ValueError("Native CUDA arrays must have 1..INT_MAX elements")
    return tuple(None if x is None else cp.ascontiguousarray(x) for x in arrays)


def _shape_ok(shape):
    if any(v <= 0 for v in shape) or int(np.prod(shape, dtype=np.int64)) > 2**31-1:
        raise ValueError(f"Unsupported native output/workspace shape: {shape}")


def gemm(a, b, variant="shared", tile=16, out=None):
    import cupy as cp
    if variant not in ("basic", "blocked", "shared") or tile not in (16, 32):
        raise ValueError("Invalid GEMM variant or tile")
    a, b = _arrays(a, b)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("GEMM expects (M,K) @ (K,N)")
    m, k = a.shape
    n = b.shape[1]
    _shape_ok((m, n))
    if (m+tile-1)//tile > 65535:
        raise ValueError("GEMM row grid exceeds CUDA grid.y limit")
    if out is None:
        out = cp.empty((m, n), dtype=cp.float32)
    else:
        _arrays(out)
        if out.shape != (m, n) or not out.flags.c_contiguous:
            raise ValueError("out must be contiguous with shape (M,N)")
        if cp.may_share_memory(out, a) or cp.may_share_memory(out, b):
            raise ValueError("out must not overlap inputs")
    _call("hy_gemm", _ptr(a), _ptr(b), _ptr(out), m, n, k,
          ("basic", "blocked", "shared").index(variant), tile)
    _counts["gemm_" + variant] += 1
    return out


def _geometry(x, kernel, stride, pad):
    from hytorch.utils import pair
    if x.ndim != 4:
        raise ValueError("Expected NCHW input")
    kh, kw = pair(kernel); sh, sw = pair(stride); ph, pw = pair(pad)
    if min(kh, kw, sh, sw) <= 0 or min(ph, pw) < 0:
        raise ValueError("Invalid kernel/stride/padding")
    n, c, h, w = x.shape
    oh, ow = (h+2*ph-kh)//sh+1, (w+2*pw-kw)//sw+1
    _shape_ok((n, c, oh, ow))
    return n, c, h, w, kh, kw, sh, sw, ph, pw, oh, ow


def conv2d(x, w, b=None, stride=1, pad=0, variant="im2col", tile=16):
    import cupy as cp
    if variant not in ("direct", "im2col") or tile not in (16, 32):
        raise ValueError("Invalid convolution variant or tile")
    x, w, b = _arrays(x, w, b)
    if w.ndim != 4 or x.ndim != 4 or x.shape[1] != w.shape[1]:
        raise ValueError("Expected NCHW input and OIHW weights with equal channels")
    g = _geometry(x, w.shape[2:], stride, pad)
    n, c, h, wi, kh, kw, sh, sw, ph, pw, oh, ow = g
    oc = w.shape[0]
    if b is not None and b.shape != (oc,):
        raise ValueError("Bias must have shape (out_channels,)")
    _shape_ok((n, oc, oh, ow))
    y = cp.empty((n, oc, oh, ow), dtype=cp.float32)
    if variant == "direct":
        _call("hy_conv", _ptr(x), _ptr(w), _ptr(b), _ptr(y),
              n, c, h, wi, oc, kh, kw, sh, sw, ph, pw, oh, ow)
    else:
        shape = (n*oh*ow, c*kh*kw)
        _shape_ok(shape)
        col = cp.empty(shape, dtype=cp.float32)
        _call("hy_im2col", _ptr(x), _ptr(col), *g)
        result = gemm(col, w.reshape(oc, -1).T, tile=tile)
        _call("hy_conv_pack", _ptr(result), _ptr(b), _ptr(y), n, oc, oh*ow)
    _counts["conv_" + variant] += 1
    return y


def max_pool2d(x, kernel, stride=1, pad=0):
    import cupy as cp
    x, = _arrays(x)
    g = _geometry(x, kernel, stride, pad)
    n, c, _, _, _, _, _, _, _, _, oh, ow = g
    y = cp.empty((n, c, oh, ow), dtype=cp.float32)
    indexes = cp.empty(y.shape, dtype=cp.int64)
    _call("hy_pool", _ptr(x), _ptr(y), _ptr(indexes), *g)
    _counts["pool_native"] += 1
    return y, indexes


def selected(op, x):
    """CPU / non-FP32 paths retain existing semantics, with visible counters."""
    mode = _config.get()[op]
    if mode == "cupy":
        return None
    from hytorch import cuda
    if cuda.get_array_module(x) is np or x.dtype != np.float32:
        _counts[op + "_fallback_cpu_or_dtype"] += 1
        return None
    return mode
```

## hytorch/functions_conv.py

完整 imports 和上下文见 `source/hytorch/functions_conv.py`。

### Conv2d（L50–81）

```python
class Conv2d(Function):
    def __init__(self, stride=1, pad=0):
        super().__init__()
        self.stride = pair(stride)
        self.pad = pair(pad)

    def forward(self, x, W, b):
        xp = cuda.get_array_module(x)

        from hytorch.backends.cuda.runtime import selected, conv2d as native_conv, current_config
        mode = selected("conv", x)
        if mode is not None:
            return native_conv(x, W, b, self.stride, self.pad,
                               variant=mode, tile=current_config()["tile"])

        if xp is np:
            return conv2d_cpu_im2col_gemm_array(x, W, b, self.stride, self.pad)

        return conv2d_cpu_tensordot_array(x, W, b, self.stride, self.pad, xp=xp)

    def backward(self, gy):
        x, W, b = self.inputs
        # ==== gx ====
        gx = deconv2d(gy, W, b=None, stride=self.stride, pad=self.pad,
                      outsize=(x.shape[2], x.shape[3]))
        # ==== gW ====
        gW = Conv2DGradW(self)(x, gy)
        # ==== gb ====
        gb = None
        if b.data is not None:
            gb = gy.sum(axis=(0, 2, 3))
        return gx, gW, gb
```

### Deconv2d（L88–132）

```python
class Deconv2d(Function):
    def __init__(self, stride=1, pad=0, outsize=None):
        super().__init__()
        self.stride = pair(stride)
        self.pad = pair(pad)
        self.outsize = outsize

    def forward(self, x, W, b):
        xp = cuda.get_array_module(x)

        Weight = W
        SH, SW = self.stride
        PH, PW = self.pad
        C, OC, KH, KW = Weight.shape
        N, C, H, W = x.shape
        if self.outsize is None:
            out_h = get_deconv_outsize(H, KH, SH, PH)
            out_w = get_deconv_outsize(W, KW, SW, PW)
        else:
            out_h, out_w = pair(self.outsize)
        img_shape = (N, OC, out_h, out_w)

        gcol = xp.tensordot(Weight, x, (0, 1))
        gcol = xp.rollaxis(gcol, 3)
        y = col2im_array(gcol, img_shape, (KH, KW), self.stride, self.pad,
                         to_matrix=False)
        # b, k, h, w
        if b is not None:
            self.no_bias = True
            y += b.reshape((1, b.size, 1, 1))
        return y

    def backward(self, gy):
        x, W, b = self.inputs

        # ==== gx ====
        gx = conv2d(gy, W, b=None, stride=self.stride, pad=self.pad)
        # ==== gW ====
        f = Conv2DGradW(self)
        gW = f(gy, x)
        # ==== gb ====
        gb = None
        if b.data is not None:
            gb = gy.sum(axis=(0, 2, 3))
        return gx, gW, gb
```

### Conv2DGradW（L139–171）

```python
class Conv2DGradW(Function):
    def __init__(self, conv2d):
        W = conv2d.inputs[1]
        kh, kw = W.shape[2:]
        self.kernel_size = (kh, kw)
        self.stride = conv2d.stride
        self.pad = conv2d.pad

    def forward(self, x, gy):
        xp = cuda.get_array_module(x)

        if xp is np:
            kh, kw = self.kernel_size
            col = im2col_array(x, self.kernel_size, self.stride, self.pad,
                               to_matrix=True)
            gy_matrix = gy.transpose(0, 2, 3, 1).reshape(-1, gy.shape[1])
            gW_matrix = cpu_gemm(gy_matrix.transpose(1, 0), col)
            return gW_matrix.reshape(gy.shape[1], x.shape[1], kh, kw)

        col = im2col_array(x, self.kernel_size, self.stride, self.pad,
                           to_matrix=False)
        gW = xp.tensordot(gy, col, ((0, 2, 3), (0, 4, 5)))
        return gW

    def backward(self, gys):
        x, gy = self.inputs
        gW, = self.outputs

        xh, xw = x.shape[2:]
        gx = deconv2d(gy, gW, stride=self.stride, pad=self.pad,
                      outsize=(xh, xw))
        ggy = conv2d(x, gW, stride=self.stride, pad=self.pad)
        return gx, ggy
```

### Pooling（L177–199）

```python
class Pooling(Function):
    def __init__(self, kernel_size, stride=1, pad=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        from hytorch.backends.cuda.runtime import selected, max_pool2d
        if selected("pool", x) is not None:
            y, self.indexes = max_pool2d(x, self.kernel_size, self.stride, self.pad)
            return y
        col = im2col_array(x, self.kernel_size, self.stride, self.pad,
                           to_matrix=False)

        N, C, KH, KW, OH, OW = col.shape
        col = col.reshape(N, C, KH * KW, OH, OW)
        self.indexes = col.argmax(axis=2)
        y = col.max(axis=2)
        return y

    def backward(self, gy):
        return Pooling2DGrad(self)(gy)
```

### Pooling2DGrad（L204–237）

```python
class Pooling2DGrad(Function):
    def __init__(self, mpool2d):
        self.mpool2d = mpool2d
        self.kernel_size = mpool2d.kernel_size
        self.stride = mpool2d.stride
        self.pad = mpool2d.pad
        self.input_shape = mpool2d.inputs[0].shape
        self.dtype = mpool2d.inputs[0].dtype
        self.indexes = mpool2d.indexes

    def forward(self, gy):
        xp = cuda.get_array_module(gy)

        N, C, OH, OW = gy.shape
        N, C, H, W = self.input_shape
        KH, KW = pair(self.kernel_size)

        gcol = xp.zeros((N * C * OH * OW * KH * KW), dtype=self.dtype)

        indexes = (self.indexes.ravel()
                   + xp.arange(0, self.indexes.size * KH * KW, KH * KW))

        gcol[indexes] = gy.ravel()
        gcol = gcol.reshape(N, C, OH, OW, KH, KW)
        gcol = xp.swapaxes(gcol, 2, 4)
        gcol = xp.swapaxes(gcol, 3, 5)

        gx = col2im_array(gcol, (N, C, H, W), self.kernel_size, self.stride,
                          self.pad, to_matrix=False)
        return gx

    def backward(self, ggx):
        f = Pooling2DWithIndexes(self.mpool2d)
        return f(ggx)
```

### Pooling2DWithIndexes（L240–257）

```python
class Pooling2DWithIndexes(Function):
    def __init__(self, mpool2d):
        self.kernel_size = mpool2d.kernel_size
        self.stride = mpool2d.stride
        self.pad = mpool2d.pad
        self.input_shape = mpool2d.inputs[0].shape
        self.dtype = mpool2d.inputs[0].dtype
        self.indexes = mpool2d.indexes

    def forward(self, x):
        col = im2col_array(x, self.kernel_size, self.stride, self.pad,
                           to_matrix=False)
        N, C, KH, KW, OH, OW = col.shape
        col = col.reshape(N, C, KH * KW, OH, OW)
        col = col.transpose(0, 1, 3, 4, 2).reshape(-1, KH * KW)
        indexes = self.indexes.ravel()
        col = col[np.arange(len(indexes)), indexes]
        return col.reshape(N, C, OH, OW)
```

### Im2col（L298–316）

```python
class Im2col(Function):
    def __init__(self, kernel_size, stride, pad, to_matrix):
        super().__init__()
        self.input_shape = None
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.to_matrix = to_matrix

    def forward(self, x):
        self.input_shape = x.shape
        y = im2col_array(x, self.kernel_size, self.stride, self.pad,
                         self.to_matrix)
        return y

    def backward(self, gy):
        gx = col2im(gy, self.input_shape, self.kernel_size, self.stride,
                    self.pad, self.to_matrix)
        return gx
```

### Col2im（L349–366）

```python
class Col2im(Function):
    def __init__(self, input_shape, kernel_size, stride, pad, to_matrix):
        super().__init__()
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.to_matrix = to_matrix

    def forward(self, x):
        y = col2im_array(x, self.input_shape, self.kernel_size, self.stride,
                         self.pad, self.to_matrix)
        return y

    def backward(self, gy):
        gx = im2col(gy, self.kernel_size, self.stride, self.pad,
                    self.to_matrix)
        return gx
```

### im2col_array（L376–402）

```python
def im2col_array(img, kernel_size, stride, pad, to_matrix=True):
    N, C, H, W = img.shape
    KH, KW = pair(kernel_size)
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    xp = cuda.get_array_module(img)
    if xp != np:
        col = _im2col_gpu(img, kernel_size, stride, pad)
    else:
        img = np.pad(img,
                     ((0, 0), (0, 0), (PH, PH + SH - 1), (PW, PW + SW - 1)),
                     mode='constant', constant_values=(0,))
        col = np.ndarray((N, C, KH, KW, OH, OW), dtype=img.dtype)

        for j in range(KH):
            j_lim = j + SH * OH
            for i in range(KW):
                i_lim = i + SW * OW
                col[:, :, j, i, :, :] = img[:, :, j:j_lim:SH, i:i_lim:SW]

    if to_matrix:
        col = col.transpose((0, 4, 5, 1, 2, 3)).reshape((N * OH * OW, -1))

    return col
```

### conv2d_cpu_tensordot_array（L405–413）

```python
def conv2d_cpu_tensordot_array(x, W, b, stride=1, pad=0, xp=np):
    """Legacy CPU/GPU conv path kept for benchmark/reference comparison."""
    KH, KW = W.shape[2:]
    col = im2col_array(x, (KH, KW), stride, pad, to_matrix=False)
    y = xp.tensordot(col, W, ((1, 2, 3), (1, 2, 3)))
    if b is not None:
        y += b
    y = xp.rollaxis(y, 3, 1)
    return y
```

### conv2d_cpu_im2col_gemm_array（L416–439）

```python
def conv2d_cpu_im2col_gemm_array(x, W, b, stride=1, pad=0):
    """CPU conv path: im2col + GEMM.

    Convolution is rewritten as matrix multiplication by unfolding each sliding
    window to one row (im2col), then multiplying with reshaped weights.

    The GEMM call is centralized so we can replace only this backend with a
    faster kernel later. Packing is intentionally not required in this stage.
    """
    N, C, H, W_in = x.shape
    OC, C_w, KH, KW = W.shape
    if C != C_w:
        raise ValueError("input channels mismatch: x.shape={}, W.shape={}".format(x.shape, W.shape))

    SH, SW = pair(stride)
    PH, PW = pair(pad)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W_in, KW, SW, PW)

    col = im2col_array(x, (KH, KW), stride, pad, to_matrix=True)
    W_matrix = W.reshape(OC, -1).transpose(1, 0)
    y_matrix = cpu_gemm(col, W_matrix, bias=b)
    y = y_matrix.reshape(N, OH, OW, OC).transpose(0, 3, 1, 2)
    return y
```

### col2im_array（L442–465）

```python
def col2im_array(col, img_shape, kernel_size, stride, pad, to_matrix=True):
    N, C, H, W = img_shape
    KH, KW = pair(kernel_size)
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    if to_matrix:
        col = col.reshape(N, OH, OW, C, KH, KW).transpose(0, 3, 4, 5, 1, 2)

    xp = cuda.get_array_module(col)
    if xp != np:
        img = _col2im_gpu(col, SH, SW, PH, PW, H, W)
        return img
    else:
        img = np.zeros((N, C, H + 2 * PH + SH - 1, W + 2 * PW + SW - 1),
                       dtype=col.dtype)
        for j in range(KH):
            j_lim = j + SH * OH
            for i in range(KW):
                i_lim = i + SW * OW
                img[:, :, j:j_lim:SH, i:i_lim:SW] += col[:, :, j, i, :, :]
        return img[:, :, PH:H + PH, PW:W + PW]
```

### _im2col_gpu（L468–504）

```python
def _im2col_gpu(img, kernel_size, stride, pad):
    """im2col function for GPU.
    This code is ported from Chainer:
    https://github.com/chainer/chainer/blob/v6.4.0/chainer/utils/conv.py
    """
    n, c, h, w = img.shape
    kh, kw = pair(kernel_size)
    sy, sx = pair(stride)
    ph, pw = pair(pad)
    out_h = get_conv_outsize(h, kh, sy, ph)
    out_w = get_conv_outsize(w, kw, sx, pw)
    dy, dx = 1, 1
    col = cuda.cupy.empty((n, c, kh, kw, out_h, out_w), dtype=img.dtype)

    cuda.cupy.ElementwiseKernel(
        'raw T img, int32 h, int32 w, int32 out_h, int32 out_w,'
        'int32 kh, int32 kw, int32 sy, int32 sx, int32 ph, int32 pw,'
        'int32 dy, int32 dx',
        'T col',
        '''
           int c0 = i / (kh * kw * out_h * out_w);
           int ky = i / (kw * out_h * out_w) % kh;
           int kx = i / (out_h * out_w) % kw;
           int out_y = i / out_w % out_h;
           int out_x = i % out_w;
           int in_y = ky * dy + out_y * sy - ph;
           int in_x = kx * dx + out_x * sx - pw;
           if (in_y >= 0 && in_y < h && in_x >= 0 && in_x < w) {
             col = img[in_x + w * (in_y + h * c0)];
           } else {
             col = 0;
           }
        ''',
        'im2col')(img.reduced_view(),
                  h, w, out_h, out_w, kh, kw, sy, sx, ph, pw, dy, dx, col)

    return col
```

### _col2im_gpu（L507–544）

```python
def _col2im_gpu(col, sy, sx, ph, pw, h, w):
    """col2im function for GPU.
    This code is ported from Chainer:
    https://github.com/chainer/chainer/blob/v6.4.0/chainer/utils/conv.py
    """
    n, c, kh, kw, out_h, out_w = col.shape
    dx, dy = 1, 1
    img = cuda.cupy.empty((n, c, h, w), dtype=col.dtype)

    cuda.cupy.ElementwiseKernel(
        'raw T col, int32 h, int32 w, int32 out_h, int32 out_w,'
        'int32 kh, int32 kw, int32 sy, int32 sx, int32 ph, int32 pw,'
        'int32 dx, int32 dy',
        'T img',
        '''
           int c0 = i / (h * w);
           int y  = i / w % h;
           int x  = i % w;
           T val = 0;
           for (int ky = 0; ky < kh; ++ky) {
             int out_y = (y + ph - ky * dy);
             if (0 > out_y || out_y >= out_h * sy) continue;
             if (out_y % sy != 0) continue;
             out_y /= sy;
             for (int kx = 0; kx < kw; ++kx) {
               int out_x = (x + pw - kx * dx);
               if (0 > out_x || out_x >= out_w * sx) continue;
               if (out_x % sx != 0) continue;
               out_x /= sx;
               int k = out_y + out_h * (kx + kw * (ky + kh * c0));
               val = val + col[out_x + out_w * k];
             }
           }
           img = val;
        ''',
        'col2im')(col.reduced_view(),
                  h, w, out_h, out_w, kh, kw, sy, sx, ph, pw, dx, dy, img)
    return img
```

## hytorch/core.py

完整 imports 和上下文见 `source/hytorch/core.py`。

### Config（L11–13）

```python
class Config:
    enable_backprop = True     # 有的时候不需要反向传播，只需要正向传递，这时候修改配置configuration
    train = True               # 在使用dropout的情况下，需要区分训练阶段和测试阶段
```

### using_config（L17–23）

```python
def using_config(name, value):
    old_value = getattr(Config, name)  # 把 Config 类（或对象）里当前 name 对应的值保存起来，以便以后恢复。
    setattr(Config, name, value)    # 把同一属性改成新值 value，进入「临时模式」
    try:
        yield                       # 交出执行权，with 语句体就在这行开始执行。
    finally:
        setattr(Config, name, old_value)  # 无论 with 体内是否抛异常，都保证把属性恢复成旧值。
```

### Variable（L44–163）

```python
class Variable:
    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, array_types):      # 加入GPU，将ndarray改为array_types
                raise TypeError('{} is not supported'.format(type(data)))
                # def的function用了as_array,Function中as_variable传入的是array
        self.data = data  # data的类型是ndarray/cparray，属性有shape,nidm,size,dtype,itemsize等
        self.name = name  # 命名变量
        self.grad = None
        self.creator = None
        self.generation = 0

    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n',
                                   '\n' + ' ' * 9)  # ndarray变成字符串，自带换行：把换行替换成换行加 9 空格，让数组的每一行（除首行）向右缩进，variable是9个字母
        return 'variable(' + p + ')'

    def __len__(self):  # 对ndarray使用len，会返回第一个维度的元素数量
        return len(self.data)

    @property  # shape方法可以作为实例变量被访问，即用x.shape替代x.shape()
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):  # 维度
        return self.data.ndim

    @property
    def size(self):  # 元素数
        return self.data.size

    @property
    def dtype(self):  # 数据类型
        return self.data.dtype

    @property
    def T(self):
        # from hytorch.functions import transpose     ?为什么前面的import dezero不行,此时的__init__里面还没有functions
        return hytorch.functions.transpose(self)

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    '''类内定义实现运算符重载
    def __mul__(self,other):
        return mul(self,other)
    def __rmul__......
    '''

    def cleargrad(self):
        self.grad = None

    def reshape(self, *shape):   # 实现从Variable对象中调用reshape
        # 个 if 是为了让你可以像 NumPy 一样，既可以直接传多个参数，也可以传一个元组/列表，
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        # from hytorch.functions import reshape
        return hytorch.functions.reshape(self, shape)

    def sum(self, axis=None, keepdims=False):   # 实现从Variable对象中调用sum
        from hytorch.functions import sum         # ?
        return sum(self, axis, keepdims)

    def to_cpu(self):   # 实现数据从gpu传输到cpu
        if self.data is not None:
            self.data = hytorch.cuda.as_numpy(self.data)

    def to_gpu(self):   # 实现数据从cpu传输到gpu
        if self.data is not None:
            self.data = hytorch.cuda.as_cupy(self.data)

    def backward1(self):
        f = self.creator
        if f is not None:
            x = f.input
            x.grad = f.backward(self.grad)
            x.backward()

    def backward(self, retain_grad=False, create_graph=False):  # 不保留不必要的grad, 默认不要反向传播的计算图
        if self.grad is None:
            xp = hytorch.cuda.get_array_module(self.data)
            self.grad = Variable(xp.ones_like(self.data))    # 计算高阶导数，Variable类只修改一处，将grad也变为Variable，可以构建计算图
        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]     # output是weak连接，要用（）
            # create_backprop为True时，反向传播时Function里面才设置辈分和连接，结束后恢复False
            with using_config('enable_backprop', create_graph):
                gxs = f.backward(*gys)
                if not isinstance(gxs, tuple):
                    gxs = (gxs,)

                for x, gx in zip(f.inputs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad = gx + x.grad  # 不能用+=，gx 和 x.grad 指向同一块内存，导致 += 变成“自己加自己

                    if x.creator is not None:
                        add_func(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None  # y是weakref

    __array_priority__ = 200  # 解决左项为ndarray的问题，Variable的优先级高于ndarray，此时调用rmul或radd
```

### as_array（L170–173）

```python
def as_array(x, array_module=np):    # cpu->gpu
    if np.isscalar(x):
        return array_module.array(x)
    return x
```

### as_variable（L176–179）

```python
def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)
```

### Function（L182–205）

```python
class Function:
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]  # Variable和ndarray一起使用

        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:  # 需要反向传播：设置“辈分” ,默认为True
            self.generation = max([x.generation for x in inputs])  # generation要在set_creator前面
            for output in outputs:  # 设置“连接”
                output.set_creator(self)
        self.inputs = inputs
        self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, x):            # forward用的是ndarray，可以自动广播
        raise NotImplementedError()

    def backward(self, gy):           # 如果进行了广播，反向传播要手动sum_to
        raise NotImplementedError()
```

### Mul（L249–261）

```python
class Mul(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0*x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0, gx1 = x1*gy, x0*gy
        if self.x0_shape != self.x1_shape:
            gx0 = hytorch.functions.sum_to(gx0, self.x0_shape)     # 如果在正向传播中进行了广播，此时应该进行广播的反向传播
            gx1 = hytorch.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1
```

### mul（L264–266）

```python
def mul(x0, x1):
    x1 = as_array(x1, hytorch.cuda.get_array_module(x0.data))
    return Mul()(x0, x1)
```

### Sub（L281–292）

```python
class Sub(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0-x1
        return y

    def backward(self, gy):
        gx0, gx1 = gy, -gy
        if self.x0_shape != self.x1_shape:
            gx0 = hytorch.functions.sum_to(gx0, self.x0_shape)     # 如果在正向传播中进行了广播，此时应该进行广播的反向传播
            gx1 = hytorch.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1
```

### sub（L295–297）

```python
def sub(x0, x1):
    x1 = as_array(x1, hytorch.cuda.get_array_module(x0.data))
    return Sub()(x0, x1)
```

### rsub（L300–302）

```python
def rsub(x0, x1):         # 如2.0-x，x->self(x0),2.0->other(x1)
    x1 = as_array(x1, hytorch.cuda.get_array_module(x0.data))
    return Sub()(x1, x0)  # 非对称的，不回退，想支持 scalar - var 就必须自己写 __rsub__。
```

### setup_variable（L350–366）

```python
def setup_variable():
    Variable.__radd__ = add
    Variable.__add__ = add  # 类外赋值实现运算符重载,radd解决int或float在运算符左边的问题，即在self位置上
    Variable.__rmul__ = mul
    Variable.__mul__ = mul  # 类外赋值实现运算符重载,rmul解决int或float在运算符左边的问题，即在self位置上
    Variable.__neg__ = neg
    Variable.__rsub__ = rsub
    Variable.__sub__ = sub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow
    Variable.__getitem__ = hytorch.functions.get_item

    Variable.matmul = hytorch.functions.matmul
    Variable.dot = hytorch.functions.matmul
    Variable.max = hytorch.functions.max
    Variable.min = hytorch.functions.min
```

## hytorch/functions.py

完整 imports 和上下文见 `source/hytorch/functions.py`。

### Tanh（L47–56）

```python
class Tanh(Function):
    def forward(self, x):   # 变量是ndarray实例，用的是Variable.data->gpu化，用get_array_module获取数据类型
        xp = cuda.get_array_module(x)
        y = xp.tanh(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()       # weakref()
        gx = gy*(1 - y*y)
        return gx
```

### Sum（L194–207）

```python
class Sum(Function):
    def __init__(self, axis, keepdims):   # axis=0,行相加，axis=1， 列相加。keepdims=False，y为标量，反之保留轴的数量
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy):
        gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        gx = broadcast_to(gy, self.x_shape)    # 广播机制，使x.shape = gx.shape
        return gx
```

### SumTo（L235–246）

```python
class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = utils.sum_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = broadcast_to(gy, self.x_shape)     # sum_to的反向传播是broadcast_to,两者相互依赖
        return gx
```

### sum_to（L249–252）

```python
def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)
```

### MatMul（L263–279）

```python
class MatMul(Function):
    def __init__(self, native=True):
        self.native = native

    def forward(self, x, w):
        from hytorch.backends.cuda.runtime import selected, gemm, current_config
        mode = selected("matmul", x) if self.native else None
        if mode is not None:
            return gemm(x, w, variant=mode, tile=current_config()["tile"])
        y = x.dot(w)   # 矩阵的乘法运算，其中包括：向量内积、多维矩阵乘法、矩阵与向量的乘法。
        return y

    def backward(self, gy):
        x, w = self.inputs
        gx = MatMul(native=False)(gy, w.T)
        gw = MatMul(native=False)(x.T, gy)    # Preserve existing backward backend.
        return gx, gw
```

### matmul（L282–283）

```python
def matmul(x, w):
    return MatMul()(x, w)
```

### linear_simple（L317–323）

```python
def linear_simple(x, W, b=None):
    t = matmul(x, W)
    if b is None:
        return t
    y = b + t
    t.data = None       # matmul的反向传播并不需要t的值，引用次数变成0，被python解释器删除
    return y
```

### Linear（L326–341）

```python
class Linear(Function):
    def forward(self, x, W, b):
        from hytorch.backends.cuda.runtime import selected, gemm, current_config
        mode = selected("matmul", x)
        y = (x.dot(W) if mode is None else
             gemm(x, W, variant=mode, tile=current_config()["tile"]))
        if b is not None:
            y += b
        return y

    def backward(self, gy):
        x, W, b = self.inputs
        gb = None if b.data is None else sum_to(gy, b.shape)
        gx = MatMul(native=False)(gy, W.T)
        gW = MatMul(native=False)(x.T, gy)
        return gx, gW, gb
```

### ReLU（L374–384）

```python
class ReLU(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.maximum(0.0, x)
        return y

    def backward(self, gy):
        x, = self.inputs
        mask = x.data > 0
        gx = gy * mask
        return gx
```

### MeanSquareError（L469–480）

```python
class MeanSquareError(Function):
    def forward(self, x0, x1):
        diff = x0 - x1
        y = (diff**2).sum()/len(diff)
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        diff = x0 - x1
        gx0 = gy * diff*(2./len(diff))
        gx1 = -gx0
        return gx0, gx1
```

## hytorch/optimizers.py

完整 imports 和上下文见 `source/hytorch/optimizers.py`。

### Optimizer（L9–36）

```python
class Optimizer:
    def __init__(self):
        self.target = None
        self.hooks = []

    def setup(self, target):
        self.target = target    # model
        return self

    def update(self):
        params = [p for p in self.target.params() if p.grad is not None]
        # 将None之外的参数汇总到列表
        # 预处理（可选）
        for f in self.hooks:
            f(params)

        # FreezeParam hooks can clear gradients after the first filter.
        params = [p for p in params if p.grad is not None]

        # 更新参数
        for param in params:
            self.update_one(param)

    def update_one(self, param):
        raise NotImplementedError

    def add_hook(self, f):
        self.hooks.append(f)
```

### SGD（L86–92）

```python
class SGD(Optimizer):
    def __init__(self, lr=0.01):
        super().__init__()
        self.lr = lr

    def update_one(self, param):  # 更新权重和偏移
        param.data -= self.lr * param.grad.data
```

## hytorch/profiling.py

完整 imports 和上下文见 `source/hytorch/profiling.py`。

### StepProfiler（L94–157）

```python
class StepProfiler:
    """Synchronized wall timing for diagnosis; boundary-only mode for throughput.

    Phase mode serializes phase boundaries deliberately. Step includes loading,
    transfer, forward+loss, cleargrad+backward, update, and unassigned Python work.
    JSON writing and metric conversion occur outside step timing. Use separate
    throughput runs to quantify observer overhead.
    """
    def __init__(self, path, synchronize=lambda: None, mode="phases", nvtx=False):
        if mode not in ("phases", "throughput"):
            raise ValueError("mode must be phases or throughput")
        self.path, self.sync, self.mode, self.nvtx = Path(path), synchronize, mode, nvtx
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows = []
        self.phase_times = {}

    def begin(self):
        self.sync()
        self.started = time.perf_counter()
        self.timestamp = time.time()
        self.phase_times = {}

    @contextmanager
    def phase(self, name):
        if self.mode == "phases":
            self.sync()
        start = time.perf_counter()
        if self.nvtx:
            import cupy as cp
            cp.cuda.nvtx.RangePush(name)
        try:
            yield
        finally:
            if self.mode == "phases":
                self.sync()
                self.phase_times[name + "_s"] = time.perf_counter() - start
            if self.nvtx:
                cp.cuda.nvtx.RangePop()

    def end(self, samples, epoch, step):
        self.sync()
        elapsed = time.perf_counter() - self.started
        phases = ("data", "transfer", "forward", "backward", "update")
        row = {"timestamp": self.timestamp, "epoch": epoch, "step": step,
               "samples": samples, "mode": self.mode, "step_s": elapsed,
               "samples_per_sec": samples/elapsed}
        row.update({p + "_s": self.phase_times.get(p + "_s") for p in phases})
        row["other_s"] = (max(0., elapsed-sum(self.phase_times.values()))
                          if self.mode == "phases" else None)
        self.rows.append(row)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, allow_nan=False) + "\n")
        return row

    def summary(self):
        total = sum(r["step_s"] for r in self.rows)
        result = {"steps": len(self.rows), "measured_step_total_s": total,
                  "mean_step_s": total/len(self.rows) if self.rows else None,
                  "samples_per_sec": sum(r["samples"] for r in self.rows)/total if total else None,
                  "mode": self.mode, "phase_fraction": {}}
        if total and self.mode == "phases":
            for key in ("data", "transfer", "forward", "backward", "update", "other"):
                result["phase_fraction"][key] = sum(r[key+"_s"] for r in self.rows)/total
        return result
```

## hytorch/layers.py

完整 imports 和上下文见 `source/hytorch/layers.py`。

### Linear（L90–120）

```python
class Linear(Layer):
    def __init__(self, out_size, nobias=False, dtype=np.float32, in_size=None):
        super().__init__()    # Linear.__init__ 里显式调用 Layer.__init__，让父类有机会初始化 _params 集合
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype

        self.W = Parameter(None, name='W')
        if self.in_size is not None:     # 如果没有指定in_size，则延后处理
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_size, dtype=dtype), name='b')

    def _init_W(self, xp=np):
        I, O = self.in_size, self.out_size
        W_data = xp.random.randn(I, O).astype(self.dtype)*np.dtype(self.dtype).type(np.sqrt(1/I))
        # “把权重初始化为均值为 0、方差为 1/I 的高斯随机数，保证前向传播时方差恒为 1，避免梯度爆炸/消失。”
        self.W.data = W_data

    def forward(self, x):
        # 在传播数据是初始化权重
        if self.W.data is None:
            self.in_size = x.shape[1]
            xp = cuda.get_array_module(x)
            self._init_W(xp)

        y = F.linear_simple(x, self.W, self.b)
        return y
```

### Conv2d（L135–181）

```python
class Conv2d(Layer):
    def __init__(self, out_channels, kernel_size, stride=1,
                 pad=0, nobias=False, dtype=np.float32, in_channels=None):
        """Two-dimensional convolutional layer.

        Args:
            out_channels (int): Number of channels of output arrays.
            kernel_size (int or (int, int)): Size of filters.
            stride (int or (int, int)): Stride of filter applications.
            pad (int or (int, int)): Spatial padding width for input arrays.
            nobias (bool): If `True`, then this function does not use the bias.
            in_channels (int or None): Number of channels of input arrays. If
            `None`, parameter initialization will be deferred until the first
            forward data pass at which time the size will be determined.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.dtype = dtype

        self.W = Parameter(None, name='W')
        if in_channels is not None:
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_channels, dtype=dtype), name='b')

    def _init_W(self, xp=np):
        C, OC = self.in_channels, self.out_channels
        KH, KW = pair(self.kernel_size)
        scale = np.dtype(self.dtype).type(np.sqrt(1 / (C * KH * KW)))
        W_data = xp.random.randn(OC, C, KH, KW).astype(self.dtype) * scale
        self.W.data = W_data

    def forward(self, x):
        if self.W.data is None:
            self.in_channels = x.shape[1]
            xp = cuda.get_array_module(x)
            self._init_W(xp)

        y = F.conv2d(x, self.W, self.b, self.stride, self.pad)
        return y
```
