# 生产修改与接口

仅修改3个生产文件，新增2个测试文件。现有GEMM128 CTA选择和所有旧CUDA kernel原样保留；旧kernels.cu是新文件的完整字节前缀。FP32修复functions.py的SHA完全不变。functions_conv中所有Function类、原_im2col_gpu/_col2im_gpu、Conv2d/Deconv2d数学实现AST均不变。

## kernels.cu:196 / :234

新增 `col2im_strided` 和可选ABI-1扩展 `hy_col2im`。每输出像素一个writer；无atomic、无shared memory；256线程/block，grid最多65535块，再用64位grid-stride覆盖输出。

对于 ty=y+PH：
```
lo=max(0, ty-(OH-1)*SH)
hi=min(KH-1, ty)
first=lo+(ty-lo)%SH  # 仅非空合法区间计算；SH=1省略mod
```
循环kh=first,first+SH,...<=hi，对应oh逐次减1。kw同理。起点做一次除法，内层没有反复整除、取模或无效贡献判断；SH/SW为1时起点直接减法。按原kh、kw升序累加FP32，地址为 `n*s0+c*s1+kh*s2+kw*s3+oh*s4+ow*s5`。

六个真实strides按元素单位作为int64值参数传入，支持负stride、零stride、非单位stride和训练rollaxis视图，不将col当连续内存。输出连续NCHW，不修改输入。图像索引/offset用64位，binding检查尺寸、总元素与可达offset范围，grid-stride末次递增也防signed overflow。C ABI正常调用须遵守binding检查的元数据约束。

## runtime.py:31 / :202

`backend(..., col2im="cupy")` 增加可选末尾参数，原位置参数不变；可选值cupy/native，默认cupy。增加hy_col2im ctypes签名：2个pointer + 18个int64 + 当前stream pointer。旧ABI-1库无新symbol仍可加载，选择native时回退CuPy。

`col2im(col,img_shape,stride,pad,out=None)` 明确检查FP32、6D、几何一致、当前设备、stride单位、宽索引界限、输出shape/dtype/连续性/别名。不调用_arrays，避免其ascontiguousarray复制输入。输入geometry不合法抛ValueError；非FP32直接调用底层native helper抛TypeError。生产dispatch对非FP32自动fallback，对超出支持的索引metadata/旧库缺扩展以NotImplementedError回退。没有默默降为FP32。

该Python校验/ctypes层有主机开销；本轮保留实现，不因训练退化引入第二套优化。

## functions_conv.py:453

只在col2im_array的GPU路径加selected("col2im",col)分支；native成功返回新输出，否则沿原_col2im_gpu。CPU路径与自动求导Col2im/Im2col类不变；显式FP64继续原CuPy路径。fallback计数记录在dispatch_stats。

示例：
```python
from hytorch.backends.cuda import backend
with backend(matmul="shared", conv="im2col", pool="native", col2im="native"):
    # Existing forward/backward code.
    loss.backward()
```
默认 `col2im="cupy"` 保留，未默认启用native。

## 测试与实验

新增tests/test_col2im_cuda.py（98 GPU cases）和tests/test_col2im_host.py（1 host case）。性能/分析脚本仅在ignored runs/native_col2im/diagnostics中；benchmark的预分配CuPy kernel从**原函数AST表达式**创建，逐case与生产函数输出核对，没有修改生产CuPy数学实现。

raw/this_task.patch是本轮增量；git_diff.patch含此前已有tracked改动及本轮新增测试，不应把既有FP32/GEMM实现归为本轮新增。baseline/保存完整必要基线源码，lib_baseline.so在服务器保留；未Commit/Push/PR。
