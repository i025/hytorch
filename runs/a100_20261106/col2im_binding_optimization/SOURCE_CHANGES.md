# 本阶段源码改动

本阶段git_diff.patch仅包含以下两处生产文件和一个新增测试；baseline/git_diff.patch保留进入阶段时工作区已有修改，禁止将后者误认为本轮修改。

1. hytorch/backends/cuda/runtime.py
   - load_library新增可选hy_col2im_fast签名[p,p,c_char_p,p]，原ABI和旧库兼容不变。
   - 新_col2im_metadata @lru_cache(maxsize=128)：以实际值key验证不变几何/strides，返回tuple+144-byte不可变CPU bytes。未命中时执行原验证；每次先operator.index，防float/int缓存key相等。
   - col2im保留逐次dtype/device/shape维数/output/alias检查；最新GPU地址、当前stream每次读取；命中缓存不保留指针。有fast符号时调用POD，缺少时旧hy_col2im回退。
2. hytorch/backends/cuda/kernels.cu
   - 只在原文件末尾追加Col2imParams及hy_col2im_fast。18×int64_t，static_assert大小144、自然对齐；memcpy从CPU bytes到aligned local，再调用原hy_col2im。参数生命周期仅覆盖同步host调用。
   - 原文件全部内容逐字节不变。col2im_strided及所有GEMM/forward/dWeight算法、block/grid、FP32语义、ABI版本1未变。
3. tests/test_col2im_binding.py：5项新增测试，见CORRECTNESS.md。

functions_conv.py未修改，FP32 functions.py未修改。默认仍cupy；无DonkeyNet shape特例。未引入pybind11或全运行时重构。诊断unchecked direct18/direct4只位于runs/.../diagnostics，生产无法选择；cache-only候选快照放在raw/cache_only_runtime.py，仅用于同进程消融基准。

POD字段顺序 n,c,h,w,kh,kw,oh,ow,sh,sw,ph,pw,s0..s5，native-endian、有符号64bit、无字段间padding；Python bytes无对齐承诺，因此C先memcpy。保留原hy_col2im参数检查、状态码、stream约定；raw C ABI客户端仍须提供有效buffers及144-byte参数区，Python负责完整几何与索引检查，契约没有降低。

附source当前必要源码和baseline对应原始源码。保存旧/新动态库及原Nsight二进制在服务器，路径和SHA见EXTERNAL_ARTIFACTS.json；压缩包不含大型二进制或权重/数据集。
