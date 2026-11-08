# 最终源码状态

Git HEAD: `edb6fbe3bfaa9eed461addd29e6943b22e28216e`。工作区已有修改完整保留，详见 raw/git_status_before.txt 与 git_diff_before.patch。

本评测不改生产代码；source/ 保存生产源码与现有测试，raw/source_hashes_before.json 保存逐文件 SHA-256 和三个动态库的路径/SHA。动态库不打包。

Native 显式启用 matmul=shared、conv=im2col、pool=native、col2im=native、dweight=packed、metadata_cache=True (F1)、compact_function=True (O1)、tile=16。CuPy 路径全部 cupy；F1/O1 仍启用。SGD 未融合。TF32 由 NVIDIA_TF32_OVERRIDE=0 和 PyTorch flags 关闭；现有构建命令无 fast-math。

结束后逐一复核 73 个源码/测试/动态库条目，SHA 全部不变，Git status 与执行前逐字一致。仅新增被 Git ignore 的 runs/ 评测产物。
