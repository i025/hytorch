from pathlib import Path
import subprocess,json,hashlib,shutil,os
R=Path(__file__).resolve().parents[4];O=Path(__file__).resolve().parents[1]
def cmd(args):return subprocess.run(args,cwd=R,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT).stdout
(O/'raw/git_status_before.txt').write_text(cmd(['git','status','--short']))
(O/'raw/git_diff_before.patch').write_text(cmd(['git','diff']))
files=list((R/'hytorch').rglob('*.py'))+list((R/'hytorch').rglob('*.cu'))+list((R/'tests').glob('*.py'))+[R/'hytorch/backends/cuda/build/libhytorch_cuda.so',R/'runs/a100_20261106/libhytorch_cuda_baseline.so',R/'runs/a100_20261106/gemm_shape_dispatch/lib_before.so']
rows=[]
for f in files:
 rows.append(dict(path=str(f.relative_to(R)),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),bytes=f.stat().st_size))
 if f.suffix!='.so':
  target=O/'source'/f.relative_to(R);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,target)
(O/'raw/source_hashes_before.json').write_text(json.dumps(rows,indent=2))
(O/'raw/build.json').write_text((R/'hytorch/backends/cuda/build/build.json').read_text())
(O/'raw/static_environment.txt').write_text(cmd(['uname','-a'])+cmd(['lscpu'])+cmd(['/usr/local/cuda/bin/nvcc','--version'])+cmd(['nsys','--version']))
(O/'SOURCE_STATUS.md').write_text('# 最终源码状态\n\nGit HEAD: `'+cmd(['git','rev-parse','HEAD']).strip()+'`。工作区已有修改完整保留，详见 raw/git_status_before.txt 与 git_diff_before.patch。\n\n本评测不改生产代码；source/ 保存生产源码与现有测试，raw/source_hashes_before.json 保存逐文件 SHA-256 和三个动态库的路径/SHA。动态库不打包。\n\nNative 显式启用 matmul=shared、conv=im2col、pool=native、col2im=native、dweight=packed、metadata_cache=True (F1)、compact_function=True (O1)、tile=16。CuPy 路径全部 cupy；F1/O1 仍启用。SGD 未融合。TF32 由 NVIDIA_TF32_OVERRIDE=0 和 PyTorch flags 关闭；现有构建命令无 fast-math。\n')
