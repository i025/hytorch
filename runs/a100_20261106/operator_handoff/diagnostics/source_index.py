from pathlib import Path
import ast,json,hashlib
root=Path.cwd();out=root/'runs/a100_20261106/operator_handoff'
paths=[r['path'] for r in json.loads((out/'source_manifest.json').read_text())]
lines=['# 核心源码交付与调用入口','', '源码为本轮开始时的逐字节快照，保持相对路径。行号均指快照中的真实文件；本轮未修改生产代码。', '', '完整文件均在 `source/`；下方附可直接读取的关键函数完整函数体，不以省略号替代。CuPy cuBLAS 的底层实现为已安装二进制，未伪造其源码。', '', '## 文件与符号索引','', '| 文件 | 符号及行号 |','|---|---|']
for name in paths:
 p=out/'source'/name
 if p.suffix=='.py':
  tree=ast.parse(p.read_text());syms=[]
  for node in tree.body:
   if isinstance(node,(ast.ClassDef,ast.FunctionDef)):
    syms.append(f'`{node.name}` L{node.lineno}–{node.end_lineno}')
  lines.append(f'| [{name}](source/{name}) | '+ '; '.join(syms)+' |')
 else:lines.append(f'| [{name}](source/{name}) | 完整文件 |')
lines += ['', '## 关键调用关系', '',
'1. `benchmarks/bench_cuda_training.py:124 main` → `DonkeyNet.forward` → `layers.Conv2d.forward` → `functions_conv.Conv2d.forward`。',
'2. native 前向：`runtime.selected` → `runtime.conv2d` → `_call(hy_im2col)` → `runtime.gemm` → `_arrays` 连续化权重 → `_call(hy_gemm)` → `_call(hy_conv_pack)`。',
'3. `Variable.backward` → `Conv2d.backward`，其调用顺序为 dInput、dWeight、dBias。dInput 使用 `Deconv2d.forward` 的 CuPy tensordot/rollaxis/col2im；dWeight 使用 `Conv2DGradW.forward` 的 CuPy im2col/tensordot；dBias 使用 `Sum.forward`。',
'4. 实际全连接层调用 `F.linear_simple`，不是 `F.Linear`：`MatMul.forward` → native dispatch，随后 Add；`MatMul.backward` 显式用 `MatMul(native=False)` 回到 CuPy dot。两种 Linear API 都提供完整源码，避免混淆。',
'5. 标量类型路径：`MeanSquareError.backward` → `core.mul` → `as_array` → `cupy.array(2./len(diff))`，实测标量为 FP64；`Tanh.backward` 的 `1-y*y` 经 `rsub/as_array` 生成 int64 标量，实测也把此表达式提升为 FP64。',
'6. `Pooling.forward` 支持 native，但 backward 的 scatter/col2im 仍为 CuPy；DonkeyNet 本轮实际调用次数为 0。',
'7. 绑定是 Python ctypes + `extern "C"` CUDA ABI，无额外 C++ binding 层。库本身不分配、不传输、不主动同步；工作区由 CuPy 管理，使用当前 stream 指针。',
'8. 数据 loader 在 CPU/NumPy 组 batch；每步 H2D；`StepProfiler` 同步边界；每步 scalar(loss) D2H；SGD Python 循环逐参数更新。', '']
for name in ['hytorch/backends/cuda/kernels.cu','hytorch/backends/cuda/runtime.py']:
 text=(out/'source'/name).read_text()
 lines += ['## '+name+'（完整文件）','', '```'+('cuda' if name.endswith('.cu') else 'python'),text.rstrip(),'```','']
selected={
'hytorch/functions_conv.py':['Conv2d','Deconv2d','Conv2DGradW','Pooling','Pooling2DGrad','Pooling2DWithIndexes','Im2col','Col2im','im2col_array','col2im_array','conv2d_cpu_tensordot_array','conv2d_cpu_im2col_gemm_array','_im2col_gpu','_col2im_gpu'],
'hytorch/core.py':['Config','using_config','Variable','Function','as_array','as_variable','Mul','mul','Sub','sub','rsub','setup_variable'],
'hytorch/functions.py':['MatMul','matmul','linear_simple','Linear','Tanh','ReLU','MeanSquareError','Sum','SumTo','sum_to'],
'hytorch/optimizers.py':['Optimizer','SGD'],
'hytorch/profiling.py':['StepProfiler'],
'hytorch/layers.py':['Linear','Conv2d'],
}
for name,names in selected.items():
 text=(out/'source'/name).read_text();src=text.splitlines();tree=ast.parse(text)
 lines += ['## '+name,'','完整 imports 和上下文见 `source/'+name+'`。','']
 for node in tree.body:
  if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in names:
   lines += [f'### {node.name}（L{node.lineno}–{node.end_lineno}）','','```python','\n'.join(src[node.lineno-1:node.end_lineno]),'```','']
(out/'OPERATOR_SOURCE.md').write_text('\n'.join(lines))
# Stable source package manifest includes external CuPy implementation.
files=[]
for p in sorted((out/'source').rglob('*')):
 if p.is_file():files.append(dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(out/'SOURCE_FILES.json').write_text(json.dumps(files,indent=2))
print('Copied source files:',len(files),'Operator markdown bytes:',(out/'OPERATOR_SOURCE.md').stat().st_size)
