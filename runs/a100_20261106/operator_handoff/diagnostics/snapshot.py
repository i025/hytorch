from pathlib import Path
import subprocess,hashlib,json,shutil
root=Path.cwd();out=root/'runs/a100_20261106/operator_handoff'
files=['README.md','requirements-cuda.txt','donkeycar_hy/model.py','donkeycar_hy/tub_dataset.py','benchmarks/bench_cuda_training.py','benchmarks/bench_cuda_gemm.py','benchmarks/bench_cuda_ops.py','benchmarks/bench_cuda_revisions.py','scripts/build_cuda.py','scripts/validate_cuda.py','scripts/report_cuda.py','scripts/run_cuda_comparison.py','hytorch/__init__.py','hytorch/core.py','hytorch/functions.py','hytorch/functions_conv.py','hytorch/layers.py','hytorch/cuda.py','hytorch/optimizers.py','hytorch/dataloaders.py','hytorch/profiling.py','hytorch/utils.py','hytorch/backends/cuda/kernels.cu','hytorch/backends/cuda/runtime.py','hytorch/backends/cuda/__init__.py','hytorch/backends/cpu/gemm.py']
files += [str(p) for p in Path('tests').glob('*.py')]
info=[]
for name in files:
 p=root/name;q=out/'source'/name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
 info.append(dict(path=name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(out/'source_manifest.json').write_text(json.dumps(info,indent=2))
for name,cmd in [('git_status.txt',['git','status','--short']),('git_branch.txt',['git','branch','--show-current']),('git_head.txt',['git','rev-parse','HEAD']),('working_tree.patch',['git','diff','edb6fbe3bfaa9eed461addd29e6943b22e28216e'])]:
 (out/'raw'/name).write_bytes(subprocess.check_output(cmd))
for name in ['build_baseline.json','build_optimized.json','packages.txt','baseline_commands.json','remaining_commands.json','memcheck.log','racecheck.log']:
 p=root/'runs/a100_20261106'/name
 if p.exists():shutil.copy2(p,out/'raw'/name)
for name in ['gemm_paired/comparison.csv','gemm_baseline/summary.json','donkey_profile_baseline/summary.json']:
 p=root/'runs/a100_20261106'/name;shutil.copy2(p,out/'raw'/name.replace('/','_'))
shutil.copy2(root/'outputs/a100_20261106/training_summary.json',out/'raw/prior_training_summary.json')
p=Path('/home/xieminhui_pubuse/hytorch-env/lib/python3.10/site-packages/cupy/linalg/_product.py')
q=out/'source/external/cupy/linalg/_product.py';q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
(out/'raw/kernel_baseline.cu').write_bytes(subprocess.check_output(['git','show','edb6fbe3bfaa9eed461addd29e6943b22e28216e:hytorch/backends/cuda/kernels.cu']))
print('Snapshot files',len(files))
