from pathlib import Path
import sys
sys.path.insert(0,str(Path.cwd()))
import ast,hashlib,json,shutil,difflib,subprocess,platform,importlib.metadata,ctypes as ct,csv
R=Path.cwd();O=R/'runs/a100_20261106/final_training_optimization'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
changed=['hytorch/backends/cuda/runtime.py','hytorch/backends/cuda/kernels.cu','hytorch/functions_conv.py','hytorch/optimizers.py','tests/test_final_optimization.py'];patch=''
for f in changed:
 base=O/'baseline'/f;current=R/f
 patch+=''.join(difflib.unified_diff(base.read_bytes().decode().splitlines(True) if base.exists() else [],current.read_bytes().decode().splitlines(True),fromfile='a/'+f if base.exists() else '/dev/null',tofile='b/'+f))
(O/'git_diff.patch').write_text(patch)
checks={}
checks['existing_cuda_exact_prefix']=(R/'hytorch/backends/cuda/kernels.cu').read_bytes().startswith((O/'baseline/hytorch/backends/cuda/kernels.cu').read_bytes())
def defs(p):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(p.read_bytes()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
for f,names in [('hytorch/backends/cuda/runtime.py',['gemm','conv2d','col2im','_col2im_metadata','_arrays','max_pool2d']),('hytorch/functions_conv.py',['Conv2d','Deconv2d','col2im_array']),('hytorch/optimizers.py',['Optimizer','MomentumSGD','Adam','AdamW','WeightDecay','ClipGrad','FreezeParam'])]:
 a,b=defs(O/'baseline'/f),defs(R/f)
 for name in names:checks[f+':'+name]=a[name]==b[name]
for f in ['hytorch/functions.py','hytorch/core.py']:checks[f]=sha(O/'baseline'/f)==sha(R/f)
checks['original_tests_unchanged']=all(sha(p)==sha(R/p.relative_to(O/'baseline')) for p in (O/'baseline/tests').glob('*.py'))
assert all(checks.values());(O/'raw/scope_checks.json').write_text(json.dumps(checks,indent=2))
for pattern in ['hytorch/**/*.py','hytorch/**/*.cu','tests/*.py','scripts/*.py','benchmarks/*.py','donkeycar_hy/*.py']:
 for p in R.glob(pattern):
  if 'build' in p.parts:continue
  dst=O/'source'/p.relative_to(R);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst)
for f in ['libhytorch_cuda.so','build.json','build.log']:
 p=O/'final_build'/f;p.parent.mkdir(exist_ok=True);shutil.copyfile(R/'hytorch/backends/cuda/build'/f,p)
for p in (R/'runs/a100_20261106/native_col2im/raw').glob('previous_gemm*'):shutil.copyfile(p,O/'raw'/p.name)
for f in ['baseline_gpu.log','baseline_host.log','baseline_dtype_host.log']:
 src=R/'runs/a100_20261106/dweight_im2col_reuse/raw'/f
 if src.exists():shutil.copyfile(src,O/'baseline'/('previous_window_'+f))
from hytorch.backends.cuda.runtime import _SgdTensor
assert ct.sizeof(_SgdTensor)==24 and ct.alignment(_SgdTensor)==8
layout={f:getattr(_SgdTensor,f).offset for f in ['p','g','n']};assert layout==dict(p=0,g=8,n=16)
(O/'ENVIRONMENT.json').write_text(json.dumps(dict(python=platform.python_version(),platform=platform.platform(),packages={k:importlib.metadata.version(k) for k in ['numpy','cupy-cuda12x','torch','pytest','matplotlib']},sgd_descriptor=dict(size=ct.sizeof(_SgdTensor),alignment=ct.alignment(_SgdTensor),offsets=layout),GPU='physical1 A100-PCIE-40GB',user_timezone='Asia/Shanghai',build=json.loads((O/'final_build/build.json').read_text()),torch=json.loads((O/'raw/torch_config.json').read_text()),torch_observed_dispatch=json.loads((O/'raw/torch_dispatch.json').read_text())),indent=2))
sha_files=[p for folder in ['baseline','source','final_build'] for p in (O/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]+[O/'ACCEPTANCE.md',O/'git_diff.patch']
(O/'SHA.json').write_text(json.dumps({str(p.relative_to(O)):sha(p) for p in sha_files},indent=2))
(O/'raw/git_status_final.txt').write_bytes(subprocess.check_output(['git','status','--short']));(O/'raw/head.txt').write_bytes(subprocess.check_output(['git','rev-parse','HEAD']));subprocess.run(['git','diff','--check'],check=True)
# Explicit mode status alongside original unmodified raw rows.
r=list(csv.DictReader((O/'TRAINING.csv').open()))
for x in r:x.update(dweight_actual='packed' if x['mode'] in ['D','DS'] else 'cupy',sgd_requested='native' if x['mode'] in ['S','DS'] else 'legacy',sgd_actual='legacy',sgd_candidate_status='REJECTED_subnormal_FTZ_mismatch')
with (O/'TRAINING_WITH_STATUS.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(r[0]));w.writeheader();w.writerows(r)
print('scope checks and SHA completed')
