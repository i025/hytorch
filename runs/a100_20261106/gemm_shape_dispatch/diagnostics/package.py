from pathlib import Path
import json,csv,hashlib,subprocess,tarfile,ast
ROOT=Path.cwd();O=ROOT/'runs/a100_20261106/gemm_shape_dispatch';R=O/'raw'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
h=json.loads((R/'before_hashes.json').read_text())
assert sha(ROOT/'hytorch/functions.py')==h['hytorch/functions.py']
assert sha(ROOT/'hytorch/backends/cuda/runtime.py')==h['hytorch/backends/cuda/runtime.py']
assert sha(O/'lib_before.so')==h['hytorch/backends/cuda/build/libhytorch_cuda.so']
assert sha(O/'lib_after.so')==sha(ROOT/'hytorch/backends/cuda/build/libhytorch_cuda.so')
b=(R/'kernels_before.cu').read_text();a=(ROOT/'hytorch/backends/cuda/kernels.cu').read_text();start=a.index('    // Conservative shape-only heuristic;');end=a.index('    cudaStream_t s=(cudaStream_t)stream;',start)
assert (a[:start]+a[end:]).replace('else if (use_register)','else if (variant==2 && m>=64 && n>=64 && k>=64)')==b
subprocess.run(['git','diff','--check'],check=True)
for p in (O/'diagnostics').glob('*.py'):ast.parse(p.read_text())
for name in ['REPORT.md','GEMM_COMPARISON.csv','TRAINING_COMPARISON.csv','DISPATCH_DECISIONS.csv','CORRECTNESS.md','SOURCE_CHANGES.md','git_diff.patch','REPRODUCE.md']:assert (O/name).stat().st_size>100
for name,count in [('baseline_gpu_retry.log',56),('gpu_after_tests.log',102),('host_after.log',22),('dtype_host_after.log',9)]:
 s=(R/name).read_text();assert f'{count} passed' in s and 'skipped' not in s and 'failed' not in s
r=list(csv.DictReader((O/'GEMM_COMPARISON.csv').open()));d=list(csv.DictReader((O/'DISPATCH_DECISIONS.csv').open()));t=list(csv.DictReader((O/'TRAINING_COMPARISON.csv').open()))
assert len(r)==len(d)==376 and len(t)==14
assert all(x['old_new_bitwise']=='True' and x['shared_new_bitwise']=='True' for x in r)
assert all(x['weights_bitwise_old']=='True' and float(x['weight_max_abs_vs_old'])==0 for x in t)
assert len({x['final_hash'] for x in t})==1
assert sum(x['original_strict_gate']=='False' for x in r)==22
for x in json.loads((O/'SOURCE_FILES.json').read_text()):assert sha(O/x['path'])==x['sha256']
(R/'final_integrity.json').write_text(json.dumps(dict(production_only_requested_dispatch=True,fp32_production_unchanged=True,runtime_abi_source_unchanged=True,baseline_library_preserved=True,kernel_math_unchanged=True,baseline_gpu_passed=56,new_gpu_passed=102,host_passed=31,strict_gate_failure_rows=22,new_gemm_numeric_regressions=0,training_weights_bitwise=True,regression_shapes_retained=[[64,8128,65],[449,897,71]],no_commit_push_pr=True),indent=2))
external=[]
for p in [O/'lib_before.so',O/'lib_after.so',ROOT/'runs/a100_20261106/libhytorch_cuda_baseline.so',ROOT/'hytorch/backends/cuda/build/libhytorch_cuda.so',*sorted((ROOT/'runs/a100_20261106/gemm_shape_dispatch_traces').glob('*'))]:
 if p.is_file():external.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),included=False))
(O/'EXTERNAL_ARTIFACTS.json').write_text(json.dumps(external,indent=2))
allowed={'.md','.csv','.json','.jsonl','.txt','.log','.patch','.py','.cu','.h','.dot'}
files=[p for p in sorted(O.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.suffix in allowed and p.name!='PACKAGE_FILES.json']
(O/'PACKAGE_FILES.json').write_text(json.dumps([dict(path=str(p.relative_to(O)),bytes=p.stat().st_size,sha256=sha(p)) for p in files],indent=2));files.append(O/'PACKAGE_FILES.json')
out=ROOT/'runs/a100_20261106/gemm_shape_dispatch.tar.gz'
with tarfile.open(out,'w:gz') as t:
 for p in files:t.add(p,arcname=str(Path(O.name)/p.relative_to(O)),recursive=False)
with tarfile.open(out,'r:gz') as t:
 assert len(t.getmembers())==len(files)
 for m in t.getmembers():
  assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
  assert hashlib.sha256(t.extractfile(m).read()).hexdigest()==sha(O/Path(m.name).relative_to(O.name))
print(json.dumps(dict(path=str(out),bytes=out.stat().st_size,sha256=sha(out),files=len(files)),indent=2))
