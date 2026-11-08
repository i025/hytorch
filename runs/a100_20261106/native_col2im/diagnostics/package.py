from pathlib import Path
import json,csv,hashlib,tarfile,subprocess,ast
ROOT=Path.cwd();O=ROOT/'runs/a100_20261106/native_col2im';R=O/'raw';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert (ROOT/'hytorch/backends/cuda/kernels.cu').read_bytes().startswith((O/'baseline/hytorch/backends/cuda/kernels.cu').read_bytes())
assert sha(ROOT/'hytorch/functions.py')==json.loads((R/'before_hashes.json').read_text())['hytorch/functions.py']
assert sha(O/'lib_baseline.so')==sha(ROOT/'runs/a100_20261106/gemm_shape_dispatch/lib_after.so')
assert sha(O/'lib_native_col2im.so')==sha(ROOT/'hytorch/backends/cuda/build/libhytorch_cuda.so')
subprocess.run(['git','diff','--check'],check=True)
for p in (O/'diagnostics').glob('*.py'):ast.parse(p.read_text())
for name in ['REPORT.md','SOURCE_CHANGES.md','COL2IM_BENCHMARK.csv','TRAINING_COMPARISON.csv','PROFILE_BEFORE_AFTER.csv','CORRECTNESS.md','git_diff.patch','REPRODUCE.md']:assert (O/name).stat().st_size>100
for name,count in [('gpu_tests.log',200),('host.log',23),('dtype_host.log',9)]:
 s=(R/name).read_text();assert f'{count} passed' in s and 'skipped' not in s and 'failed' not in s
assert 'ERROR SUMMARY: 0 errors' in (R/'memcheck.log').read_text()
assert '0 hazards displayed (0 errors, 0 warnings)' in (R/'racecheck.log').read_text()
b=list(csv.DictReader((O/'COL2IM_BENCHMARK.csv').open()));t=list(csv.DictReader((O/'TRAINING_COMPARISON.csv').open()))
assert len(b)==72 and len(t)==14
assert all(r['bitwise_equal']=='True' and float(r['max_abs_error'])==0 for r in b)
assert all(r['gradients_bitwise_cupy']=='True' and r['weights_bitwise_old']=='True' and float(r['grad_max_abs_vs_cupy'])==0 for r in t)
assert len({r['final_hash'] for r in t})==1
assert all(float(r['paired_cupy_minus_native_s'])<0 for r in t)
for x in json.loads((O/'SOURCE_FILES.json').read_text()):assert sha(O/x['path'])==x['sha256']
s=json.loads((R/'profile_summary.json').read_text());assert s['cupy']['kernel_count']==s['native']['kernel_count']==1470
assert all(v['memcpy_count']==20 and v['memcpy_bytes']==6336160 and v['fp64_cast_count']==0 for v in s.values())
external=[]
for p in [O/'lib_baseline.so',O/'lib_native_col2im.so',ROOT/'hytorch/backends/cuda/build/libhytorch_cuda.so',*sorted((ROOT/'runs/a100_20261106/native_col2im_traces').glob('*'))]:
 if p.is_file():external.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),included=False))
(O/'EXTERNAL_ARTIFACTS.json').write_text(json.dumps(external,indent=2))
(R/'final_integrity.json').write_text(json.dumps(dict(scope_checks=json.loads((R/'scope_checks.json').read_text()),gpu_tests_passed=200,host_tests_passed=32,memcheck_errors=0,racecheck_hazards=0,benchmark_cases=36,native_regression_cases=5,training_pairs=7,training_all_pairs_slower=True,loss_gradients_weights_bitwise_equal=True,default_backend='cupy',no_commit_push_pr=True),indent=2))
allowed={'.md','.csv','.json','.jsonl','.txt','.log','.patch','.py','.cu','.h','.dot'}
files=[p for p in sorted(O.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.suffix in allowed and p.name!='PACKAGE_FILES.json']
(O/'PACKAGE_FILES.json').write_text(json.dumps([dict(path=str(p.relative_to(O)),bytes=p.stat().st_size,sha256=sha(p)) for p in files],indent=2));files.append(O/'PACKAGE_FILES.json')
archive=ROOT/'runs/a100_20261106/native_col2im.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
 for p in files:tar.add(p,arcname=str(Path(O.name)/p.relative_to(O)),recursive=False)
with tarfile.open(archive,'r:gz') as tar:
 assert len(tar.getmembers())==len(files)
 for m in tar.getmembers():
  assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
  assert hashlib.sha256(tar.extractfile(m).read()).hexdigest()==sha(O/Path(m.name).relative_to(O.name))
print(json.dumps(dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),files=len(files)),indent=2))
