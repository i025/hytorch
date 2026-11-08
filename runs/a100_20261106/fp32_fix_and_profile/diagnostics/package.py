"""CPU-only integrity checks and upload artifact. No GPU operations."""
from pathlib import Path
import json,hashlib,csv,tarfile,subprocess,ast
ROOT=Path.cwd();O=ROOT/'runs/a100_20261106/fp32_fix_and_profile';R=O/'raw';OLD=ROOT/'runs/a100_20261106/operator_handoff'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert all(sha(ROOT/p)==v for p,v in json.loads((R/'kernel_sha_before.json').read_text()).items())
# Prior handoff manifest verifies untouched project files including earlier edits.
changes=[]
for r in json.loads((OLD/'source_manifest.json').read_text()):
 p=ROOT/r['path']
 if sha(p)!=r['sha256']:changes.append(r['path'])
assert changes==['hytorch/functions.py'] or set(changes)=={'hytorch/functions.py'},changes
subprocess.run(['git','diff','--check'],check=True)
for p in (O/'diagnostics').glob('*.py'):ast.parse(p.read_text())
for name in ['REPORT.md','DTYPE_AUDIT.csv','TRAINING_COMPARISON.csv','GEMM_REAL_SHAPES.csv','PROFILE.csv','SOURCE_CHANGES.md','NEXT_OPTIMIZATION.md','REPRODUCE.md']:assert (O/name).stat().st_size>100
for file in ['dtype_gpu_tests.log','original_gpu_tests.log']:
 s=(R/file).read_text();assert 'passed' in s and 'skipped' not in s and 'failed' not in s
for mode in ['workload','before_workload']:assert all(json.loads((R/mode/'summary.json').read_text())['checks'].values())
p=json.loads((R/'training_profile_summary.json').read_text())
assert all(v['unattributed_kernels']==0 for v in p.values())
for mode in ['after_native','after_cupy','torch']:
 assert p[mode]['float32_to_float64_kernel_count']==0 and p[mode]['fp64_named_kernel_count']==0
rows=list(csv.DictReader((O/'GEMM_REAL_SHAPES.csv').open()));assert len(rows)==72
assert all(r['current_bitwise_original']=='True' for r in rows)
assert sum(r['original_strict_gate']=='False' for r in rows)==4
training=list(csv.DictReader((O/'TRAINING_COMPARISON.csv').open()));assert len(training)==20
assert all(r['weights_allclose']=='True' for r in training)
for mode in ['before_native','after_native','after_cupy','torch']:
 assert len(set(r['final_hash'] for r in training if r['mode']==mode))==1
for x in json.loads((O/'SOURCE_FILES.json').read_text()):assert sha(O/x['path'])==x['sha256']
(R/'final_integrity.json').write_text(json.dumps(dict(changed_preexisting_project_files=changes,new_test='tests/test_backward_dtype.py',cuda_source_runtime_and_library_unchanged=True,prior_user_changes_preserved=True,strict_gemm_gate_failures_retained=4,gpu_tests_actually_executed=55,host_final_tests=31,final_dtype_metadata_guard_only_cpu_verified=True,no_commit_push_pr=True),indent=2))
external=[]
for q in sorted((ROOT/'runs/a100_20261106/fp32_fix_and_profile_traces').glob('*')):
 if q.is_file():external.append(dict(path=str(q),bytes=q.stat().st_size,sha256=sha(q),included=False))
for q in [ROOT/'hytorch/backends/cuda/build/libhytorch_cuda.so',ROOT/'runs/a100_20261106/libhytorch_cuda_baseline.so',ROOT/'runs/a100_20261106/libhytorch_cuda_optimized.so']:
 external.append(dict(path=str(q),bytes=q.stat().st_size,sha256=sha(q),included=False))
(O/'EXTERNAL_ARTIFACTS.json').write_text(json.dumps(external,indent=2))
allowed={'.md','.py','.cu','.h','.json','.jsonl','.csv','.txt','.log','.patch'}
files=[p for p in sorted(O.rglob('*')) if p.is_file() and p.suffix in allowed and '__pycache__' not in p.parts and p.name!='PACKAGE_FILES.json']
(O/'PACKAGE_FILES.json').write_text(json.dumps([dict(path=str(p.relative_to(O)),bytes=p.stat().st_size,sha256=sha(p)) for p in files],indent=2));files.append(O/'PACKAGE_FILES.json')
archive=ROOT/'runs/a100_20261106/fp32_fix_and_profile.tar.gz'
with tarfile.open(archive,'w:gz') as t:
 for p in files:t.add(p,arcname=str(Path(O.name)/p.relative_to(O)),recursive=False)
with tarfile.open(archive,'r:gz') as t:
 assert len(t.getmembers())==len(files)
 for m in t.getmembers():
  assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
  assert hashlib.sha256(t.extractfile(m).read()).hexdigest()==sha(O/Path(m.name).relative_to(O.name))
print(json.dumps(dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),files=len(files),source_files=len(json.loads((O/'SOURCE_FILES.json').read_text()))),indent=2))
