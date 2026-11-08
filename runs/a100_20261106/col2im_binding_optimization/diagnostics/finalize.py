from pathlib import Path
import csv,json,hashlib,shutil,subprocess,difflib,pstats,statistics
R=Path.cwd();O=R/'runs/a100_20261106/col2im_binding_optimization';T=R/'runs/a100_20261106/col2im_binding_traces'
def write(path,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with path.open('w') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
# Baseline includes full Python/native sources and tests; prior changed files are preserved.
for pattern in ['hytorch/**/*.py','hytorch/**/*.cu','tests/*.py','scripts/build_cuda.py','benchmarks/bench_cuda_training.py','donkeycar_hy/*.py']:
 for p in R.glob(pattern):
  if 'build' in p.parts:continue
  rel=p.relative_to(R);dst=O/'source'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst)
  base=O/'baseline'/rel
  if not base.exists() and str(rel)!='tests/test_col2im_binding.py':base.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,base)
changed=['hytorch/backends/cuda/runtime.py','hytorch/backends/cuda/kernels.cu','tests/test_col2im_binding.py'];patch=''
for f in changed:
 old=O/'baseline'/f;new=R/f
 patch+=''.join(difflib.unified_diff(old.read_text().splitlines(True) if old.exists() else [],new.read_text().splitlines(True),fromfile='a/'+f if old.exists() else '/dev/null',tofile='b/'+f))
(O/'git_diff.patch').write_text(patch)
checks=dict(cuda_baseline_exact_prefix=(R/'hytorch/backends/cuda/kernels.cu').read_bytes().startswith((O/'baseline/hytorch/backends/cuda/kernels.cu').read_bytes()),functions_conv_unchanged=sha(R/'hytorch/functions_conv.py')==sha(O/'baseline/hytorch/functions_conv.py'),functions_dtype_fix_unchanged=sha(R/'hytorch/functions.py')==sha(O/'baseline/hytorch/functions.py'))
assert all(checks.values());(O/'raw/scope_checks.json').write_text(json.dumps(checks,indent=2))
rows=list(csv.DictReader((O/'CPU_OVERHEAD.csv').open()))
# Keep profile attribution separate from unprofiled timings, do not sum them.
if not any(r['scope']=='cProfile_instrumented_self_elapsed' for r in rows):
 for path in sorted((O/'raw').glob('*_old.pstats'))+sorted((O/'raw').glob('*_fast.pstats'))+sorted((O/'raw').glob('*_cupy.pstats')):
  stats=pstats.Stats(str(path))
  for (file,line,name),(cc,nc,tt,ct,callers) in sorted(stats.stats.items(),key=lambda kv:kv[1][2],reverse=True)[:12]:
   rows.append(dict(case=path.stem,component=f'{Path(file).name}:{line}:{name}',scope='cProfile_instrumented_self_elapsed',median_us='',profile_calls=nc,profile_total_self_us=tt*1e6,profile_self_us_per_call=tt*1e6/nc,profile_cumulative_us=ct*1e6))
write(O/'CPU_OVERHEAD.csv',rows)
external=[O/'baseline/libhytorch_cuda.so',O/'lib_native_fast.so']+sorted(T.glob('*'))
(O/'EXTERNAL_ARTIFACTS.json').write_text(json.dumps([dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in external],indent=2))
(O/'raw/git_status_final.txt').write_bytes(subprocess.check_output(['git','status','--short']))
subprocess.run(['git','diff','--check'],check=True)
print(checks)
