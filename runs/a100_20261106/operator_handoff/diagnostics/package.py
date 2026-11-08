"""CPU-only integrity check and shareable source/evidence package."""
from pathlib import Path
import hashlib,json,tarfile,subprocess,datetime,csv,ast
ROOT=Path.cwd();OUT=ROOT/'runs/a100_20261106/operator_handoff';RAW=OUT/'raw'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((OUT/'source_manifest.json').read_text())
checks=[dict(path=r['path'],unchanged=sha(ROOT/r['path'])==r['sha256']) for r in manifest]
status=subprocess.check_output(['git','status','--short'])
diff=subprocess.check_output(['git','diff','edb6fbe3bfaa9eed461addd29e6943b22e28216e'])
lib=ROOT/'hytorch/backends/cuda/build/libhytorch_cuda.so';previous=ROOT/'runs/a100_20261106/libhytorch_cuda_optimized.so'
integrity=dict(project_source_checks=checks,all_project_sources_unchanged=all(r['unchanged'] for r in checks),git_status_unchanged=status==(RAW/'git_status.txt').read_bytes(),tracked_patch_unchanged=diff==(RAW/'working_tree.patch').read_bytes(),cuda_library_matches_preexisting_optimized_copy=sha(lib)==sha(previous),cuda_library_sha256=sha(lib),no_commit_push_pr=True)
assert all(integrity[k] for k in ['all_project_sources_unchanged','git_status_unchanged','tracked_patch_unchanged','cuda_library_matches_preexisting_optimized_copy'])
(RAW/'final_integrity.json').write_text(json.dumps(integrity,indent=2))
artifacts=[]
for p in sorted((ROOT/'runs/a100_20261106/operator_handoff_traces').glob('*')):
 if p.is_file():artifacts.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),included_in_archive=False,kind='Nsight original report or SQLite; exported CSV retained in package'))
for p in [lib,previous,ROOT/'runs/a100_20261106/libhytorch_cuda_baseline.so']:
 artifacts.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),included_in_archive=False,kind='compiled library'))
(OUT/'EXTERNAL_ARTIFACTS.json').write_text(json.dumps(artifacts,indent=2))
required=['REPORT.md','OPERATOR_SOURCE.md','WORKLOAD.csv','PROFILE.csv','OPTIMIZATION_CANDIDATES.md','REPRODUCE.md']
for name in required:assert (OUT/name).stat().st_size>100
for p in (OUT/'diagnostics').glob('*.py'):ast.parse(p.read_text())
for name in ['WORKLOAD.csv','PROFILE.csv']:
 with (OUT/name).open() as f:r=list(csv.DictReader(f));assert r
 print(name,len(r),'rows')
# Validate arithmetic and provenance of generated profile summaries.
s=json.loads((RAW/'profile_summary.json').read_text())
for name,trace in s.items():
 assert trace['missing_correlations']==0
 assert abs(sum(x['total_us'] for x in trace['by_category'].values())-trace['kernel_total_us'])<1e-5
 assert abs(sum(x['total_us'] for x in trace['by_phase'].values())-trace['kernel_total_us'])<1e-5
assert s['native_plain']['kernel_count']==s['native_trace']['kernel_count']==1610
assert all(json.loads((RAW/'workload/summary.json').read_text())['checks'].values())
allowed={'.md','.csv','.json','.jsonl','.txt','.log','.patch','.py','.cu','.c','.h'}
files=[p for p in sorted(OUT.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.suffix in allowed and p.name!='PACKAGE_FILES.json']
(OUT/'PACKAGE_FILES.json').write_text(json.dumps([dict(path=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=sha(p)) for p in files],indent=2))
files.append(OUT/'PACKAGE_FILES.json')
archive=ROOT/'runs/a100_20261106/hytorch_operator_handoff.tar.gz'
with tarfile.open(archive,'w:gz') as t:
 for p in files:t.add(p,arcname=str(Path('operator_handoff')/p.relative_to(OUT)),recursive=False)
with tarfile.open(archive,'r:gz') as t:
 members=t.getmembers();assert len(members)==len(files)
 for m in members:
  p=Path(m.name)
  assert not p.is_absolute() and '..' not in p.parts
  assert p.suffix in allowed
  content=t.extractfile(m).read();assert hashlib.sha256(content).hexdigest()==sha(OUT/p.relative_to('operator_handoff'))
print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),files=len(files),source_files=len(json.loads((OUT/'SOURCE_FILES.json').read_text())),production_unchanged=True),indent=2))
