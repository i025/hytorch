from pathlib import Path
import hashlib,json,tarfile,subprocess
O=Path('runs/a100_20261106/dweight_im2col_reuse').resolve()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
subprocess.run(['git','diff','--check'],check=True)
required=['REPORT.md','SOURCE_CHANGES.md','DWEIGHT_BENCHMARK.csv','TRAINING_COMPARISON.csv','MEMORY_COMPARISON.csv','PROFILE_BEFORE_AFTER.csv','CORRECTNESS.md','git_diff.patch','REPRODUCE.md']
assert all((O/f).is_file() for f in required)
files=[p for p in sorted(O.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.so','.pyc'] and p.name!='PACKAGE_FILES.json']
manifest=[dict(path=str(p.relative_to(O)),bytes=p.stat().st_size,sha256=sha(p)) for p in files];(O/'PACKAGE_FILES.json').write_text(json.dumps(manifest,indent=2));files.append(O/'PACKAGE_FILES.json')
out=O.with_suffix('.tar.gz');assert not out.exists()
with tarfile.open(out,'w:gz') as t:
 for p in files:t.add(p,arcname='dweight_im2col_reuse/'+str(p.relative_to(O)),recursive=False)
with tarfile.open(out) as t:
 for r in manifest:assert hashlib.sha256(t.extractfile('dweight_im2col_reuse/'+r['path']).read()).hexdigest()==r['sha256']
print(json.dumps(dict(path=str(out),bytes=out.stat().st_size,sha256=sha(out),files=len(files)),indent=2))
