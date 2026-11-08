from pathlib import Path
import hashlib,json,tarfile,subprocess,tempfile,shutil,csv
R=Path.cwd();O=R/'runs/a100_20261106/col2im_binding_optimization'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
required=['REPORT.md','CPU_OVERHEAD.csv','COL2IM_COMPARISON.csv','TRAINING_COMPARISON.csv','PROFILE_BEFORE_AFTER.csv','CORRECTNESS.md','SOURCE_CHANGES.md','git_diff.patch','REPRODUCE.md']
assert all((O/f).is_file() for f in required)
assert len(list(csv.DictReader((O/'COL2IM_COMPARISON.csv').open())))==108
assert len(list(csv.DictReader((O/'TRAINING_COMPARISON.csv').open())))==30
with tempfile.TemporaryDirectory(prefix='col2im-patch-check-') as d:
 for f in ['hytorch/backends/cuda/runtime.py','hytorch/backends/cuda/kernels.cu']:
  dst=Path(d)/f;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(O/'baseline'/f,dst)
 subprocess.run(['git','apply','--check',str(O/'git_diff.patch')],cwd=d,check=True)
 subprocess.run(['git','apply',str(O/'git_diff.patch')],cwd=d,check=True)
 for f in ['hytorch/backends/cuda/runtime.py','hytorch/backends/cuda/kernels.cu','tests/test_col2im_binding.py']:assert sha(Path(d)/f)==sha(R/f)
subprocess.run(['git','diff','--check'],check=True)
files=[p for p in sorted(O.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.so','.pyc','.nsys-rep','.sqlite'] and p.name!='PACKAGE_FILES.json']
manifest=[dict(path=str(p.relative_to(O)),bytes=p.stat().st_size,sha256=sha(p)) for p in files]
(O/'PACKAGE_FILES.json').write_text(json.dumps(manifest,indent=2));files.append(O/'PACKAGE_FILES.json')
target=O.with_suffix('.tar.gz');assert not target.exists(),'Never overwrite delivered archive'
with tarfile.open(target,'w:gz') as tar:
 for p in files:tar.add(p,arcname='col2im_binding_optimization/'+str(p.relative_to(O)),recursive=False)
with tarfile.open(target) as tar:
 for r in manifest:
  f=tar.extractfile('col2im_binding_optimization/'+r['path']);assert hashlib.sha256(f.read()).hexdigest()==r['sha256']
print(json.dumps(dict(path=str(target),bytes=target.stat().st_size,sha256=sha(target),files=len(files),patch_check='applies to baseline and exactly reproduces tested sources'),indent=2))
