from pathlib import Path
import json,hashlib,tarfile,subprocess
O=Path('runs/a100_20261106/final_training_optimization').resolve()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
subprocess.run(['git','diff','--check'],check=True)
for name,h in json.loads((O/'SHA.json').read_text()).items():assert sha(O/name)==h,name
for name in ['REPORT.md','ACCEPTANCE.md','CLAIMS.md','PPT_SUMMARY.md','REPRODUCE.md','SOURCE_CHANGES.md','CORRECTNESS.md','TRAINING.csv','NUMERICS.csv','MEMORY.csv','PROFILE.csv','FULL_WRAPPERS.csv']:assert (O/name).exists()
files=[p for p in sorted(O.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc' and p.name!='PACKAGE_FILES.json']
manifest=[dict(path=str(p.relative_to(O)),bytes=p.stat().st_size,sha256=sha(p)) for p in files];(O/'PACKAGE_FILES.json').write_text(json.dumps(manifest,indent=2));files.append(O/'PACKAGE_FILES.json')
target=O.with_suffix('.tar.gz');assert not target.exists(),'Do not overwrite an existing delivery'
with tarfile.open(target,'w:gz') as t:
 for p in files:t.add(p,arcname='final_training_optimization/'+str(p.relative_to(O)),recursive=False)
with tarfile.open(target) as t:
 for r in manifest:assert hashlib.sha256(t.extractfile('final_training_optimization/'+r['path']).read()).hexdigest()==r['sha256'],r['path']
print(json.dumps(dict(path=str(target),bytes=target.stat().st_size,sha256=sha(target),files=len(files)),indent=2))
