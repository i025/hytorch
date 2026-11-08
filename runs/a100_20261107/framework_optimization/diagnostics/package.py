from pathlib import Path
import json,hashlib,tarfile,csv
O=Path(__file__).resolve().parents[1];target=O.parent/'framework_optimization.tar.gz'
required=['OPERATOR_COMPARISON.csv','CONV_FORWARD.csv','CONV_BACKWARD.csv','GEMM_COMPARISON.csv','OTHER_OPERATORS.csv','TRAINING_HOTSPOTS.csv','OPERATOR_REPORT.md','FRAMEWORK_PROFILE.md','OPTIMIZATION_DESIGN.md','FRAMEWORK_COMPARISON.csv','TRAINING_ABLATION.csv','MEMORY_COMPARISON.csv','CORRECTNESS.md','SOURCE_CHANGES.md','git_diff.patch','REPRODUCE.md','PPT_SUMMARY.md','REPORT.md']
assert all((O/n).is_file() and (O/n).stat().st_size for n in required)
assert len(list((O/'figures').glob('*.png')))==6 and len(list((O/'figures').glob('*.svg')))==6
r=list(csv.DictReader((O/'TRAINING_ABLATION.csv').open()));assert len(r)==36
assert all(sum(x['mode']==m for x in r)==12 for m in ['B0','F1','torch'])
assert all(int(x['kernel_repeats'])==15 for x in csv.DictReader((O/'OPERATOR_COMPARISON.csv').open()))
for n in ['TRAINING_ERRORS.csv','SUPPLEMENT_ERRORS.csv','stage_a/OPERATOR_ERRORS_a.csv']:assert all(int(x['failed'])==0 for x in csv.DictReader((O/n).open()))
manifest={}
for p in sorted(O.rglob('*')):
 if p.is_file() and '__pycache__' not in p.parts and p.name!='PACKAGE_FILES.json':manifest[str(p.relative_to(O))]=dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
(O/'PACKAGE_FILES.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
if target.exists():raise SystemExit('Refuse to overwrite archive')
with tarfile.open(target,'w:gz') as t:
 for name in [*manifest,'PACKAGE_FILES.json']:t.add(O/name,arcname=O.name+'/'+name,recursive=False)
with tarfile.open(target,'r:gz') as t:
 for name,meta in manifest.items():
  data=t.extractfile(O.name+'/'+name).read();assert len(data)==meta['bytes'] and hashlib.sha256(data).hexdigest()==meta['sha256']
print(json.dumps(dict(path=str(target),files=len(manifest)+1,bytes=target.stat().st_size,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),readback_hashes_verified=True),indent=2))
