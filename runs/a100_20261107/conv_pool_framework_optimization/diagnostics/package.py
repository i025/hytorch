from pathlib import Path
import hashlib,json,csv,collections,tarfile
O=Path(__file__).resolve().parents[1];R=O.parents[2]
def read(n):return list(csv.DictReader((O/n).open()))
required=['REPORT.md','CALL_PATH.md','CPU_OVERHEAD.csv','CONV_BACKWARD_COMPARISON.csv','POOLING_COMPARISON.csv','TRAINING_ABLATION.csv','PROFILE_BEFORE_AFTER.csv','MEMORY_COMPARISON.csv','CORRECTNESS.md','SOURCE_CHANGES.md','REPRODUCE.md','git_diff.patch']
assert all((O/n).is_file() and (O/n).stat().st_size for n in required)
rows=read('TRAINING_ABLATION.csv');assert len(rows)==36
assert collections.Counter(r['mode'] for r in rows)==dict(B0=12,O1=12,torch=12)
orders=[]
for i in range(12):orders.append(tuple(r['mode'] for r in sorted([r for r in rows if int(r['repeat'])==i],key=lambda x:int(x['order']))))
assert set(collections.Counter(orders).values())=={2}
local=read('LAYER_COMPARISON.csv');assert all(int(r['n_unprofiled'])==18 and int(r['n_cupti'])==15 for r in local)
for name in ['TRAINING_ERRORS.csv','SUPPLEMENT_ERRORS.csv','raw/LAYER_ERRORS_balanced18.csv','raw/LAYER_ERRORS_actual18.csv']:
 es=read(name);assert all(int(r['failed'])==0 for r in es)
 assert all(r['bitwise_equal']=='True' for r in es if r.get('mode',r.get('implementation'))=='O1')
assert '319 passed, 2 skipped' in (O/'raw/tests_after.log').read_text()
assert '291 passed, 2 skipped' in (O/'raw/tests_on.log').read_text()
assert 'ERROR SUMMARY: 0 errors' in (O/'raw/memcheck.log').read_text()
assert '0 hazards displayed (0 errors, 0 warnings)' in (O/'raw/racecheck.log').read_text()
assert len(list((O/'figures').glob('*.png')))==5 and len(list((O/'figures').glob('*.svg')))==5
integrity=json.loads((O/'SOURCE_INTEGRITY.json').read_text());assert [r['path'] for r in integrity if not r['unchanged']]==['hytorch/core.py']
for r in integrity:assert hashlib.sha256((R/r['path']).read_bytes()).hexdigest()==r['final_sha256']
assert all(json.loads(line)['returncode']==0 for line in (O/'raw/commands.jsonl').read_text().splitlines())
validation=dict(required_files=required,training_rows=len(rows),balanced_orders=dict((str(k),v) for k,v in collections.Counter(orders).items()),layer_summary_rows=len(local),unprofiled_per_case=18,cupti_per_case=15,all_numerical_checks_pass=True,O1_bitwise_equal=True,production_change=['hytorch/core.py'],plots=10,patch_apply_verified=json.loads((O/'raw/PATCH_VERIFICATION.json').read_text()))
(O/'raw/DELIVERY_VALIDATION.json').write_text(json.dumps(validation,indent=2))
files=[p for p in sorted(O.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='MANIFEST_SHA256.json' and p.suffix!='.pyc']
manifest=[dict(path=str(p.relative_to(O)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]
(O/'MANIFEST_SHA256.json').write_text(json.dumps(manifest,indent=2));files.append(O/'MANIFEST_SHA256.json')
archive=O.with_suffix('.tar.gz')
if archive.exists():raise RuntimeError('Refuse overwriting an existing archive')
with tarfile.open(archive,'w:gz',compresslevel=6) as tf:
 for p in files:tf.add(p,arcname=str(Path(O.name)/p.relative_to(O)),recursive=False)
with tarfile.open(archive,'r:gz') as tf:
 for r in manifest:
  data=tf.extractfile(O.name+'/'+r['path']).read();assert hashlib.sha256(data).hexdigest()==r['sha256']
sha=hashlib.sha256(archive.read_bytes()).hexdigest();archive.with_suffix('.gz.sha256').write_text(sha+'  '+archive.name+'\n')
print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,files=len(files),sha256=sha),indent=2))
