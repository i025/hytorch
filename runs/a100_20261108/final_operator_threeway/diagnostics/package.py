from pathlib import Path
import hashlib,json,tarfile,csv,collections,math
O=Path(__file__).resolve().parents[1];R=O.parents[2]
required=['REPORT.md','WORKLOAD.csv','GEMM_RESULTS.csv','IM2COL_RESULTS.csv','COL2IM_RESULTS.csv','CONV_FORWARD_RESULTS.csv','DWEIGHT_RESULTS.csv','POOL_RESULTS.csv','ACCURACY.csv','SPEEDUP_SUMMARY.csv','RAW_TIMINGS.csv','CORRECTNESS.md','REPRODUCE.md','SOURCE_STATUS.md','PPT_SUMMARY.md']
assert all((O/f).is_file() for f in required)
def read(n):return list(csv.DictReader((O/n).open()))
r=read('RAW_TIMINGS.csv');a=read('ACCURACY.csv');summ=read('RESULTS_ALL.csv');profiles=read('PROFILE_SUMMARY.csv')
assert len(a)==804 and all(int(v['mixed_fail_count'])==0 for v in a)
assert all(int(v['repeats'])>=15 for v in summ)
assert len(read('WORKLOAD.csv'))==122
keys=lambda v:(v['op'],v['case'],v['scope'],v['mode'])
seen=collections.Counter(keys(v) for v in a);assert all(v==1 for v in seen.values())
rawgroups=collections.defaultdict(list)
for v in r:rawgroups[keys(v)].append(v)
assert set(rawgroups)==set(seen)
for v in summ:
 rr=rawgroups[keys(v)];vals=[float(x[v['metric']]) for x in rr];assert len(vals)==int(v['repeats']);assert math.isclose(sum(vals)/len(vals),float(v['mean']),rel_tol=1e-12)
assert all(int(v['repeats'])==15 for v in profiles)
assert len(profiles)==560
assert len(list((O/'figures').glob('*.png')))==7 and len(list((O/'figures').glob('*.svg')))==7
old=json.loads((O/'raw/source_hashes_before.json').read_text())
assert all(hashlib.sha256((R/v['path']).read_bytes()).hexdigest()==v['sha256'] for v in old)
external=[]
for p in sorted((O/'traces').glob('*')):
 if p.is_file():external.append(dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),included=False))
for v in old:
 if v['path'].endswith('.so'):external.append(dict(path=str(R/v['path']),bytes=v['bytes'],sha256=v['sha256'],included=False))
(O/'EXTERNAL_ARTIFACTS.json').write_text(json.dumps(external,indent=2))
files=[p for p in O.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.so','.sqlite','.nsys-rep') and p.name not in ('PACKAGE_FILES.json',)]
manifest=[dict(path=str(p.relative_to(O)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(files)]
(O/'PACKAGE_FILES.json').write_text(json.dumps(manifest,indent=2))
archive=O.with_suffix('.tar.gz')
if archive.exists():raise SystemExit('Archive exists: refusing to overwrite')
with tarfile.open(archive,'w:gz') as t:
 for p in sorted(files+[O/'PACKAGE_FILES.json']):t.add(p,arcname=str(Path(O.name)/p.relative_to(O)),recursive=False)
with tarfile.open(archive) as t:
 assert len(t.getmembers())==len(files)+1
 for p in t.getnames():assert not p.endswith(('.so','.sqlite','.nsys-rep','.pyc'))
sha=hashlib.sha256(archive.read_bytes()).hexdigest();archive.with_suffix(archive.suffix+'.sha256').write_text(f'{sha}  {archive.name}\n')
print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,sha256=sha,files=len(files)+1,timing_rows=len(r),accuracy_comparisons=len(a),profile_groups=len(profiles),source_hashes_verified=len(old)),indent=2))
