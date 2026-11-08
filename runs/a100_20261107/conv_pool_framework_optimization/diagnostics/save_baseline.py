from pathlib import Path
import hashlib,json,subprocess,shutil,sys,platform,importlib.metadata
R=Path(__file__).resolve().parents[4]
O=R/'runs/a100_20261107/conv_pool_framework_optimization'
B=O/'baseline'
B.mkdir(exist_ok=False)
for name,args in [('git_status.txt',['status','--short']),('HEAD.txt',['rev-parse','HEAD']),('working_tree.patch',['diff','--binary','HEAD']),('untracked.txt',['ls-files','--others','--exclude-standard'])]:
 (B/name).write_bytes(subprocess.check_output(['git',*args],cwd=R))
files=[]
for folder in ['hytorch','tests','benchmarks','donkeycar_hy','scripts']:
 for p in sorted((R/folder).rglob('*')):
  if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.cu','.cpp','.h','.so','.json','.log'):
   q=B/'source'/p.relative_to(R);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
   files.append({'path':str(p.relative_to(R)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size})
(B/'SHA.json').write_text(json.dumps(files,indent=2))
env={'python':sys.version,'platform':platform.platform(),'packages':{}}
for name in ['torch','cupy-cuda12x','numpy','pytest']:
 try:env['packages'][name]=importlib.metadata.version(name)
 except importlib.metadata.PackageNotFoundError:env['packages'][name]=None
(B/'ENVIRONMENT.json').write_text(json.dumps(env,indent=2))
print(f'Saved {len(files)} source/build files to {B}')
