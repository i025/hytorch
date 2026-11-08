from pathlib import Path
import shutil,sys
O=Path(__file__).resolve().parents[1];dest=Path(sys.argv[1]).resolve()
if dest.exists():raise SystemExit('Destination must not exist; refusing to overwrite evidence')
shutil.copytree(O/'baseline/source',dest)
n=dest/'runs/a100_20261107/framework_optimization';n.mkdir(parents=True)
for folder in ['baseline','diagnostics','source']:shutil.copytree(O/folder,n/folder,ignore=shutil.ignore_patterns('__pycache__'))
for folder in ['raw','stage_a','traces','figures']:(n/folder).mkdir()
for file in ['git_diff.patch','EXPERIMENT_PROTOCOL.md']:shutil.copy2(O/file,n/file)
print('Baseline workspace prepared:',dest)
print('Run stage A, freeze, then git apply the supplied incremental patch before stage B.')
