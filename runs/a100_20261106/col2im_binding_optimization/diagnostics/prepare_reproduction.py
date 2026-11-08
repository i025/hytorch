"""Create a fresh, minimal reproduction workspace; never overwrite the measured run."""
from pathlib import Path
import shutil,sys
source=Path(__file__).resolve().parents[1]
target=Path(sys.argv[1]).resolve();target.mkdir(parents=True,exist_ok=False)
shutil.copytree(source/'source',target,dirs_exist_ok=True)
out=target/'runs/a100_20261106/col2im_binding_optimization'
out.mkdir(parents=True);(out/'raw').mkdir()
shutil.copytree(source/'diagnostics',out/'diagnostics')
shutil.copytree(source/'baseline',out/'baseline',ignore=shutil.ignore_patterns('*.so','__pycache__'))
for f in ['cache_only_runtime.py','reference_kernel_order.csv']:shutil.copyfile(source/'raw'/f,out/'raw'/f)
print(target)
