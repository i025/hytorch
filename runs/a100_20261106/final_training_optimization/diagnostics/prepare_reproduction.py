from pathlib import Path
import sys,shutil
source=Path(__file__).resolve().parents[1];target=Path(sys.argv[1]).resolve();target.mkdir(parents=True,exist_ok=False)
shutil.copytree(source/'source',target,dirs_exist_ok=True)
o=target/'runs/a100_20261106/final_training_optimization';o.mkdir(parents=True);(o/'raw').mkdir();(o/'figures').mkdir()
shutil.copytree(source/'diagnostics',o/'diagnostics');shutil.copytree(source/'baseline',o/'baseline')
for name in ['order.json','prototype_tests.py']:shutil.copyfile(source/'raw'/name,o/'raw'/name)
shutil.copyfile(source/'ACCEPTANCE.md',o/'ACCEPTANCE.md')
print(target)
