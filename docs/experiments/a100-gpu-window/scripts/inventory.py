"""Read-only inventory of tools and dataset candidates within the user's workspace."""
import json
from pathlib import Path
import re
import shutil
import subprocess

home=Path.home()
roots=[home/'projects',home/'data',home/'datasets',Path('/data'),Path('/datasets')]
report=dict(profiling_tools={name:shutil.which(name) for name in ('nsys','ncu','compute-sanitizer')},
            dataset_search_roots=[str(p) for p in roots if p.exists()],donkey_filename_candidates={})
for root in [p for p in roots if p.exists() and p.is_dir()]:
    result=subprocess.run(['rg','--files','--hidden','--no-ignore',
        '-g','!**/.git/**','-g','!**/.venv/**','-g','!**/node_modules/**',
        '-g','!**/.cache/**','-g','*.jpg',str(root)],text=True,capture_output=True)
    if result.returncode not in (0,1):
        report.setdefault('search_errors',[]).append(dict(root=str(root),error=result.stderr))
    for name in result.stdout.splitlines():
        path=Path(name)
        if re.fullmatch(r'\d+_[-+]?\d+(?:\.\d+)?\.jpg',path.name):
            parent=str(path.parent)
            report['donkey_filename_candidates'][parent]=report['donkey_filename_candidates'].get(parent,0)+1
report['classification_dataset_status']='NOT_CONFIRMED; no user-specified labelled dataset path'
print(json.dumps(report,indent=2))
