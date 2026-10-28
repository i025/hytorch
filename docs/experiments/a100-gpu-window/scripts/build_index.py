"""Index local and published artifacts by path/size, without hashing files."""
import csv
import json
from pathlib import Path
root=Path(__file__).resolve().parents[4]
report=root/'docs/experiments/a100-gpu-window'
docs=[root/'docs/experiments'/name for name in ('a100-stage2a-gemm','a100-stage2b-ops','a100-gpu-window')]
patterns=['gpu_window_*','gemm_stage2a_01','gemm_stage2a_pilot_02','gemm_stage2a_pilot_03','ops_stage2b_01','torch_gemm_01','gemm_diagnosis_01','comparison_cpu_*','synthetic_*','window_*_monitor_01','fixed_*_monitor_01','nsys_*','ncu_window_*','ncu_gemm_*','gpu_memcheck_window_01','pretrained_*_01','dtype_*','validation_dtype_fix_01','host_after_dtype_fix_01*','remaining_stages_01.jsonl','tiny_*fixed_report_01*','donkey_*fixed_report_01*']
files=set()
for pattern in patterns:
 for p in (root/'runs').glob(pattern):
  files.update(q for q in p.rglob('*') if q.is_file()) if p.is_dir() else files.add(p)
raw=[dict(path=str(p),bytes=p.stat().st_size,server_only=p.suffix in {'.npz','.nsys-rep','.sqlite','.ncu-rep'}) for p in sorted(files)]
(report/'raw-server-files.json').write_text(json.dumps(raw,indent=2))
with (report/'files.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['repository_path','server_path','bytes','github_url'],lineterminator='\n');w.writeheader()
 for base in docs:
  for p in sorted(base.rglob('*')):
   if not p.is_file() or '__pycache__' in str(p) or p.name=='files.csv':continue
   rel=p.relative_to(root)
   w.writerow(dict(repository_path=str(rel),server_path=str(p),bytes=p.stat().st_size,github_url='https://github.com/i025/hytorch/blob/experiment/a100-gpu-complete/'+str(rel)))
print(dict(published_index='files.csv',raw_files=len(raw),raw_bytes=sum(r['bytes'] for r in raw)))
