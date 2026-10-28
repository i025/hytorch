"""Copy numerical artifacts; redact process IDs from published GPU monitoring."""
import argparse
import json
from pathlib import Path
import shutil
p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('destination',type=Path);a=p.parse_args()
a.destination.mkdir(parents=True,exist_ok=False)
for source in sorted(a.source.rglob('*')):
    if not source.is_file() or source.suffix not in {'.json','.jsonl','.csv','.xml','.log','.txt','.png'}: continue
    dest=a.destination/source.relative_to(a.source);dest.parent.mkdir(parents=True,exist_ok=True)
    if source.name=='before.json':
        value=json.loads(source.read_text());value['compute_process_count']=len(value.pop('processes','').splitlines());dest.write_text(json.dumps(value,indent=2))
    elif source.name=='gpu-load.jsonl':
        with dest.open('x') as f:
            for line in source.read_text().splitlines():
                row=json.loads(line);processes=row.pop('processes','').splitlines();pid=row.pop('benchmark_pid')
                row['compute_process_count']=len(processes)
                row['root_benchmark_process_observed']=any(v.split(',')[0].strip()==str(pid) for v in processes)
                f.write(json.dumps(row)+'\n')
    else: shutil.copyfile(source,dest)
print(a.destination)
