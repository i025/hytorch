"""Publish logs and resource evidence; retain unredacted originals under runs/."""
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[4]
DEST=ROOT/'docs/experiments/gemm-block32-tree'
for phase in ('correctness','accuracy','benchmark','ncu','nsys'):
    versions=(1,2) if phase=='nsys' else (1,)
    for version in versions:
        source=ROOT/f'runs/gemm_block32_tree_{phase}_monitor_{version:02d}'
        target=DEST/f'logs/{phase}-monitor-{version:02d}'
        target.mkdir(exist_ok=False)
        for path in source.iterdir():
            out=target/path.name
            if path.name in ('before.json','gpu-load.jsonl'):
                rows=json.loads(path.read_text()) if path.name=='before.json' else [json.loads(line) for line in path.read_text().splitlines()]
                for row in rows:
                    for key in ('process_ids','external_process_ids','owned_process_ids','owned_outside_group_ids'):
                        if key in row:
                            row[key.replace('_ids','_count')]=len(row.pop(key))
                out.write_text(json.dumps(rows,indent=2) if path.name=='before.json' else ''.join(json.dumps(row)+'\n' for row in rows))
            else:
                shutil.copyfile(path,out)
for name in ('build.json','build.log'):
    shutil.copyfile(ROOT/'runs/gemm_block32_tree_build_01'/name,DEST/'environment'/name)
shutil.copyfile(ROOT/'hytorch/backends/cuda/build/build.json',DEST/'environment/original-build.json')
print('Archived monitor logs with process IDs removed; raw logs retained under runs/')
