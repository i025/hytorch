"""List server raw artifacts and published evidence by path/size, without hashes."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DOC = ROOT/'docs/experiments/donkeynet-backward-profiling'
raw = set()
for pattern in ['nsys_dtype_fixed_01.*', 'stage4a_*', 'donkeynet_4a_*']:
    for path in (ROOT/'runs').glob(pattern):
        if path.is_file():
            raw.add(path)
        else:
            raw.update(p for p in path.rglob('*') if p.is_file())
for mode, repeats in [('throughput',3), ('phases',1)]:
    for method in ['cupy','combined','torch']:
        for repeat in range(1,repeats+1):
            folder = ROOT/'runs'/f'synthetic_donkey_{mode}_fixed_01'/f'{method}_r{repeat}'
            raw.update(p for p in folder.rglob('*') if p.is_file())
with (DOC/'raw-file-index.tsv').open('w', newline='') as stream:
    writer=csv.writer(stream, delimiter='\t', lineterminator='\n')
    writer.writerow(['server_path','bytes'])
    writer.writerows((str(p),p.stat().st_size) for p in sorted(raw))
files=[p for p in DOC.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='file-index.tsv']
files.append(ROOT/'benchmarks/bench_cuda_training.py')
with (DOC/'file-index.tsv').open('w', newline='') as stream:
    writer=csv.writer(stream, delimiter='\t', lineterminator='\n')
    writer.writerow(['repository_path','bytes'])
    writer.writerows((str(p.relative_to(ROOT)),p.stat().st_size) for p in sorted(files))
print(f'{len(files)} published files indexed, plus file-index.tsv itself; {len(raw)} server raw files; no checksums')
