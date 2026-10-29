"""Copy portable raw records; index large server-only files without checksums."""
import argparse
import csv
import gzip
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[4]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--destination', type=Path, required=True)
p.add_argument('sources', type=Path, nargs='+')
a = p.parse_args()
a.destination.mkdir(parents=True, exist_ok=False)
text_extensions = {'.txt', '.log', '.json', '.jsonl', '.csv', '.xml', '.md'}
rows = []
for source in a.sources:
    for file in sorted(source.rglob('*')):
        if not file.is_file():
            continue
        size = file.stat().st_size
        portable = ''
        if file.suffix in text_extensions:
            target = a.destination/source.name/file.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            if size > 1_000_000:
                target = target.with_suffix(target.suffix+'.gz')
                with file.open('rb') as src, target.open('xb') as dst:
                    with gzip.GzipFile(fileobj=dst, mode='wb', mtime=0) as zipped:
                        shutil.copyfileobj(src, zipped)
            else:
                with file.open('rb') as src, target.open('xb') as dst:
                    shutil.copyfileobj(src, dst)
            portable = str(target.relative_to(ROOT)) if target.is_absolute() else str(target)
        rows.append(dict(server_path=str(file.resolve()), bytes=size,
                         github_record=portable,
                         storage='Git portable record and server original' if portable else 'server original only'))
with (a.destination/'file-index.csv').open('x', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print('Indexed', len(rows), 'server files; copied', sum(bool(r['github_record']) for r in rows), 'portable records')
