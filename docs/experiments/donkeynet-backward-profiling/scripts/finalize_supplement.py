"""Check deliverable structure and write a size/path index without hashes."""
import ast
import csv
import json
from pathlib import Path
import re

BASE = Path(__file__).resolve().parent.parent / 'supplement-01'


def main():
    files = sorted(p for p in BASE.rglob('*') if p.is_file())
    json_count = 0
    for path in files:
        if path.suffix == '.json':
            json.loads(path.read_text())
            json_count += 1
        elif path.suffix == '.jsonl':
            for line in path.read_text().splitlines():
                json.loads(line)
        if path.suffix in {'.npz', '.npy', '.so', '.sqlite', '.nsys-rep', '.qdstrm'}:
            raise ValueError('Server-only binary in publication: ' + str(path))
        if path.stat().st_size > 50_000_000:
            raise ValueError('Unexpected large publication file: ' + str(path))
        if path.suffix != '.png':
            text = path.read_text()
            if re.search(r'(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY)', text):
                raise ValueError('Credential-like content detected in ' + str(path))
    for path in BASE.glob('*.md'):
        for target in re.findall(r'\]\(([^)]+)\)', path.read_text()):
            target = target.split('#', 1)[0]
            if target and not target.startswith(('https:', 'http:')) and target != 'file-index.tsv':
                if not (path.parent / target).exists():
                    raise ValueError('Broken link: ' + str(path) + ' -> ' + target)
    scripts = sorted((BASE.parent/'scripts').glob('*.py'))
    for path in scripts:
        ast.parse(path.read_text(), filename=str(path))
    rows = list(csv.DictReader((BASE/'raw-file-index.tsv').open(), delimiter='\t'))
    for row in rows:
        server = Path(row['server_path'])
        if not server.is_file() or server.stat().st_size != int(row['bytes']):
            raise ValueError('Raw file index size/path mismatch: ' + str(server))
        if row['github_path'] and (BASE/row['github_path']).stat().st_size != int(row['bytes']):
            raise ValueError('Published archive size mismatch: ' + row['github_path'])
    result = dict(status='PASSED', json_files=json_count, python_scripts=len(scripts),
                  raw_files_indexed=len(rows), png_figures=len(list((BASE/'figures').glob('*.png'))),
                  checks=['JSON/JSONL syntax', 'relative report links', 'Python AST syntax',
                          'raw index existence and byte sizes', 'no server-only binaries',
                          'credential pattern scan'], hashes='NOT_PERFORMED')
    (BASE/'delivery-check.json').write_text(json.dumps(result, indent=2)+'\n')
    with (BASE/'file-index.tsv').open('x', newline='') as stream:
        writer = csv.writer(stream, delimiter='\t')
        writer.writerow(['path', 'bytes'])
        for path in sorted(BASE.rglob('*')):
            if path.is_file() and path.name != 'file-index.tsv':
                writer.writerow([str(path.relative_to(BASE)), path.stat().st_size])
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
