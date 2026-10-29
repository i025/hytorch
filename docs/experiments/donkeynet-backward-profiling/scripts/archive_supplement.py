"""Archive portable supplement evidence and index server-only raw data, no hashes."""
import argparse
import csv
import json
from pathlib import Path
import shutil


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw', type=Path, required=True)
    p.add_argument('--monitor', type=Path, required=True)
    p.add_argument('--analysis', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    destination = a.output / 'raw'
    destination.mkdir(parents=True, exist_ok=False)
    groups = [('gpu-run', a.raw), ('monitor', a.monitor), ('trace-analysis', a.analysis)]
    root = Path(__file__).resolve().parents[4]
    groups.extend((name.name, name) for name in sorted(root.glob('runs/donkeynet_4a_supplement_trace_parser_check_*')))
    groups.extend((name.name, name) for name in sorted(root.glob('runs/donkeynet_4a_supplement_plot_draft_*')))
    records = []
    for group, source in groups:
        for path in sorted(source.rglob('*')):
            if not path.is_file():
                continue
            relative = path.relative_to(source)
            portable = path.suffix.lower() in {'.json', '.jsonl', '.csv', '.tsv', '.txt', '.log', '.xml'}
            target = destination / group / relative
            if portable:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
                # Direct equality only; no checksum or fingerprint operation.
                if path.read_bytes() != target.read_bytes():
                    raise ValueError('Archive content mismatch: ' + str(path))
            records.append(dict(group=group, server_path=str(path.resolve()),
                                bytes=path.stat().st_size,
                                github_path=str(target.relative_to(a.output)) if portable else '',
                                policy='copied verbatim' if portable else 'server only; binary/weights/inputs'))
    with (a.output/'raw-file-index.tsv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(records)
    result = dict(files=len(records), portable_files=sum(bool(r['github_path']) for r in records),
                  raw_bytes=sum(r['bytes'] for r in records),
                  portable_bytes=sum(r['bytes'] for r in records if r['github_path']),
                  verification='direct byte equality; no hashes', groups=[str(s.resolve()) for _,s in groups])
    (a.output/'archive-summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
