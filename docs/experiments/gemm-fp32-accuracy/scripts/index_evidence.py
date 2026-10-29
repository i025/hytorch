"""Index retained files and actual command metadata without checksums."""
import json
from pathlib import Path
import shlex

root = Path(__file__).resolve().parents[1]
records = []
for path in root.rglob('*.command.json'):
    meta = json.loads(path.read_text())
    records.append((meta['started_at'], path, meta))
lines = ['# 逐次真实执行记录', '',
         '以下从即时生成的命令元数据整理。时间为 UTC；退出码非零的失败记录完整保留。', '',
         '每个 CPU 命令由 `docs/experiments/a100-stage1/run_logged.py --log <对应日志> -- <命令>` 执行。', '']
for start, path, meta in sorted(records):
    relative = path.relative_to(root)
    log = str(relative)[:-len('.command.json')]
    lines.extend([f"## {start} — exit {meta['returncode']}", '',
                  f'[原始日志]({log}) · [完整命令/环境元数据]({relative})', '',
                  '```bash', shlex.join(meta['command']), '```', ''])
launch = json.loads((root / 'evidence/gpu-monitor-01/launch.json').read_text())
completion = json.loads((root / 'evidence/gpu-monitor-01/completion.json').read_text())
lines.extend([f"## GPU {launch['started_at']} — exit {completion['returncode']}", '',
              '[原始日志](evidence/gpu-monitor-01/benchmark.log) · [启动](evidence/gpu-monitor-01/launch.json) · [结束](evidence/gpu-monitor-01/completion.json)', '',
              '```bash', shlex.join(launch['command']), '```', '',
              '外层 GPU 空闲检查/隔离、归档与其他操作见 [commands.md](commands.md)。'])
(root / 'command-records.md').write_text('\n'.join(lines) + '\n')
files = [p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name != 'file-index.tsv']
with (root / 'file-index.tsv').open('w') as stream:
    stream.write('path\tbytes\n')
    for path in sorted(files):
        stream.write(f'{path.relative_to(root)}\t{path.stat().st_size}\n')
print(f'Indexed {len(files)} files; index excludes itself and Python bytecode cache')
