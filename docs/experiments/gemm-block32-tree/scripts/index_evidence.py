"""List all retained artifacts and reconstruct commands from run metadata."""
import json
from pathlib import Path
import shlex

ROOT=Path(__file__).resolve().parents[4]
DOCS=ROOT/'docs/experiments/gemm-block32-tree'
records=[]
for path in DOCS.rglob('*.command.json'):
    meta=json.loads(path.read_text())
    relative=path.relative_to(DOCS)
    log=str(relative)[:-len('.command.json')]
    command=['python','docs/experiments/a100-stage1/run_logged.py','--log',str(path.relative_to(ROOT))[:-len('.command.json')],'--']+meta['command']
    records.append((meta['started_at'],f"exit={meta['returncode']}",command,log,str(relative)))
for path in sorted((DOCS/'logs').glob('*-monitor-*/launch.json')):
    launch=json.loads(path.read_text())
    completion=json.loads(path.with_name('completion.json').read_text())
    phase,_,version=path.parent.name.rpartition('-monitor-')
    record=f'runs/gemm_block32_tree_{phase}_monitor_{version}'
    command=['python','docs/experiments/gemm-block32-tree/scripts/run_gpu.py','--gpu',launch['uuid'],'--record',record,'--']+launch['command']
    wrapper_exit=75 if completion['external_load_interruption'] else completion['returncode']
    records.append((launch['started_at'],f"wrapper exit={wrapper_exit}, child exit={completion['returncode']}",command,
        str(path.with_name('terminal.log').relative_to(DOCS)),str(path.relative_to(DOCS))))
lines=['# 真实运行命令索引','','下列时间来自即时执行元数据，均为 UTC（北京时间加 8 小时）。错误、中断和最终成功均保留。','']
for start,status,command,log,meta in sorted(records):
    lines.extend([f'## {start} — {status}','','[原始输出]('+log+') · [命令元数据]('+meta+')','','```bash',shlex.join(command),'```',''])
lines.extend(['独立 nvcc 原始命令另见 [build.json](environment/build.json)。分支操作、源码读取及执行门槛见 [commands.md](commands.md)。'])
(DOCS/'command-records.md').write_text('\n'.join(lines)+'\n')
raw=[]
for path in sorted((ROOT/'runs').glob('gemm_block32_tree_*')):
    raw.extend(p for p in path.rglob('*') if p.is_file()) if path.is_dir() else raw.append(path)
with (DOCS/'raw-file-index.tsv').open('w') as stream:
    stream.write('server_path\tbytes\n')
    for path in sorted(raw):stream.write(f'{path}\t{path.stat().st_size}\n')
files=[p for p in DOCS.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='file-index.tsv']
files += [p for p in (ROOT/'experiments/gemm_block32_tree').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
with (DOCS/'file-index.tsv').open('w') as stream:
    stream.write('repository_path\tbytes\n')
    for path in sorted(files):stream.write(f'{path.relative_to(ROOT)}\t{path.stat().st_size}\n')
print(f'{len(files)} experiment files indexed, plus file-index.tsv itself; {len(raw)} raw server files; no checksums')
