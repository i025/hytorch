"""Index original guard samples and their overlap with recorded subprocesses."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    windows=[];commands=[]
    for root in sorted(a.runs.glob('conv_dinput_opt_*')):
        for file in sorted(root.rglob('gpu-load.jsonl')):
            rows=[json.loads(line) for line in file.read_text().splitlines()]
            completion=file.parent/'completion.json'
            launch=json.loads((file.parent/'launch.json').read_text())
            windows.append(dict(root=str(root.resolve()),monitor=str(file.resolve()),samples=len(rows),
                external_samples=sum(bool(r['external_process_ids']) for r in rows),
                uuid=launch['uuid'],physical_gpu=launch['physical_gpu'],logical_gpu=launch['logical_gpu'],
                started_at=launch['started_at'],completion=json.loads(completion.read_text()) if completion.exists() else None,
                sampled_peak_card_memory_mib=max(float(r['gpu']['memory.used']) for r in rows)))
            for file in root.rglob('commands.json'):
                for c in json.loads(file.read_text()):
                    if 'finished_at' not in c:continue
                    start,end=[datetime.fromisoformat(c[k]) for k in ('started_at','finished_at')]
                    selected=[r for r in rows if start<=datetime.fromisoformat(r['timestamp_utc'])<=end]
                    commands.append(dict(root=str(root.resolve()),name=c['name'],uuid=launch['uuid'],
                        returncode=c['returncode'],started_at=c['started_at'],finished_at=c['finished_at'],samples=len(selected),
                        sampled_peak_card_memory_mib=max([float(r['gpu']['memory.used']) for r in selected],default=None),
                        min_sm_mhz=min([float(r['gpu']['clocks.sm']) for r in selected],default=None),
                        max_sm_mhz=max([float(r['gpu']['clocks.sm']) for r in selected],default=None),
                        sampled_peak_power_w=max([float(r['gpu']['power.draw']) for r in selected],default=None),
                        sampled_peak_temperature_c=max([float(r['gpu']['temperature.gpu']) for r in selected],default=None)))
    with (a.output/'commands.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(commands[0]));writer.writeheader();writer.writerows(commands)
    (a.output/'windows.json').write_text(json.dumps(windows,indent=2)+'\n')
    (a.output/'scope.txt').write_text('nvidia-smi whole-card samples at approximately 1.2-second intervals, not allocator/process peaks.\n'
        'A subprocess with zero overlapping samples has empty hardware fields; no interpolation.\n'
        'External flags reflect the guard at that time; see report for exiting-client attribution caveats.\n')
    print('Indexed',len(windows),'guard windows and',len(commands),'subprocess hardware intervals')


if __name__=='__main__':main()
