"""Stage 4E guard with explicit exiting-client handling after an observed Nsight race."""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--gpu', required=True)
p.add_argument('--idle-memory-mib', type=float, default=0)
p.add_argument('--record', type=Path, required=True)
p.add_argument('command', nargs=argparse.REMAINDER)
args = p.parse_args()
command = args.command[1:] if args.command[:1] == ['--'] else args.command
args.record.mkdir(parents=True, exist_ok=False)
fields = 'index,uuid,name,driver_version,memory.total,memory.used,utilization.gpu,clocks.sm,clocks.mem,power.draw,temperature.gpu'


def snapshot():
    gpu = subprocess.check_output(['nvidia-smi', '--id=' + args.gpu,
        '--query-gpu=' + fields, '--format=csv,noheader,nounits'], text=True).strip()
    processes = subprocess.check_output(['nvidia-smi', '--id=' + args.gpu,
        '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
    values = dict(zip(fields.split(','), [s.strip() for s in next(csv.reader([gpu]))]))
    return dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), gpu=values,
                process_ids=[int(v) for v in processes.splitlines()])


before = [snapshot()]
time.sleep(1)
before.append(snapshot())
(args.record / 'before.json').write_text(json.dumps(before, indent=2))
if any(v['process_ids'] or float(v['gpu']['memory.used']) > args.idle_memory_mib or float(v['gpu']['utilization.gpu']) for v in before):
    raise SystemExit('BLOCKED: selected GPU is not idle; no GPU work started')
marker = 'HYTORCH_EXPERIMENT_RUN=' + str(args.record.resolve())
env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, CUDA_DEVICE_ORDER='PCI_BUS_ID',
           HYTORCH_EXPERIMENT_RUN=str(args.record.resolve()))
(args.record / 'launch.json').write_text(json.dumps(dict(command=command, physical_gpu=before[-1]['gpu']['index'],
    logical_gpu=0, uuid=args.gpu, started_at=datetime.now(timezone.utc).isoformat(),
    environment={k:env.get(k) for k in ('CUDA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER','CUPY_TF32','NVIDIA_TF32_OVERRIDE','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS')}), indent=2))
blocked = False


def belongs_to_run(pid):
    # Profilers may put their application in a new process group/session.
    # Inspect only our run marker; never log process environment contents.
    try:
        ancestor = pid
        while ancestor > 1:
            if ancestor == proc.pid:
                return True
            fields = Path(f'/proc/{ancestor}/stat').read_text().rsplit(')', 1)[1].split()
            ancestor = int(fields[1])
        return marker.encode() in Path(f'/proc/{pid}/environ').read_bytes().split(b'\0')
    except (ProcessLookupError, FileNotFoundError, PermissionError):
        return False


with (args.record / 'terminal.log').open('x') as log, (args.record / 'gpu-load.jsonl').open('x') as monitor:
    proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        while proc.poll() is None:
            row = snapshot()
            external = []
            owned = []
            outside_group = []
            exiting = []
            unclassified = []
            for pid in row['process_ids']:
                try:
                    group = os.getpgid(pid)
                except ProcessLookupError:  # A client may exit between the query and getpgid.
                    continue
                if group == proc.pid or belongs_to_run(pid):
                    owned.append(pid)
                    if group != proc.pid:
                        outside_group.append(pid)
                else:
                    # NVML can retain a client while its process exits. A prior
                    # Nsight child lost /proc/environ at this boundary.
                    try:
                        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
                        state, flags = fields[0], int(fields[6])
                    except (FileNotFoundError, ProcessLookupError):
                        exiting.append(pid)
                        continue
                    unclassified.append(dict(pid=pid,state=state,flags=flags,parent=int(fields[1])))
                    if state == 'Z' or flags & 4:  # Linux PF_EXITING; no more user submissions.
                        exiting.append(pid)
                        continue
                    external.append(pid)
            row['unclassified_processes'] = unclassified
            row['exiting_process_ids'] = exiting
            row['external_process_ids'] = external
            row['owned_process_ids'] = owned
            row['owned_outside_group_ids'] = outside_group
            monitor.write(json.dumps(row) + '\n')
            monitor.flush()
            if external:
                blocked = True
                for pid in outside_group:
                    if belongs_to_run(pid):
                        os.kill(pid, signal.SIGTERM)
                os.killpg(proc.pid, signal.SIGTERM)
                break
            time.sleep(1)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
        code = proc.wait()
(args.record / 'completion.json').write_text(json.dumps(dict(returncode=code,
    external_load_interruption=blocked, finished_at=datetime.now(timezone.utc).isoformat()), indent=2))
print((args.record / 'terminal.log').read_text(), end='')
raise SystemExit(75 if blocked else code)
