"""Compile only the independent kernel and retain nvcc resource diagnostics."""
import argparse
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
args = p.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
source = Path(__file__).with_name('kernel.cu').resolve()
library = args.output.resolve() / 'libblock32_tree.so'
if library.exists():
    raise FileExistsError(library)
command = ['/usr/local/cuda-12.2/bin/nvcc', '-O3', '-std=c++14', '-lineinfo',
           '--shared', '--cudart', 'shared', '-Xcompiler', '-fPIC',
           '-gencode=arch=compute_80,code=sm_80', '-gencode=arch=compute_80,code=compute_80',
           '--ptxas-options=-v,-warn-spills', str(source), '-o', str(library)]
started = datetime.now(timezone.utc).isoformat()
with (args.output / 'build.log').open('x') as stream:
    result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, text=True)
meta = dict(command=command, started_at=started, finished_at=datetime.now(timezone.utc).isoformat(),
            returncode=result.returncode, architecture='sm_80', library=str(library),
            production_kernel_modified=False, maximum_k=4096, partial_state_slots=8)
with (args.output / 'build.json').open('x') as stream:
    json.dump(meta, stream, indent=2)
print((args.output / 'build.log').read_text())
raise SystemExit(result.returncode)
