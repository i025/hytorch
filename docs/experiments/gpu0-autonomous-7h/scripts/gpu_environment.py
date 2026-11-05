"""GPU metadata only after this stage's idle-device gate."""
import argparse
from datetime import datetime,timezone
import importlib.metadata as md
import json
import os
from pathlib import Path
import platform


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--run-gpu',action='store_true',required=True)
    a=p.parse_args()
    import cupy as cp
    cp.cuda.Device(0).use()
    info=cp.cuda.runtime.getDeviceProperties(0)
    name=info['name'].decode() if isinstance(info['name'],bytes) else info['name']
    result=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),visible_uuid=os.environ['CUDA_VISIBLE_DEVICES'],
        logical_device=0,name=name,compute_capability=[info['major'],info['minor']],
        total_bytes=info['totalGlobalMem'],runtime_version=cp.cuda.runtime.runtimeGetVersion(),
        driver_api_version=cp.cuda.runtime.driverGetVersion(),python=platform.python_version(),
        packages={n:md.version(n) for n in ('numpy','cupy-cuda12x','torch')},
        tf32_override=os.environ['NVIDIA_TF32_OVERRIDE'])
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
