"""Record one experiment command without overwriting a previous log."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a command is required")
    path = Path(args.log)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(command=command, cwd=os.getcwd(),
                    started_at=datetime.now(timezone.utc).isoformat(),
                    environment={key: os.environ.get(key) for key in (
                        "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "CUDA_PATH",
                        "LD_LIBRARY_PATH", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                        "HYTORCH_GEMM_BUILD_DIR", "MPLCONFIGDIR", "CUPY_CACHE_DIR",
                        "CUDA_CACHE_PATH", "PYTHONPATH", "PYTEST_ADDOPTS", "HYTORCH_NUMERIC_AUDIT")})
    start = time.monotonic()
    with path.open("xb") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    metadata.update(returncode=result.returncode, elapsed_s=time.monotonic()-start,
                    finished_at=datetime.now(timezone.utc).isoformat())
    with path.with_suffix(path.suffix + ".command.json").open("x") as output:
        json.dump(metadata, output, indent=2)
    print(path.read_text(errors="replace"), end="")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
