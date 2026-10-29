"""Build the CUDA C library on Linux/WSL2 or a Windows x64 MSVC prompt."""
import argparse
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def build_command(nvcc, output, arch):
    if not re.fullmatch(r"[0-9]{2,3}[a-z]?", arch):
        raise ValueError("--arch must be a CUDA compute capability, e.g. 86 or 89")
    command = [nvcc, "-O3", "-std=c++14", "-lineinfo", "--shared", "--cudart", "shared",
               f"-gencode=arch=compute_{arch},code=sm_{arch}",
               f"-gencode=arch=compute_{arch},code=compute_{arch}"]
    if os.name != "nt":
        command += ["-Xcompiler", "-fPIC"]
    return command + [str(ROOT / "hytorch/backends/cuda/kernels.cu"),
                      str(ROOT / "hytorch/backends/cuda/fused_sgd.cu"),
                      str(ROOT / "hytorch/backends/cuda/conv_dinput.cu"), "-o", str(output)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", help="SM compute capability without dot; auto-detect if omitted")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resource-usage", action="store_true", help="Include nvcc/ptxas resource report")
    p.add_argument("--output", type=Path,
                   help="Directory for the library, build.log and build.json; defaults to the CUDA build directory")
    args = p.parse_args()
    arch = args.arch
    if arch is None:
        if args.dry_run:
            p.error("--dry-run requires --arch")
        import cupy as cp
        arch = cp.cuda.Device(args.device).compute_capability
    output_dir = args.output or ROOT / "hytorch/backends/cuda/build"
    output = output_dir.resolve() / ("hytorch_cuda.dll" if os.name == "nt" else "libhytorch_cuda.so")
    nvcc = shutil.which("nvcc")
    command = build_command(nvcc or "nvcc", output, arch)
    if args.resource_usage:
        command.insert(1, "--resource-usage")
    print(json.dumps(command, indent=2))
    if args.dry_run:
        return
    if sys.platform == "darwin" or not nvcc:
        p.error("Build requires NVIDIA CUDA Toolkit/nvcc on Linux or Windows; Mac cannot build this library")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(command, cwd=output.parent, text=True, capture_output=True)
    (output.parent / "build.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    print(result.stdout + result.stderr)
    result.check_returncode()
    info = {"command": command, "platform": platform.platform(), "arch": arch,
            "nvcc": subprocess.check_output([nvcc, "--version"], text=True)}
    (output.parent / "build.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"Built: {output}")


if __name__ == "__main__":
    main()
