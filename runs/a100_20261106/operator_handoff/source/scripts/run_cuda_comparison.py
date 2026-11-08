"""Run matched experiments in separate processes, repeated three times by default."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True)
    p.add_argument("--data")
    p.add_argument("--model",choices=["tiny","donkey"],default="donkey")
    p.add_argument("--samples",type=int,default=64)
    p.add_argument("--batch",type=int,default=4)
    p.add_argument("--epochs",type=int,default=1)
    p.add_argument("--workers",type=int,default=0)
    p.add_argument("--device",type=int,default=0)
    p.add_argument("--repeats",type=int,default=3)
    p.add_argument("--mode",choices=["phases","throughput"],default="throughput")
    p.add_argument("--include-torch",action="store_true")
    args=p.parse_args()
    if args.repeats<1:
        p.error("repeats must be positive")
    output=Path(args.output).resolve(); output.mkdir(parents=True,exist_ok=False)
    variants=[("cupy",[]),("conv_direct",["--conv","direct"]),
              ("conv_im2col",["--conv","im2col"]),("gemm_shared",["--matmul","shared"])]
    if args.model == "tiny":
        variants.append(("pool_native",["--pool","native"]))
    combined=["--matmul","shared","--conv","im2col"]
    if args.model == "tiny":
        combined += ["--pool","native"]
    variants.append(("combined",combined))
    if args.include_torch:
        variants.append(("torch",["--framework","torch"]))
    commands=[]
    for repeat in range(args.repeats):
        # Rotate order across repeats to reduce systematic warm/thermal ordering bias.
        ordered=variants[repeat%len(variants):]+variants[:repeat%len(variants)]
        for name,flags in ordered:
            run=output/f"{name}_r{repeat+1}"
            command=[sys.executable,str(ROOT/"benchmarks/bench_cuda_training.py"),
                     "--output",str(run),"--gpu",str(args.device),"--model",args.model,
                     "--samples",str(args.samples),"--batch",str(args.batch),"--epochs",str(args.epochs),
                     "--workers",str(args.workers),"--mode",args.mode]+flags
            if args.data:
                command += ["--data",str(Path(args.data).resolve())]
            commands.append(command)
            (output/"commands.json").write_text(json.dumps(commands,indent=2),encoding="utf-8")
            print(f"Running {name}, repeat {repeat+1}",flush=True)
            with (output/f"{name}_r{repeat+1}.log").open("w",encoding="utf-8") as log:
                subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)


if __name__=="__main__":
    main()
