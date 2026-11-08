"""Conv2d/MaxPool forward scaling with NumPy, CuPy and native CUDA C."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmarks.bench_cuda_gemm import cpu_time, gpu_time, errors
from hytorch.functions_conv import conv2d_cpu_tensordot_array, Pooling
from hytorch.backends.cuda.runtime import conv2d, max_pool2d, load_library


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batches",nargs="+",type=int,default=[1,4,16])
    p.add_argument("--repeats",type=int,default=3)
    p.add_argument("--device",type=int,default=0)
    p.add_argument("--output",default="runs/cuda_ops")
    args=p.parse_args()
    if min(args.batches)<=0 or args.repeats<3:
        p.error("positive batches and at least 3 repeats required")
    import cupy as cp
    cp.cuda.Device(args.device).use(); load_library()
    out=Path(args.output); out.mkdir(parents=True,exist_ok=False)
    rows=[]
    for n in args.batches:
        rng=np.random.default_rng(42+n)
        x=rng.normal(size=(n,3,66,200)).astype(np.float32)
        w=rng.normal(size=(24,3,5,5)).astype(np.float32)
        b=rng.normal(size=(24,)).astype(np.float32)
        dx,dw,db=cp.asarray(x),cp.asarray(w),cp.asarray(b)
        for op in ("conv","pool"):
            cpu_fn=(lambda:conv2d_cpu_tensordot_array(x,w,b,2,0)) if op=="conv" else (
                lambda:Pooling(2,2,0).forward(x))
            reference,cpu_ms=cpu_time(cpu_fn,args.repeats)
            functions=([("cupy",lambda:conv2d_cpu_tensordot_array(dx,dw,db,2,0,xp=cp)),
                        ("cuda_direct",lambda:conv2d(dx,dw,db,2,0,"direct")),
                        ("cuda_im2col_16",lambda:conv2d(dx,dw,db,2,0,"im2col",16)),
                        ("cuda_im2col_32",lambda:conv2d(dx,dw,db,2,0,"im2col",32))] if op=="conv" else
                       [("cupy",lambda:Pooling(2,2,0).forward(dx)),
                        ("cuda_native",lambda:max_pool2d(dx,2,2)[0])])
            entries=[("numpy",reference,cpu_ms)]
            for name,fn in functions:
                actual,times=gpu_time(fn,cp,args.repeats)
                entries.append((name,cp.asnumpy(actual),times))
            cupy_ms=statistics.mean(entries[1][2])
            for name,actual,times in entries:
                absolute,relative=errors(actual,reference)
                mean=statistics.mean(times)
                rows.append(dict(op=op,batch=n,input_shape=json.dumps(x.shape),implementation=name,
                                 mean_ms=mean,std_ms=statistics.stdev(times),repeats_ms=json.dumps(times),
                                 speedup_numpy=statistics.mean(cpu_ms)/mean,speedup_cupy=cupy_ms/mean,
                                 max_abs_error=absolute,relative_l2_error=relative,
                                 correct=bool(np.allclose(actual,reference,rtol=1e-4,atol=1e-4))))
        print(f"Finished batch={n}",flush=True)
    with (out/"ops.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (out/"config.json").write_text(json.dumps(dict(vars(args),dtype="float32",seed="42+batch",
        timing="CUDA event, includes operator workspace allocation/packing; no H2D/D2H",
        gpu=str(cp.cuda.runtime.getDeviceProperties(args.device)["name"]),cupy=cp.__version__),indent=2),encoding="utf-8")
    if not all(r["correct"] for r in rows):
        raise SystemExit("Numerical check failed; inspect ops.csv")


if __name__=="__main__":
    main()
