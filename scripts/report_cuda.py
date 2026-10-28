"""Generate figures and comparison tables from real logs; never synthesize timings."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
import numpy as np


def read_rows(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


def compare_training(paths):
    runs=[]
    matched_keys=("initial_state_sha256","dataset_sha256","model","batch","epochs","steps",
                  "workers","seed","lr","mode","device","gpu","gpu_name","physical_monitor_id",
                  "warmup","no_monitor","interval","nvtx","optimizer","dropout","dtype","platform")
    for path in paths:
        config=json.loads((path/"config.json").read_text(encoding="utf-8"))
        summary=json.loads((path/"summary.json").read_text(encoding="utf-8"))
        if summary.get("status")!="COMPLETED":
            raise ValueError(f"Incomplete run: {path}")
        runs.append((path,config,summary))
    first=runs[0][1]
    for path,config,_ in runs[1:]:
        mismatch=[k for k in matched_keys if config.get(k)!=first.get(k)]
        if mismatch:
            raise ValueError(f"Unmatched training conditions in {path}: {mismatch}")
    groups={}
    for path,config,summary in runs:
        label=("torch" if config["framework"]=="torch" else
               f"hytorch:m={config['matmul']},c={config['conv']},p={config['pool']},t={config['tile']}")
        wall=sum(e["train_wall_s"] for e in summary["epochs"])
        groups.setdefault(label,[]).append((wall,summary,path))
    baseline_key="hytorch:m=cupy,c=cupy,p=cupy,t=16"
    baseline=statistics.mean(v[0] for v in groups[baseline_key]) if baseline_key in groups else None
    rows=[]
    reference_path=groups[baseline_key][0][2] if baseline_key in groups else runs[0][0]
    with np.load(reference_path/"final_weights.npz",allow_pickle=False) as state:
        reference={k:state[k] for k in state.files}
    for label,values in groups.items():
        times=[v[0] for v in values]
        max_difference=0.
        correct=True
        for _,_,path in values:
            with np.load(path/"final_weights.npz",allow_pickle=False) as final:
                if set(final.files)!=set(reference):
                    raise ValueError("Mismatched final parameter names")
                for key in reference:
                    if final[key].shape!=reference[key].shape:
                        raise ValueError("Mismatched final parameter shapes")
                    max_difference=max(max_difference,float(np.max(np.abs(final[key]-reference[key]))))
                    correct=correct and bool(np.allclose(final[key],reference[key],atol=1e-4,rtol=1e-3))
        total_samples=sum(sum(e["samples"] for e in v[1]["epochs"]) for v in values)
        rows.append(dict(implementation=label,repeats=len(values),mean_train_wall_s=statistics.mean(times),
                         std_train_wall_s=statistics.stdev(times) if len(times)>1 else None,
                         speedup_vs_hytorch_cupy=baseline/statistics.mean(times) if baseline else None,
                         samples_per_sec=total_samples/sum(times),
                         final_weight_max_abs_difference=max_difference,final_weights_close=correct,
                         gpu_status=values[0][1]["gpu_status"],data_kind=first["data_kind"]))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gemm",help="directory containing gemm.csv")
    p.add_argument("--training",nargs="*",default=[],help="run directories with config/summary/steps")
    p.add_argument("--comparison",help="root from run_cuda_comparison.py")
    p.add_argument("--output",required=True)
    args=p.parse_args()
    if not (args.gemm or args.training or args.comparison):
        p.error("Provide at least one input")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out=Path(args.output); out.mkdir(parents=True,exist_ok=False)
    if args.gemm:
        with (Path(args.gemm)/"gemm.csv").open(encoding="utf-8") as f:
            rows=list(csv.DictReader(f))
        fig,axes=plt.subplots(1,2,figsize=(11,4.2))
        labels=sorted({r["implementation"]+("-"+r["tile"] if r["tile"] else "") for r in rows})
        for label in labels:
            selected=[r for r in rows if r["implementation"]+("-"+r["tile"] if r["tile"] else "")==label]
            selected.sort(key=lambda r:int(r["size"]))
            x=[int(r["size"]) for r in selected]
            axes[0].plot(x,[float(r["mean_ms"]) for r in selected],"o-",label=label)
            axes[1].plot(x,[float(r["speedup_numpy"]) for r in selected],"o-",label=label)
        axes[0].set(ylabel="Mean compute time (ms)",yscale="log")
        axes[1].set(ylabel="Speedup vs NumPy")
        for ax in axes:
            ax.set_xlabel("Square matrix size"); ax.grid(alpha=.2); ax.legend(fontsize=8)
        title="GEMM measured results" if any(r["implementation"]=="cupy" for r in rows) else "CPU ONLY - CUDA NOT EXECUTED"
        fig.suptitle(title); fig.tight_layout(); fig.savefig(out/"gemm_scaling.png",dpi=180); plt.close(fig)
    paths=[Path(p) for p in args.training]
    if args.comparison:
        paths+=sorted(p.parent for p in Path(args.comparison).glob("*/summary.json"))
    if paths:
        rows=compare_training(paths)
        with (out/"training_comparison.csv").open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        fig,ax=plt.subplots(figsize=(11,max(3,.55*len(rows))))
        ax.barh([r["implementation"] for r in rows],[r["mean_train_wall_s"] for r in rows])
        ax.set_xlabel("Mean training wall time (s), excludes validation")
        ax.set_title("Matched training: "+rows[0]["data_kind"]+" / GPU "+rows[0]["gpu_status"])
        fig.tight_layout(); fig.savefig(out/"training_comparison.png",dpi=160); plt.close(fig)
        for path in paths:
            steps=read_rows(path/"steps.jsonl")
            if not steps:
                continue
            fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True)
            times=[r["timestamp"]-steps[0]["timestamp"] for r in steps]
            phase_names=("data","transfer","forward","backward","update","other")
            if steps[0]["mode"]=="phases":
                axes[0].stackplot(times,*[[r[p+"_s"]*1000 for r in steps] for p in phase_names],labels=phase_names)
            else:
                axes[0].plot(times,[r["step_s"]*1000 for r in steps],label="step")
            axes[0].set_ylabel("Step / phase time (ms)"); axes[0].legend(ncol=3,fontsize=8)
            hardware=read_rows(path/"hardware.jsonl") if (path/"hardware.jsonl").exists() else []
            for key in ("gpu_utilization_pct","cpu_utilization_pct"):
                available=[r for r in hardware if r.get(key) is not None]
                if available:
                    axes[1].plot([r["timestamp"]-steps[0]["timestamp"] for r in available],
                                 [r[key] for r in available],".-",label=key)
            axes[1].set(xlabel="Elapsed wall time (s)",ylabel="Utilization (%)")
            handles,_=axes[1].get_legend_handles_labels()
            if handles:
                axes[1].legend(fontsize=8)
            else:
                axes[1].text(.5,.5,"No hardware samples available",ha="center",transform=axes[1].transAxes)
            fig.suptitle(path.name); fig.tight_layout(); fig.savefig(out/f"{path.name}_timeline.png",dpi=160); plt.close(fig)
    print(f"Reports: {out}")


if __name__=="__main__":
    main()
