# Data index

Server root: `/home/xymo/projects/hytorch/runs/autonomous_optimization_01/`
(canonical `/srv/lab-disks/disk2/users/xymo/projects/hytorch/runs/autonomous_optimization_01/`).

| Path under server root | Contents / status |
|---|---|
| `environment.txt`, `python-packages.txt` | Full CPU/toolkit/software environment; existing venv |
| `host-baseline.log*`, `host-prototypes.log*`, `host-final.log*`, `host-additional.log*` | Full test output and exact command metadata, no discarded failure |
| `build/`, `build-prototypes/` | Baseline and prototype SM80 build logs/JSON/native libraries; no binaries in Git |
| `cpu-preparation/` | 12 loader timing/equality checks, exact protocol reference, 6330-image first-epoch CPU NPZ (large, server only) |
| `cpu-state-diagnosis/` | 510 same-state CPU comparisons, 1020 FP64 comparisons and complete predicted/gradient/update arrays |
| `cpu-trajectory/` | Three1600-step loss series, 18 common-state replay records, parameter differences and checkpoint/final NPZ |
| `sgd-rounding/` | Same-parameter/gradient isolated SGD scalar data and sampled libm.fmaf checks |
| `quality-analysis/` | Unchanged historical held-out labels/predictions, moments/MSE decomposition/bins |
| `availability.jsonl` | Initial all-card observations, no CUDA allocation |
| `gpu-window/` | Interrupted17:54:57–17:55:02 GPU0 window; first job SIGTERM on new external process, no completed jobs |
| `availability-retry.jsonl`, `gpu-controller-retry.log*` | Continuing same-GPU0 observation; no memory reservation |
| `gpu-window-02/` | Will exist only if sustained idle permits retry; status is not inferred from prepared commands |

Portable CPU raw records and every NPZ/binary's absolute server path/size are indexed
in [cpu-raw-01/file-index.csv](results/cpu-raw-01/file-index.csv),87 files/19 portable
records. All arrays remain on server. Read the summary's scope: CPU comparisons do
not become GPU results. First interrupted raw records and final complete GPU/raw
index are added after the corresponding processes finish, preserving old records.

Historical sources are linked, never overwritten:

- `/home/xymo/projects/hytorch/runs/udacity_real_prepared_01/` — original protocol,
  manifest,1024 throughput inputs and full dataset row orders.
- `/home/xymo/projects/hytorch/runs/udacity_real_window_02/` — original4H initial/final
  weights,3-step GPU arrays,4749-step trajectories and Nsight reports.
- `/home/xymo/projects/hytorch/runs/udacity_real_analysis_01/` — original SQLite,
  correlationId-attributed CSV exports.
- [4H report](../udacity-real-training/README.md),[original file index](../udacity-real-training/data-index.md).
