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
| `cpu-engine-recheck/` | Generic model initialization avoids a discarded model;1530 saved CPU arrays match the first diagnosis exactly |
| `loading-profile/` | Serial CPU preprocessing cProfile, CSV, text and server-only pstats |
| `availability.jsonl` | Initial all-card observations, no CUDA allocation |
| `gpu-window/` | Interrupted17:54:57–17:55:02 GPU0 window; first job SIGTERM on new external process, no completed jobs |
| `availability-retry.jsonl`, `gpu-controller-retry.log` | Same-GPU0 observation17:58–18:44UTC; stops after precheck failure. Outer command metadata was not finalized and is absent |
| `gpu-window-02/` | Directory created then precheck failed; no CUDA child or numerical/performance records |
| `unfinished-epoch-timing.patch` | Unvalidated timing draft, server only, not used in measurements; delivery restores committed benchmark |
| `final-status.json` | PARTIAL,103 CPU tests pass, GPU/throughput/Nsight NOT_RUN, six-hour wall delivery not met, push blocked by credentials |

Portable CPU raw records and every NPZ/binary's absolute server path/size are indexed
in [cpu-raw-01/file-index.csv](results/cpu-raw-01/file-index.csv),87 files/19 portable
records. All arrays remain on server. Read the summary's scope: CPU comparisons do
not become GPU results. First interrupted records are in
[resource-interruption-01](results/resource-interruption-01/file-index.csv).
The final archive is[final-raw-01/file-index.csv](results/final-raw-01/file-index.csv):
198 server files,81 portable records. It includes all completed CPU logs, both
resource interruptions and availability samples, not a successful GPU experiment.

[cpu-additional-01/file-index.csv](results/cpu-additional-01/file-index.csv) adds52
server files/6 portable records for initialization recheck and loading profile.

The original Git push failure is preserved in[push.log](logs/publication-01/push.log)
and[command/status JSON](logs/publication-01/push.json), with server originals in
`/home/xymo/projects/hytorch/runs/autonomous_publication_01/`. No token is recorded.

After the user restored credentials, the archived branch was pushed successfully
and remote commit `25c910751533e84b163d9f1877ab52a7ded58333` was verified. See
[successful push output](logs/publication-02/push.log),
[remote verification](logs/publication-02/remote.log) and
[publication metadata](logs/publication-02/publication.json). Server copies are in
`/home/xymo/projects/hytorch/runs/autonomous_publication_02/`; final documentation
push/verification logs also stay there. No experiments were restarted. The archived
`final-raw-01/` status remains the original pre-publication snapshot; the current
[final-status.json](results/final-status.json) records successful publication.

Historical sources are linked, never overwritten:

- `/home/xymo/projects/hytorch/runs/udacity_real_prepared_01/` — original protocol,
  manifest,1024 throughput inputs and full dataset row orders.
- `/home/xymo/projects/hytorch/runs/udacity_real_window_02/` — original4H initial/final
  weights,3-step GPU arrays,4749-step trajectories and Nsight reports.
- `/home/xymo/projects/hytorch/runs/udacity_real_analysis_01/` — original SQLite,
  correlationId-attributed CSV exports.
- [4H report](../udacity-real-training/README.md),[original file index](../udacity-real-training/data-index.md).
