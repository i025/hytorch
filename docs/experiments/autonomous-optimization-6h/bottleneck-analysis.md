# Evidence used to select prototypes

Historical stage 4H measured synchronous phase wall times, in a separate shared-GPU
window. It is guidance only, never a denominator for new speedups:

| I path, per real batch4 | Time |
|---|---:|
| Read/decode/resize/normalize/batch | 3.0408ms |
| H2D transfer | .1403ms |
| Forward | 1.7232ms |
| Backward | 2.6329ms |
| Update | .1253ms |

The separately instrumented I trace has 39 forward and 52 backward kernels/step.
Backward kernel sum is 1071.923us; its host NVTX range is 4224.490us. These are
different clocks/scopes; their difference is not directly GPU utilization or a
measured Python-exclusive cost. Seven ReLU gradients each create a mask then
multiply, producing 14 launches and temporary boolean arrays. Their host submission
and autograd objects make them a small, generic fusion candidate.

## New CPU preprocessing results

Same1024 real images/labels, three independently constructed loaders, rotated order,
no altered augmentation, no disk-cache flush. Exact array equality in all12 runs.

| Worker threads | Images/s mean ± sample SD |
|---:|---:|
| 0 | 1096.82 ± 147.93 |
| 1 | 1092.18 ± 55.00 |
| 2 | 1681.71 ± 45.54 |
| 4 | 1774.12 ± 145.80 |

Two workers were selected before GPU timing as a smaller, less variable configuration.
The first slower serial result is retained. CPU throughput is not a training speedup
and is not compared numerically with old shared-GPU phase timings.

## Scope and rejected directions

No third optimization: do not rewrite another convolution or GEMM kernel based on
kernel time alone. Existing Split-K workspace and dInput paths remain unchanged.
No data split, preprocessing math, optimizer formula or learning strategy is changed.
The two prototypes address separate costs: CPU pipeline waiting and ReLU mask/launch
overhead. P/R/J ablations and a PyTorch-prefetch control are required before attribution.

New GPU operator/phase/throughput/Nsight results will be recorded separately. If
the card remains occupied, they are NOT_RUN and no end-to-end benefit is asserted.
