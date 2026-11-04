# Decisions before GPU execution

## Priority A: diagnose, do not change tolerance

Stage 4H retained final weights, exact row orders and every loss, but no intermediate states.
Collect 1600 updates from each original A/F/I/E implementation, including the first partial
batch and the next epoch. Repeat I and E independently. At 18 selected positions preserve the
entire pre-step state, batch, predictions, loss, gradients and updated state. Replay the
same F and I state at ten positions on all four backends. Stateless SGD has no momentum
buffer; checkpoint metadata records lr=.001, momentum=0, weight_decay=0.

These are sampled checks, not a proof of every possible state. Local errors retain the
existing forward/gradient (1e-5,1e-4) and loss/update (1e-6,1e-5) tolerances. A trajectory
failure is not converted into a local PASS. Isolated real dWeight operands at steps
1/863/1422 are preserved for independent sampled FP64 dot checks after GPU release.

## Two independent performance prototypes

1. **P: ordered CPU prefetch.** Stage 4H measured about 3ms/batch for JPEG decode,
   resize, normalization and batch construction. CPU measurements in this branch
   show 2 workers outperform serial loading with exactly equal pixels/labels.
   Reuse existing AsyncDataLoader and add only an optional `indices` argument to
   preserve the established protocol. Default workers=0 and default indexing stay
   unchanged. No GPU calls in data workers. Iterator startup/drain is timed.
2. **R: native ReLU backward.** Historical I trace has seven ReLU gradients per
   DonkeyNet step, each launching mask + multiply. Use a generic flat FP32 kernel
   to multiply gy by the predicate without allocating the intermediate mask.
   `__fmul_rn` preserves the existing multiply semantics, including signed zero
   and nonfinite inputs. No shape/model/layer specialization. Noncontiguous
   operands use the runtime's existing explicit contiguous materialization.
   `backend(relu_backward='fused')` is opt-in; CPU/non-FP32 execution is visibly
   counted as the existing path, and higher-order graph construction retains
   the original differentiable operations.

**J = I + P + R**. P and R are also measured independently. No changes to existing
convolution, GEMM, Split-K dWeight or SGD kernels. No third optimization is planned.

## Measurement protocol

A/F/I/P/R/J/E, same physical UUID, real Udacity rows/FP32 inputs, seed42 canonical
initialization, batch4, lr=.001, no dropout/TF32, 10 warmup updates and reset to
identical initialization. DonkeyNet and independent SpatialNet both use the same
real 3x66x200 images. Formal compute and full-I/O measurements each run 256 steps
in at least three independent processes, rotating order. Independent synchronized
phases are not used as formal throughput. Nsight is collected separately at the end.
An additional PyTorch two-worker I/O control is included: primary E retains the
stage 4H serial pipeline, while E-prefetch quantifies the same generic loading
benefit for PyTorch. A J/E(serial) gain must not be presented as a CUDA-only gain.

ReLU microbenchmarks measure complete backward calls (not merely native launch),
including layout preparation, CUDA events and synchronization. Event spans include
possible host submission gaps; only CUPTI kernel sums are called kernel time.
Large traces and operand/state NPZ files remain server-only.
