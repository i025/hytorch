# Source facts and numerical mechanisms

Baseline: `experiment/real-udacity-training`, `b7446c6046c084b4a28d826b605df4a8761a7152`.

## Training protocol and actual calls

The stage 4H `engine.py` uses the same explicit seed42 FP32 initialization from
`benchmarks/bench_cuda_training.py::initialize` for every framework. DonkeyNet has
five Conv2d/ReLU blocks, flatten, Linear100/ReLU, Linear50/ReLU, Linear1/Tanh.
Dropout is constructed with p=0. Labels are one steering value per image, so
HyTorch's squared-error sum divided by batch length equals PyTorch mean MSE.
SGD is lr=.001 with no momentum, decay or hooks. There are 16 parameter arrays,
251749 scalar parameters. No optimizer buffers need to be reconstructed.

`layers.py::Linear.forward` calls `linear_simple`, which uses MatMul plus broadcast
bias addition. `functions.py::MatMul.backward` deliberately constructs
`MatMul(native=False)` for both derivatives; these remain CuPy dot/cuBLAS even
when native forward GEMM is enabled. This experiment does not change that choice.

A uses original CuPy forward/backward and ordinary SGD. F selects shared GEMM,
im2col forward, tile16, original native dInput gather, Fused SGD and explicit
first-input pruning. I adds Split-K dWeight. PyTorch uses matching architecture,
cuDNN deterministic mode, no benchmark mode and no TF32. `NVIDIA_TF32_OVERRIDE=0`
is also set for new GPU jobs. No model/data/optimizer mathematical change is made.

## FP32 differences that are confirmed by source

- Native GEMM in `kernels.cu` accumulates FP32 products along its K traversal.
  CPU BLAS, cuBLAS and cuDNN need not use the same reduction tree or instruction
  schedule. Different FP32 ordering is therefore possible without a wrong formula.
- `conv_dweight.cu` computes implicit-GEMM partials with `__fmaf_rn` over blocks
  of 128 reduction terms. A second kernel sums the partials using eight FP32
  lanes and a 4/2/1 tree. The original Im2col + CuPy tensordot path uses a different
  cuBLAS reduction. Neither is an FP64 reference.
- `fused_sgd.cu` explicitly uses `__fmul_rn` then `__fsub_rn`, matching HyTorch's
  separate multiply/subtract update. Whether a PyTorch update fuses those
  roundings is a measurement question; CPU results do not establish GPU behavior.
- ReLU changes derivative at zero; repeated nonlinear SGD updates can amplify
  initially tiny perturbations. This is a possible amplification mechanism, not
  a measured identification of a specific first sign flip in stage 4H.
- `core.py::Variable.backward` constructs derivative graphs only when requested.
  The new ReLU kernel is used only when `enable_backprop=False` inside backward;
  higher-order graph creation retains the existing differentiable expressions.

## New optional code

`hytorch/backends/cuda/relu_backward.cu` computes one FP32 predicate-multiply
per element. It supports arbitrary nonempty tensor dimensions after the existing
runtime's contiguous preparation, on the current CUDA stream. Layout copies are
included in the complete operator scope when necessary. `__fmul_rn` keeps the
original NaN/Inf and signed-zero multiplication behavior. No model, layer number,
convolution geometry or fixed tensor shape appears in the kernel.

`backend(relu_backward='fused')` is explicit; default `original` is unchanged.
The existing `selected` CPU/non-FP32 path retains original semantics with visible
dispatch counters. No hidden alternative native algorithm is selected.

`DataLoader(..., indices=...)` supplies the exact existing sample order, including
repeats and partial batches, to either original serial loading or the existing
ordered threaded loader. No new thread-pool architecture is introduced. Workers
only decode/preprocess CPU data; H2D and all model work stay in the main thread.
The existing result queue has bounded size, but its out-of-order pending dictionary
is not a strict total-memory bound; this experiment does not claim otherwise.

## Facts versus hypotheses

CPU-only free training and same-state replay are actual new measurements. Historical
GPU traces identify candidate costs but cannot establish new GPU speedups. The
current GPU replay and performance status is recorded separately; a successful
SM80 compilation does not establish numerical GPU correctness.
