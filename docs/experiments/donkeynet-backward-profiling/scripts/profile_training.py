"""Opt-in NVTX labels around unchanged operations; no per-operator GPU sync.

Arguments not owned by this wrapper are passed to bench_cuda_training.main().
CPU tests use a fake range sink; actual profiling requires CUDA and --nvtx.
"""
import argparse
from collections import Counter
from contextlib import contextmanager, ExitStack
import functools
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))


def load_benchmark():
    spec = importlib.util.spec_from_file_location(
        "stage4a_training_benchmark", ROOT / "benchmarks/bench_cuda_training.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def array_shapes(values):
    """Read metadata only: no conversion, device operation, or scalar read."""
    return ";".join("x".join(map(str, value.shape)) or "scalar"
                    for value in values if getattr(value, "shape", None) is not None)


def function_classes(base):
    seen = set()
    todo = list(base.__subclasses__())
    while todo:
        cls = todo.pop()
        if cls not in seen:
            seen.add(cls)
            yield cls
            todo.extend(cls.__subclasses__())


class OperatorRanges:
    """Reversible Python wrappers; ranges measure host submission intervals."""
    FORWARDS = {"Conv2d", "Deconv2d", "Conv2DGradW", "MatMul", "Linear",
                "Transpose", "Reshape", "Im2col", "Col2im", "ReLU", "Tanh"}

    def __init__(self, push, pop, cupy_module=None):
        self.push, self.pop, self.cp = push, pop, cupy_module
        self.patches = []
        self.stack = []
        self.phase = "outside"
        self.calls = Counter()
        self.memory = {}

    @contextmanager
    def range(self, label):
        self.push(label)
        self.stack.append(label)
        self.calls[(self.phase, label)] += 1
        try:
            yield
        finally:
            self.stack.pop()
            self.pop()

    def patch(self, owner, name, wrapped):
        self.patches.append((owner, name, getattr(owner, name)))
        setattr(owner, name, wrapped)

    def wrap_method(self, cls, method, direction):
        original = cls.__dict__[method]

        @functools.wraps(original)
        def call(instance, *args, **kwargs):
            values = instance.inputs if direction == "bwd" else args
            label = "OP/{}:{}|in={}".format(direction, cls.__name__, array_shapes(values))
            with self.range(label):
                return original(instance, *args, **kwargs)
        self.patch(cls, method, call)

    def wrap_array_function(self, module, name, prefix):
        original = getattr(module, name)

        @functools.wraps(original)
        def call(*args, **kwargs):
            label = "OP/internal:{}{}|in={}".format(prefix, name, array_shapes(args))
            with self.range(label):
                return original(*args, **kwargs)
        self.patch(module, name, call)

    def __enter__(self):
        from hytorch.core import Function
        import hytorch.functions_conv as FC
        from hytorch.profiling import StepProfiler
        for cls in function_classes(Function):
            if "backward" in cls.__dict__:
                self.wrap_method(cls, "backward", "bwd")
            if cls.__name__ in self.FORWARDS and "forward" in cls.__dict__:
                self.wrap_method(cls, "forward", "fwd")
        for name in ("im2col_array", "col2im_array"):
            self.wrap_array_function(FC, name, "")
        if self.cp is not None:
            for name in ("dot", "tensordot", "ascontiguousarray", "rollaxis"):
                self.wrap_array_function(self.cp, name, "cupy.")
        original_phase = StepProfiler.phase

        @contextmanager
        def phase(instance, name):
            previous = self.phase
            self.phase = name
            try:
                with original_phase(instance, name):
                    yield
            finally:
                self.phase = previous
        self.patch(StepProfiler, "phase", phase)
        return self

    def __exit__(self, *exc):
        for owner, name, original in reversed(self.patches):
            setattr(owner, name, original)
        self.patches.clear()

    def memory_record(self, kind, size, requested=0):
        key = (self.phase, self.stack[-1] if self.stack else "unlabelled", kind)
        row = self.memory.setdefault(key, {"calls": 0, "bytes": 0, "requested_bytes": 0})
        row["calls"] += 1
        row["bytes"] += size
        row["requested_bytes"] += requested

    def summary(self):
        return dict(
            scope="whole benchmark process; phase=outside includes initialization/warmup/final snapshots",
            timing="NVTX host ranges; no added synchronization or CUDA Events; GPU attribution requires CUDA correlation IDs",
            memory_semantics="CuPy pool malloc requests, pool backing alloc, and pool free; counts are not live or peak bytes; PyTorch allocator excluded",
            labels="OP/{bwd|fwd}:Class|in=shape;shape or OP/internal:function|in=shape;shape",
            calls=[dict(phase=p, label=l, calls=n) for (p, l), n in sorted(self.calls.items())],
            memory=[dict(phase=p, label=l, kind=k, **v) for (p, l, k), v in sorted(self.memory.items())],
            unclosed_ranges=len(self.stack))


def memory_hook(cp, recorder):
    """CuPy 12.3 MemoryHook callbacks; collect aggregates in RAM only."""
    class Counts(cp.cuda.memory_hook.MemoryHook):
        name = "Stage4AMemoryCounts"

        def malloc_postprocess(self, **kwargs):
            recorder.memory_record("malloc", kwargs["mem_size"], kwargs["size"])

        def alloc_postprocess(self, **kwargs):
            recorder.memory_record("alloc", kwargs["mem_size"])

        def free_postprocess(self, **kwargs):
            recorder.memory_record("free", kwargs["mem_size"])
    return Counts()


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--operator-nvtx", action="store_true")
    parser.add_argument("--memory-stats", action="store_true")
    options, forwarded = parser.parse_known_args()
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--framework", default="hytorch")
    config_parser.add_argument("--device", default="cuda")
    config_parser.add_argument("--output", required=True)
    config_parser.add_argument("--nvtx", action="store_true")
    config_parser.add_argument("--no-fingerprints", action="store_true")
    config, _ = config_parser.parse_known_args(forwarded)
    if not config.no_fingerprints:
        parser.error("Stage 4A requires --no-fingerprints")
    if options.operator_nvtx and (config.device != "cuda" or not config.nvtx):
        parser.error("--operator-nvtx requires --device cuda --nvtx")
    if options.memory_stats and (not options.operator_nvtx or config.framework != "hytorch"):
        parser.error("--memory-stats requires HyTorch --operator-nvtx")
    benchmark = load_benchmark()
    recorder = None
    original_argv = sys.argv
    sys.argv = [str(ROOT / "benchmarks/bench_cuda_training.py"), *forwarded]
    try:
        with ExitStack() as stack:
            if options.operator_nvtx:
                import cupy as cp
                if config.framework == "hytorch":
                    recorder = stack.enter_context(OperatorRanges(
                        cp.cuda.nvtx.RangePush, cp.cuda.nvtx.RangePop, cp))
                    if options.memory_stats:
                        stack.enter_context(memory_hook(cp, recorder))
                else:
                    import torch
                    stack.enter_context(torch.autograd.profiler.emit_nvtx(record_shapes=True))
            benchmark.main()
    finally:
        sys.argv = original_argv
    if recorder is not None:
        (Path(config.output) / "instrumentation.json").write_text(
            json.dumps(recorder.summary(), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
