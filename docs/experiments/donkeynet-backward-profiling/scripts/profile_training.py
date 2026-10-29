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
import weakref

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
                "Transpose", "Reshape", "Im2col", "Col2im", "ReLU", "Tanh",
                "Sum", "Neg", "MeanSquareError"}

    def __init__(self, push, pop, cupy_module=None):
        self.push, self.pop, self.cp = push, pop, cupy_module
        self.patches = []
        self.stack = []
        self.phase = "outside"
        self.calls = Counter()
        self.memory = {}
        self.nodes = []
        self.active_layers = []
        self.models = weakref.WeakSet()
        self.layer_names = weakref.WeakKeyDictionary()
        self.parameter_names = weakref.WeakKeyDictionary()
        self.function_layers = weakref.WeakKeyDictionary()

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
            parent = self.nodes[-1] if self.nodes else None
            layer = self.function_layers.get(instance, "")
            if direction == "fwd":
                layer = (parent[3] if parent else
                         self.active_layers[-1] if self.active_layers else "")
                if layer:
                    self.function_layers[instance] = layer
            values = instance.inputs if direction == "bwd" else args
            label = "OP/{}:{}|in={}".format(direction, cls.__name__, array_shapes(values))
            if layer:
                label += "|layer=" + layer
            part = None
            if direction == "fwd" and parent and parent[1] == "bwd":
                if parent[0] == "Conv2d":
                    component = {"Deconv2d": "dInput", "Conv2DGradW": "dWeight",
                                 "Sum": "dBias"}.get(cls.__name__)
                    if component:
                        part = "OP/part:Conv2d.{}|layer={}".format(component, parent[3])
                        if component == "dInput":
                            part += "|leaf_input=" + str(parent[2].inputs[0].creator is None).lower()
                elif parent[0] == "MeanSquareError" and cls.__name__ == "Neg":
                    part = "OP/part:MeanSquareError.dTarget|component=negate_only"
            self.nodes.append((cls.__name__, direction, instance, layer))
            try:
                with ExitStack() as scopes:
                    if part:
                        scopes.enter_context(self.range(part))
                    scopes.enter_context(self.range(label))
                    return original(instance, *args, **kwargs)
            finally:
                self.nodes.pop()
        self.patch(cls, method, call)

    def wrap_layers_and_updates(self):
        """Identify actual named model parameters without reading tensor values."""
        from hytorch.layers import Layer
        from hytorch.optimizers import SGD
        original_call = Layer.__call__

        def register(layer, path=""):
            self.layer_names[layer] = path
            for name in sorted(layer._params):
                value = getattr(layer, name)
                child_path = path + "/" + name if path else name
                if isinstance(value, Layer):
                    register(value, child_path)
                else:
                    self.parameter_names[value] = child_path

        @functools.wraps(original_call)
        def call(layer, *args, **kwargs):
            if not self.active_layers and layer not in self.models:
                register(layer)
                self.models.add(layer)
            self.active_layers.append(self.layer_names.get(layer, ""))
            try:
                return original_call(layer, *args, **kwargs)
            finally:
                self.active_layers.pop()
        self.patch(Layer, "__call__", call)
        original_update = SGD.update_one

        @functools.wraps(original_update)
        def update(optimizer, parameter):
            name = self.parameter_names.get(parameter, "unnamed")
            with self.range("OP/update:SGD|parameter=" + name):
                return original_update(optimizer, parameter)
        self.patch(SGD, "update_one", update)

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
        self.wrap_layers_and_updates()
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
            labels="OP/{bwd|fwd}:Class|in=shape;shape[|layer=name]; OP/part:Conv2d.{dInput|dWeight|dBias}|layer=name; OP/internal:function|in=shape; OP/update:SGD|parameter=name",
            component_scope="Conv2d components enclose original Deconv2d/Conv2DGradW/Sum.forward calls; dTarget encloses only original Neg.forward, shared MSE arithmetic stays in the MSE parent. Host component ranges exclude surrounding Function.__call__ bookkeeping; GPU kernels are attributed using CUDA API correlations.",
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
