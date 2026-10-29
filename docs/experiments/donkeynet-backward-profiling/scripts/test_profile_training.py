"""CPU-only equivalence and reversible instrumentation checks; no CUDA work."""
from contextlib import contextmanager, nullcontext
import json
import sys
from types import SimpleNamespace
import numpy as np
import pytest

from profile_training import OperatorRanges, load_benchmark, array_shapes
import profile_training


class Sink:
    def __init__(self):
        self.stack = []
        self.labels = []
        self.maximum_depth = 0
        self.paths = []

    def push(self, label):
        self.stack.append(label)
        self.labels.append(label)
        self.paths.append(tuple(self.stack))
        self.maximum_depth = max(self.maximum_depth, len(self.stack))

    def pop(self):
        self.stack.pop()


def training_result(kind, instrumented):
    benchmark = load_benchmark()
    from donkeycar_hy.model import DonkeyNet
    import hytorch.functions as F
    from hytorch.optimizers import SGD
    shape = (3, 16, 16) if kind == "tiny" else (3, 66, 200)
    model = benchmark.TinyNet() if kind == "tiny" else DonkeyNet(dropout_rate=0.)
    benchmark.initialize(model, shape, 42)
    dataset = benchmark.SyntheticDataset(2, shape, 42)
    optimizer = SGD(.001).setup(model)
    sink = Sink()
    recorder = OperatorRanges(sink.push, sink.pop)
    with recorder if instrumented else nullcontext():
        loss = F.mean_square_error(model(dataset.x), dataset.t)
        model.cleargrads()
        loss.backward()
        grads = {k: v.grad.data.copy() for k, v in benchmark.parameters(model).items()}
        optimizer.update()
    return (loss.data.copy(), grads, benchmark.snapshot(model)), sink, recorder


@pytest.mark.parametrize("kind", ["tiny", "donkey"])
def test_cpu_training_bitwise_unchanged(kind):
    plain, _, _ = training_result(kind, False)
    observed, sink, recorder = training_result(kind, True)
    np.testing.assert_array_equal(observed[0], plain[0])
    for before, after in zip(plain[1:], observed[1:]):
        assert before.keys() == after.keys()
        for key in before:
            np.testing.assert_array_equal(after[key], before[key])
    assert not sink.stack
    assert not recorder.stack
    assert sink.maximum_depth >= 3
    for prefix in ("OP/bwd:Conv2d|", "OP/fwd:Deconv2d|", "OP/fwd:Conv2DGradW|",
                   "OP/bwd:MatMul|", "OP/fwd:MatMul|", "OP/internal:im2col_array|",
                   "OP/internal:col2im_array|"):
        assert any(label.startswith(prefix) for label in sink.labels)
    assert recorder.summary()["unclosed_ranges"] == 0
    assert not recorder.nodes
    assert not recorder.active_layers
    if kind == "donkey":
        for layer in range(1, 6):
            for component in ("dInput", "dWeight", "dBias"):
                prefix = "OP/part:Conv2d.{}|layer=conv{}".format(component, layer)
                labels = [label for label in sink.labels if label.startswith(prefix)]
                assert len(labels) == 1
                if component == "dInput":
                    assert labels[0].endswith("leaf_input=" + str(layer == 1).lower())
            for operation in ("im2col_array", "col2im_array"):
                part = "dWeight" if operation == "im2col_array" else "dInput"
                assert any(path[-1].startswith("OP/internal:" + operation + "|")
                    and any(label.startswith("OP/part:Conv2d.{}|layer=conv{}".format(part, layer))
                            for label in path) for path in sink.paths)
        for layer in range(1, 4):
            assert any(label.startswith("OP/bwd:MatMul|") and label.endswith("|layer=fc" + str(layer))
                       for label in sink.labels)
        assert len([label for label in sink.labels if label.startswith(
            "OP/part:MeanSquareError.dTarget|component=negate_only")]) == 1
        for name in plain[1]:
            assert "OP/update:SGD|parameter=" + name in sink.labels


def test_exception_pops_and_restores_original_methods():
    import hytorch.functions as F
    import hytorch.functions_conv as FC
    from hytorch.profiling import StepProfiler
    from hytorch.layers import Layer
    from hytorch.optimizers import SGD
    originals = (F.MatMul.forward, F.MatMul.backward, FC.im2col_array,
                 StepProfiler.phase, Layer.__call__, SGD.update_one)
    sink = Sink()
    recorder = OperatorRanges(sink.push, sink.pop)
    with pytest.raises(ValueError):
        with recorder:
            with recorder.range("test-outer"):
                F.matmul(np.ones((2, 3), np.float32), np.ones((4, 2), np.float32))
    assert not sink.stack
    assert not recorder.stack
    assert not recorder.nodes
    assert not recorder.active_layers
    assert (F.MatMul.forward, F.MatMul.backward, FC.im2col_array,
            StepProfiler.phase, Layer.__call__, SGD.update_one) == originals


def test_phase_attribution_and_memory_aggregation(tmp_path):
    from hytorch.profiling import StepProfiler
    sink = Sink()
    recorder = OperatorRanges(sink.push, sink.pop)
    profile = StepProfiler(tmp_path / "steps.jsonl", mode="phases")
    with recorder:
        profile.begin()
        with profile.phase("backward"):
            with recorder.range("OP/internal:example|in=4x4"):
                recorder.memory_record("malloc", 512, 64)
                recorder.memory_record("malloc", 512, 64)
        profile.end(2, 1, 1)
    summary = recorder.summary()
    assert summary["memory"] == [dict(phase="backward", label="OP/internal:example|in=4x4",
                                      kind="malloc", calls=2, bytes=1024, requested_bytes=128)]
    assert summary["calls"][0]["phase"] == "backward"
    assert recorder.phase == "outside"
    assert not recorder.stack
    assert json.loads((tmp_path / "steps.jsonl").read_text())["backward_s"] >= 0


def test_shapes_do_not_coerce_arrays():
    class MetadataOnly:
        shape = (4, 3, 66, 200)

        def __array__(self):
            raise AssertionError("Must not convert tensors while labelling")
    assert array_shapes([MetadataOnly(), None, np.float32(1)]) == "4x3x66x200;scalar"


@pytest.mark.parametrize("arguments", [
    ["--device", "cpu"],
    ["--no-fingerprints", "--operator-nvtx", "--device", "cpu", "--nvtx"],
    ["--no-fingerprints", "--memory-stats"],
])
def test_invalid_instrumentation_cli_is_rejected_before_training(monkeypatch, arguments):
    monkeypatch.setattr(sys, "argv", ["profile_training.py", "--output", "unused", *arguments])
    with pytest.raises(SystemExit) as error:
        profile_training.main()
    assert error.value.code == 2


def test_wrapper_forwards_benchmark_arguments_and_restores_argv(monkeypatch):
    arguments = ["--no-fingerprints", "--device", "cpu", "--seed", "42",
                 "--model", "donkey", "--output", "unused"]
    original_argv = ["profile_training.py", *arguments]
    received = []
    monkeypatch.setattr(sys, "argv", original_argv)
    monkeypatch.setattr(profile_training, "load_benchmark", lambda: SimpleNamespace(
        main=lambda: received.extend(sys.argv[1:])))
    profile_training.main()
    assert received == arguments
    assert sys.argv is original_argv


def test_torch_nvtx_context_contract_without_gpu(monkeypatch):
    events = []

    @contextmanager
    def emit_nvtx(record_shapes):
        events.append(("enter", record_shapes))
        try:
            yield
        finally:
            events.append(("exit", record_shapes))

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(autograd=SimpleNamespace(
        profiler=SimpleNamespace(emit_nvtx=emit_nvtx))))
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace())
    monkeypatch.setattr(profile_training, "load_benchmark", lambda: SimpleNamespace(
        main=lambda: events.append(("benchmark", "--operator-nvtx" not in sys.argv))))
    monkeypatch.setattr(sys, "argv", ["profile_training.py", "--framework", "torch",
        "--device", "cuda", "--nvtx", "--operator-nvtx", "--no-fingerprints", "--output", "unused"])
    profile_training.main()
    assert events == [("enter", True), ("benchmark", True), ("exit", True)]
