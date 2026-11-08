import copy
import numpy as np

from hytorch import cuda
from hytorch.core import Variable, as_variable, using_config
import hytorch.functions as F
import hytorch.functions_conv as C

try:
    import onnx
    from onnx import TensorProto, helper, numpy_helper
except ImportError as e:
    onnx = None
    TensorProto = None
    helper = None
    numpy_helper = None
    _ONNX_IMPORT_ERROR = e
else:
    _ONNX_IMPORT_ERROR = None


class UnsupportedONNXOpError(NotImplementedError):
    pass


class ONNXExportError(RuntimeError):
    pass


def load(path):
    """Load an ONNX model from disk."""
    _require_onnx()
    return onnx.load(str(path))


def save(model, path):
    """Save an ONNX model to disk after validating it."""
    _require_onnx()
    check(model)
    onnx.save(_to_model(model), str(path))


def check(model_or_path):
    """Validate an ONNX model or model path."""
    _require_onnx()
    try:
        onnx.checker.check_model(_to_model(model_or_path))
    except Exception as e:
        raise ONNXExportError(f"Invalid ONNX model: {e}") from e


def get_node_info(model_or_path):
    """Return printable metadata for every node in an ONNX graph."""
    model = _to_model(model_or_path)
    info = []
    for index, node in enumerate(model.graph.node):
        info.append({
            "index": index,
            "name": node.name,
            "op_type": node.op_type,
            "domain": node.domain,
            "inputs": list(node.input),
            "outputs": list(node.output),
        })
    return info


def print_node_info(model_or_path):
    """Print ONNX node metadata in a classroom-friendly format."""
    for node in get_node_info(model_or_path):
        print(
            "index={index} name={name} op_type={op_type} domain={domain} "
            "inputs={inputs} outputs={outputs}".format(**node)
        )


def remove_node(model_or_path, node_name=None, op_type=None, index=None):
    """Remove one ONNX node selected by name, op type, or index."""
    selectors = [node_name is not None, op_type is not None, index is not None]
    if sum(selectors) != 1:
        raise ValueError("remove_node requires exactly one selector.")

    model = copy.deepcopy(_to_model(model_or_path))
    node_index = _find_node_index(model, node_name=node_name, op_type=op_type, index=index)
    del model.graph.node[node_index]
    check(model)
    return model


def add_identity_node(model_or_path, input_name, output_name,
                      node_name="AddedIdentity"):
    """Append a simple Identity node for ONNX graph-editing demonstrations."""
    model = copy.deepcopy(_to_model(model_or_path))
    node = helper.make_node(
        "Identity",
        [input_name],
        [output_name],
        name=node_name,
    )
    model.graph.node.append(node)
    check(model)
    return model


def export_torch_model(model, inputs, path, input_names=("input",),
                       output_names=("output",), opset_version=13,
                       dynamic_axes=None):
    """Export a PyTorch model to ONNX through torch.onnx.export."""
    _require_onnx()
    try:
        import torch
    except ImportError as e:
        raise ONNXExportError(
            "PyTorch is required for export_torch_model(). Install torch first."
        ) from e

    model.eval()
    with torch.no_grad():
        torch.onnx.export(
            model,
            inputs,
            str(path),
            export_params=True,
            input_names=list(input_names),
            output_names=list(output_names),
            opset_version=opset_version,
            dynamic_axes=dynamic_axes,
            dynamo=False,
        )
    check(path)


def export(model, inputs, path, input_names=("input",), output_names=("output",),
           opset_version=13, dynamic_batch=True):
    """Export a single-input HyTorch model to ONNX for inference."""
    if _ONNX_IMPORT_ERROR is not None:
        raise ONNXExportError(
            "The `onnx` package is required for ONNX export. "
            "Install dependencies from requirements.txt."
        ) from _ONNX_IMPORT_ERROR

    if isinstance(inputs, (tuple, list)):
        raise ONNXExportError("ONNX export v1 supports exactly one input.")
    if len(input_names) != 1 or len(output_names) != 1:
        raise ONNXExportError("ONNX export v1 supports one input and one output name.")

    x = as_variable(inputs)
    if x.creator is not None:
        raise ONNXExportError("ONNX export input must be a leaf Variable or ndarray.")

    with using_config("train", False), using_config("enable_backprop", True):
        y = model(x)

    if isinstance(y, tuple):
        if len(y) != 1:
            raise ONNXExportError("ONNX export v1 supports exactly one output.")
        y = y[0]
    if not isinstance(y, Variable):
        raise ONNXExportError("Model output must be a HyTorch Variable.")

    exporter = _GraphExporter(input_names[0], output_names[0], x, y, dynamic_batch)
    graph = exporter.build()
    opset = helper.make_operatorsetid("", opset_version)
    model_proto = helper.make_model(
        graph,
        producer_name="hytorch",
        opset_imports=[opset],
    )
    onnx.checker.check_model(model_proto)
    onnx.save(model_proto, str(path))


def _require_onnx():
    if _ONNX_IMPORT_ERROR is not None:
        raise ONNXExportError(
            "The `onnx` package is required for ONNX export. "
            "Install dependencies from requirements.txt."
        ) from _ONNX_IMPORT_ERROR


def _to_model(model_or_path):
    _require_onnx()
    if isinstance(model_or_path, onnx.ModelProto):
        return model_or_path
    return load(model_or_path)


def _find_node_index(model, node_name=None, op_type=None, index=None):
    nodes = list(model.graph.node)
    if index is not None:
        if index < 0 or index >= len(nodes):
            raise ValueError(f"ONNX node index out of range: {index}")
        return index
    if node_name is not None:
        matches = [i for i, node in enumerate(nodes) if node.name == node_name]
    else:
        matches = [i for i, node in enumerate(nodes) if node.op_type == op_type]
    if not matches:
        selector = node_name if node_name is not None else op_type
        raise ValueError(f"ONNX node not found: {selector}")
    if len(matches) > 1:
        raise ValueError("Selector matched multiple ONNX nodes; use index instead.")
    return matches[0]


class _GraphExporter:
    def __init__(self, input_name, output_name, input_var, output_var, dynamic_batch):
        self.input_name = input_name
        self.output_name = output_name
        self.input_var = input_var
        self.output_var = output_var
        self.dynamic_batch = dynamic_batch
        self.names = {id(input_var): input_name, id(output_var): output_name}
        self.used_names = {input_name, output_name}
        self.initializer_ids = set()
        self.initializers = []
        self.nodes = []
        self.extra_value_infos = []

    def build(self):
        funcs = _collect_functions(self.output_var)
        for func in funcs:
            self._emit_node(func)

        graph = helper.make_graph(
            self.nodes,
            "hytorch_graph",
            [self._make_value_info(self.input_name, self.input_var)],
            [self._make_value_info(self.output_name, self.output_var)],
            self.initializers,
            value_info=self.extra_value_infos,
        )
        return graph

    def _emit_node(self, func):
        name = func.__class__.__name__

        if isinstance(func, (F.Reshape,)):
            self._emit_reshape(func)
        elif isinstance(func, F.Transpose):
            self._emit_transpose(func)
        elif isinstance(func, F.MatMul):
            self._emit_simple(func, "MatMul")
        elif isinstance(func, F.Linear):
            self._emit_linear(func)
        elif isinstance(func, F.ReLU):
            self._emit_simple(func, "Relu")
        elif isinstance(func, F.Sigmoid):
            self._emit_simple(func, "Sigmoid")
        elif isinstance(func, F.Tanh):
            self._emit_simple(func, "Tanh")
        elif isinstance(func, F.Exp):
            self._emit_simple(func, "Exp")
        elif isinstance(func, C.Conv2d):
            self._emit_conv(func)
        elif isinstance(func, C.Pooling):
            self._emit_pool(func, "MaxPool")
        elif isinstance(func, C.AveragePooling):
            self._emit_pool(func, "AveragePool")
        elif name in {"Add", "Sub", "Mul", "Div"}:
            self._emit_simple(func, name)
        elif name == "Neg":
            self._emit_simple(func, "Neg")
        else:
            raise UnsupportedONNXOpError(
                f"{name} is not supported for ONNX export v1"
            )

    def _emit_simple(self, func, op_type):
        self.nodes.append(helper.make_node(
            op_type,
            self._input_names(func.inputs),
            self._output_names(func.outputs),
        ))

    def _emit_reshape(self, func):
        input_var = func.inputs[0]
        target = list(func.shape)
        if (
            self.dynamic_batch
            and target
            and input_var.data is not None
            and target[0] == input_var.shape[0]
        ):
            target[0] = 0
        shape_name = self._new_aux_name("reshape_shape")
        self.initializers.append(numpy_helper.from_array(
            np.asarray(target, dtype=np.int64),
            shape_name,
        ))
        self.nodes.append(helper.make_node(
            "Reshape",
            [self._name(input_var), shape_name],
            self._output_names(func.outputs),
        ))

    def _emit_transpose(self, func):
        attrs = {}
        if func.axes is not None:
            attrs["perm"] = list(func.axes)
        self.nodes.append(helper.make_node(
            "Transpose",
            self._input_names(func.inputs),
            self._output_names(func.outputs),
            **attrs,
        ))

    def _emit_linear(self, func):
        x, W, b = func.inputs
        if b.data is None:
            self.nodes.append(helper.make_node(
                "MatMul",
                [self._name(x), self._name(W)],
                self._output_names(func.outputs),
            ))
            return
        self.nodes.append(helper.make_node(
            "Gemm",
            [self._name(x), self._name(W), self._name(b)],
            self._output_names(func.outputs),
        ))

    def _emit_conv(self, func):
        inputs = [self._name(func.inputs[0]), self._name(func.inputs[1])]
        if func.inputs[2].data is not None:
            inputs.append(self._name(func.inputs[2]))
        ph, pw = _pair(func.pad)
        sh, sw = _pair(func.stride)
        self.nodes.append(helper.make_node(
            "Conv",
            inputs,
            self._output_names(func.outputs),
            pads=[ph, pw, ph, pw],
            strides=[sh, sw],
        ))

    def _emit_pool(self, func, op_type):
        kh, kw = _pair(func.kernel_size)
        ph, pw = _pair(func.pad)
        sh, sw = _pair(func.stride)
        self.nodes.append(helper.make_node(
            op_type,
            self._input_names(func.inputs),
            self._output_names(func.outputs),
            kernel_shape=[kh, kw],
            pads=[ph, pw, ph, pw],
            strides=[sh, sw],
        ))

    def _name(self, var):
        if id(var) not in self.names:
            self.names[id(var)] = self._unique_name(var.name or "v")
        if self._should_be_initializer(var):
            self._add_initializer(var)
        return self.names[id(var)]

    def _input_names(self, variables):
        return [self._name(v) for v in variables if v.data is not None or v.creator is not None]

    def _output_names(self, weak_outputs):
        return [self._name(ref()) for ref in weak_outputs]

    def _should_be_initializer(self, var):
        return (
            var.data is not None
            and id(var) != id(self.input_var)
            and var.creator is None
        )

    def _add_initializer(self, var):
        var_id = id(var)
        if var_id in self.initializer_ids:
            return
        self.initializer_ids.add(var_id)
        array = _as_numpy(var.data)
        self.initializers.append(numpy_helper.from_array(array, self._name(var)))

    def _new_aux_name(self, prefix):
        return self._unique_name(prefix)

    def _unique_name(self, base):
        candidate = str(base)
        if candidate not in self.used_names:
            self.used_names.add(candidate)
            return candidate
        index = 1
        while True:
            candidate = f"{base}_{index}"
            if candidate not in self.used_names:
                self.used_names.add(candidate)
                return candidate
            index += 1

    def _make_value_info(self, name, var):
        return helper.make_tensor_value_info(
            name,
            _tensor_dtype(var.data),
            _onnx_shape(var.shape, self.dynamic_batch),
        )


def _collect_functions(output):
    funcs = []
    seen = set()

    def add_func(func):
        if func is None or func in seen:
            return
        seen.add(func)
        for x in func.inputs:
            add_func(x.creator)
        funcs.append(func)

    add_func(output.creator)
    return sorted(funcs, key=lambda f: f.generation)


def _as_numpy(array):
    return cuda.as_numpy(array).astype(array.dtype, copy=False)


def _tensor_dtype(array):
    dtype = np.dtype(array.dtype)
    if dtype == np.float32:
        return TensorProto.FLOAT
    if dtype == np.float64:
        return TensorProto.DOUBLE
    if dtype == np.int64:
        return TensorProto.INT64
    if dtype == np.int32:
        return TensorProto.INT32
    if dtype == np.bool_:
        return TensorProto.BOOL
    raise ONNXExportError(f"Unsupported ONNX tensor dtype: {dtype}")


def _onnx_shape(shape, dynamic_batch):
    dims = list(shape)
    if dynamic_batch and dims:
        dims[0] = "batch"
    return dims


def _pair(value):
    if isinstance(value, tuple):
        return value
    return (value, value)
