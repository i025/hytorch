from dataclasses import dataclass

import numpy as np


QUANTIZED_FORMAT = "int8_symmetric_v1"
FORMAT_KEY = "__quantized_format__"


class QuantizationError(RuntimeError):
    pass


@dataclass(frozen=True)
class QuantizedArray:
    values: np.ndarray
    scale: np.ndarray
    original_dtype: str


def quantize_array_int8(array):
    """Symmetrically quantize a floating-point array to int8."""
    array = np.asarray(array)
    if not np.issubdtype(array.dtype, np.floating):
        raise QuantizationError(f"Only floating arrays can be quantized, got {array.dtype}.")

    fp32 = array.astype(np.float32, copy=False)
    max_abs = float(np.max(np.abs(fp32))) if fp32.size else 0.0
    scale = np.float32(max_abs / 127.0) if max_abs > 0 else np.float32(1.0)
    values = np.round(fp32 / scale).clip(-127, 127).astype(np.int8)
    return QuantizedArray(
        values=values,
        scale=np.asarray(scale, dtype=np.float32),
        original_dtype=str(array.dtype),
    )


def dequantize_array_int8(values, scale):
    """Dequantize an int8 array back to float32."""
    values = np.asarray(values)
    if values.dtype != np.int8:
        raise QuantizationError(f"Expected int8 values, got {values.dtype}.")
    return values.astype(np.float32) * np.asarray(scale, dtype=np.float32)


def quantized_linear_int8(x, weight_values, weight_scale, bias=None, return_info=False):
    """Run a Linear layer with int8 inputs/weights and int32 accumulation.

    The activation input is dynamically quantized per call. The stored weight
    must already be int8 with a matching symmetric scale.
    """
    x = np.asarray(x)
    weight_values = np.asarray(weight_values)
    if not np.issubdtype(x.dtype, np.floating):
        raise QuantizationError(f"Only floating activations can be quantized, got {x.dtype}.")
    if weight_values.dtype != np.int8:
        raise QuantizationError(f"Expected int8 weight values, got {weight_values.dtype}.")
    if x.ndim != 2 or weight_values.ndim != 2:
        raise QuantizationError(
            f"quantized_linear_int8 expects 2D input and weight matrices, got {x.shape} and {weight_values.shape}."
        )
    if x.shape[1] != weight_values.shape[0]:
        raise QuantizationError(f"Shape mismatch: x.shape={x.shape}, weight.shape={weight_values.shape}.")

    qx = quantize_array_int8(x)
    accumulator = qx.values.astype(np.int32) @ weight_values.astype(np.int32)
    y = accumulator.astype(np.float32) * qx.scale * np.asarray(weight_scale, dtype=np.float32)
    if bias is not None:
        y = y + np.asarray(bias, dtype=np.float32)

    if not return_info:
        return y.astype(np.float32, copy=False)
    info = {
        "input_dtype": str(qx.values.dtype),
        "weight_dtype": str(weight_values.dtype),
        "accumulator_dtype": str(accumulator.dtype),
        "accumulator_shape": list(accumulator.shape),
        "input_scale": qx.scale,
        "weight_scale": np.asarray(weight_scale, dtype=np.float32),
    }
    return y.astype(np.float32, copy=False), info


def quantize_npz_weights(input_path, output_path):
    """Quantize all floating-point arrays in an npz weight file to int8."""
    source = np.load(input_path)
    arrays = {FORMAT_KEY: np.asarray(QUANTIZED_FORMAT)}
    original_bytes = 0
    quantized_bytes = 0
    quantized_tensors = 0

    for key in source.files:
        array = source[key]
        original_bytes += array.nbytes
        if np.issubdtype(array.dtype, np.floating):
            quantized = quantize_array_int8(array)
            arrays[key] = quantized.values
            arrays[f"{key}.scale"] = quantized.scale
            arrays[f"{key}.dtype"] = np.asarray(quantized.original_dtype)
            quantized_bytes += quantized.values.nbytes + quantized.scale.nbytes
            quantized_tensors += 1
        else:
            arrays[key] = array
            quantized_bytes += array.nbytes

    np.savez_compressed(output_path, **arrays)
    ratio = 0.0
    if original_bytes:
        ratio = quantized_bytes / original_bytes
    report = {
        "format": QUANTIZED_FORMAT,
        "tensors": quantized_tensors,
        "original_bytes": original_bytes,
        "quantized_bytes": quantized_bytes,
        "compression_ratio": round(ratio, 6),
    }
    report.update({
        key: value
        for key, value in quantization_report(input_path, output_path).items()
        if key in {"max_abs_error", "mean_abs_error"}
    })
    return report


def load_quantized_npz_weights(path):
    """Load a quantized npz file and return dequantized float32 arrays."""
    npz = np.load(path)
    if FORMAT_KEY not in npz.files or str(npz[FORMAT_KEY]) != QUANTIZED_FORMAT:
        raise QuantizationError("Unsupported or missing quantized weight format.")

    arrays = {}
    for key in npz.files:
        if key == FORMAT_KEY or key.endswith(".scale") or key.endswith(".dtype"):
            continue
        scale_key = f"{key}.scale"
        if scale_key in npz.files:
            arrays[key] = dequantize_array_int8(npz[key], npz[scale_key])
        else:
            arrays[key] = npz[key]
    return arrays


def quantization_report(original_path, quantized_path):
    """Compare FP32 weights with an int8 npz and return error statistics."""
    original = np.load(original_path)
    quantized = np.load(quantized_path)
    if FORMAT_KEY not in quantized.files or str(quantized[FORMAT_KEY]) != QUANTIZED_FORMAT:
        raise QuantizationError("Unsupported or missing quantized weight format.")

    restored = load_quantized_npz_weights(quantized_path)
    tensor_reports = {}
    all_abs_errors = []
    original_bytes = 0
    quantized_bytes = 0

    for key in original.files:
        source = np.asarray(original[key])
        original_bytes += source.nbytes
        if key not in restored:
            raise QuantizationError(f"Missing quantized tensor: {key}")
        recovered = np.asarray(restored[key])
        if source.shape != recovered.shape:
            raise QuantizationError(f"Shape mismatch for {key}: {source.shape} vs {recovered.shape}")

        if f"{key}.scale" in quantized.files:
            quantized_bytes += quantized[key].nbytes + quantized[f"{key}.scale"].nbytes
            dtype = str(quantized[key].dtype)
        else:
            quantized_bytes += quantized[key].nbytes
            dtype = str(quantized[key].dtype)

        if np.issubdtype(source.dtype, np.floating):
            diff = np.abs(source.astype(np.float32) - recovered.astype(np.float32))
            all_abs_errors.append(diff.reshape(-1))
            tensor_reports[key] = {
                "dtype": dtype,
                "shape": list(source.shape),
                "max_abs_error": float(diff.max()) if diff.size else 0.0,
                "mean_abs_error": float(diff.mean()) if diff.size else 0.0,
                "scale": float(quantized[f"{key}.scale"]) if f"{key}.scale" in quantized.files else None,
            }
        else:
            tensor_reports[key] = {
                "dtype": dtype,
                "shape": list(source.shape),
                "max_abs_error": 0.0,
                "mean_abs_error": 0.0,
                "scale": None,
            }

    if all_abs_errors:
        errors = np.concatenate(all_abs_errors)
        max_abs_error = float(errors.max()) if errors.size else 0.0
        mean_abs_error = float(errors.mean()) if errors.size else 0.0
    else:
        max_abs_error = 0.0
        mean_abs_error = 0.0

    ratio = quantized_bytes / original_bytes if original_bytes else 0.0
    return {
        "format": QUANTIZED_FORMAT,
        "num_tensors": len(original.files),
        "original_bytes": int(original_bytes),
        "quantized_bytes": int(quantized_bytes),
        "compression_ratio": round(float(ratio), 6),
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "tensors": tensor_reports,
    }
