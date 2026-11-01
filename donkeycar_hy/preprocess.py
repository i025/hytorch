"""Shared RGB preprocessing for recorded images and future telemetry inference.

Inputs are RGB Pillow images or HWC uint8 NumPy arrays in [0, 255]. Already
normalized arrays are deliberately rejected to prevent double normalization.
There are no HyTorch, CUDA, OpenCV or PyTorch imports in this module.
"""

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image


CROP_TOP = 40
TARGET_SIZE = (200, 66)  # Pillow uses (width, height).
_BILINEAR = getattr(Image, "Resampling", Image).BILINEAR


def load_rgb_image(path: Union[str, Path]) -> Image.Image:
    """Decode an image as RGB, closing its file before returning."""
    with Image.open(Path(path).expanduser()) as image:
        return image.convert("RGB")


def preprocess_rgb(image: Union[Image.Image, np.ndarray]) -> np.ndarray:
    """Crop 40 top rows, bilinear resize, normalize, then return CHW FP32.

    No bottom crop or color-space conversion is applied. Both file decoding
    and telemetry decoding should explicitly produce RGB before calling this
    function. The input is never modified; the result is C-contiguous with
    shape (3, 66, 200) and values in [-1, 1].
    """
    if isinstance(image, np.ndarray):
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected an HWC RGB array with three channels")
        if image.dtype != np.uint8:
            raise TypeError("Expected uint8 RGB pixels in [0, 255]")
        image = Image.fromarray(image)
    elif not isinstance(image, Image.Image):
        raise TypeError("Expected an RGB Pillow image or HWC uint8 NumPy array")

    if image.mode != "RGB":
        raise ValueError("Expected RGB mode; convert the decoded image to RGB first")
    width, height = image.size
    if width <= 0 or height <= CROP_TOP:
        raise ValueError("Image must have positive width and more than 40 rows")

    resized = image.crop((0, CROP_TOP, width, height)).resize(TARGET_SIZE, _BILINEAR)
    pixels = np.asarray(resized, dtype=np.float32)
    normalized = pixels / np.float32(127.5) - np.float32(1.0)
    return np.ascontiguousarray(normalized.transpose(2, 0, 1), dtype=np.float32)
