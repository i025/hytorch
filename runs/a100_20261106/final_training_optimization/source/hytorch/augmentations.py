import numpy as np


def one_hot(labels, num_classes=None, dtype=np.float32):
    """Convert integer labels to one-hot rows; pass soft labels through."""
    labels = np.asarray(labels)
    if labels.ndim == 2:
        return labels.astype(dtype, copy=False)
    flat = labels.astype(np.int64).reshape(-1)
    if num_classes is None:
        num_classes = int(flat.max()) + 1 if flat.size else 0
    encoded = np.zeros((flat.size, int(num_classes)), dtype=dtype)
    if flat.size:
        encoded[np.arange(flat.size), flat] = 1.0
    return encoded


def _rng(rng):
    return rng if rng is not None else np.random


def _beta(rng, a, b):
    if hasattr(rng, "beta"):
        return rng.beta(a, b)
    return np.random.beta(a, b)


def _permutation(rng, n):
    if hasattr(rng, "permutation"):
        return rng.permutation(n)
    return np.random.permutation(n)


def _randint(rng, low, high=None):
    if hasattr(rng, "integers"):
        return int(rng.integers(low, high))
    return int(np.random.randint(low, high))


class MixUp:
    """Batch-level MixUp augmentation.

    The transform returns ``(mixed_x, mixed_targets, info)`` so it can be passed
    directly to ``Trainer(batch_augment=...)``. Integer labels are converted to
    soft labels. Use ``softmax_cross_entropy_soft_label`` as the loss function.
    """

    def __init__(self, alpha=1.0, num_classes=None, rng=None):
        self.alpha = float(alpha)
        self.num_classes = num_classes
        self.rng = _rng(rng)
        self.last_info = {}

    def __call__(self, x, t):
        x = np.asarray(x)
        targets = one_hot(t, self.num_classes)
        n = x.shape[0]
        if n == 0:
            self.last_info = {"method": "mixup", "lambda": 1.0, "alpha": self.alpha, "permutation": []}
            return x, targets, dict(self.last_info)

        lam = float(_beta(self.rng, self.alpha, self.alpha)) if self.alpha > 0 else 1.0
        perm = _permutation(self.rng, n)
        mixed_x = lam * x + (1.0 - lam) * x[perm]
        mixed_t = lam * targets + (1.0 - lam) * targets[perm]
        self.last_info = {
            "method": "mixup",
            "lambda": lam,
            "alpha": self.alpha,
            "permutation": perm.tolist(),
        }
        return mixed_x.astype(x.dtype, copy=False), mixed_t.astype(np.float32, copy=False), dict(self.last_info)


class CutMix:
    """Batch-level CutMix augmentation for image batches.

    Default layout is NCHW, matching the rest of HyTorch image code. Set
    ``channel_axis=-1`` for NHWC batches.
    """

    def __init__(self, alpha=1.0, num_classes=None, channel_axis=1, rng=None):
        self.alpha = float(alpha)
        self.num_classes = num_classes
        self.channel_axis = channel_axis
        self.rng = _rng(rng)
        self.last_info = {}

    def __call__(self, x, t):
        x = np.asarray(x)
        if x.ndim != 4:
            raise ValueError("CutMix expects a 4D image batch, e.g. NCHW or NHWC.")
        targets = one_hot(t, self.num_classes)
        n = x.shape[0]
        if n == 0:
            self.last_info = {"method": "cutmix", "lambda": 1.0, "alpha": self.alpha, "permutation": []}
            return x, targets, dict(self.last_info)

        lam = float(_beta(self.rng, self.alpha, self.alpha)) if self.alpha > 0 else 1.0
        perm = _permutation(self.rng, n)
        mixed = x.copy()
        x1, y1, x2, y2 = self._rand_bbox(x.shape, lam)

        if self.channel_axis in (1, -3):
            mixed[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
            height, width = x.shape[2], x.shape[3]
        elif self.channel_axis in (-1, 3):
            mixed[:, y1:y2, x1:x2, :] = x[perm, y1:y2, x1:x2, :]
            height, width = x.shape[1], x.shape[2]
        else:
            raise ValueError("CutMix supports channel_axis=1 for NCHW or channel_axis=-1 for NHWC.")

        patch_area = max(0, x2 - x1) * max(0, y2 - y1)
        lam = 1.0 - patch_area / float(width * height)
        mixed_t = lam * targets + (1.0 - lam) * targets[perm]
        self.last_info = {
            "method": "cutmix",
            "lambda": float(lam),
            "alpha": self.alpha,
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "permutation": perm.tolist(),
        }
        return mixed.astype(x.dtype, copy=False), mixed_t.astype(np.float32, copy=False), dict(self.last_info)

    def _rand_bbox(self, shape, lam):
        if self.channel_axis in (1, -3):
            height, width = shape[2], shape[3]
        elif self.channel_axis in (-1, 3):
            height, width = shape[1], shape[2]
        else:
            raise ValueError("CutMix supports channel_axis=1 for NCHW or channel_axis=-1 for NHWC.")
        cut_ratio = np.sqrt(max(0.0, 1.0 - lam))
        cut_w = max(1, int(width * cut_ratio))
        cut_h = max(1, int(height * cut_ratio))
        cx = _randint(self.rng, 0, width)
        cy = _randint(self.rng, 0, height)
        x1 = np.clip(cx - cut_w // 2, 0, width)
        x2 = np.clip(cx + cut_w // 2, 0, width)
        y1 = np.clip(cy - cut_h // 2, 0, height)
        y2 = np.clip(cy + cut_h // 2, 0, height)
        if x1 == x2:
            x2 = min(width, x1 + 1)
        if y1 == y2:
            y2 = min(height, y1 + 1)
        return int(x1), int(y1), int(x2), int(y2)

