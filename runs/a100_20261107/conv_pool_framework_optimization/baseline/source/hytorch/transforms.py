import numpy as np

try:
    import Image
except ImportError:
    from PIL import Image
from hytorch.utils import pair


class Compose:
    """Compose several transforms.

    Args:
        transforms (list): list of transforms
    """

    def __init__(self, transforms=[]):
        self.transforms = transforms

    def __call__(self, img):
        if not self.transforms:
            return img
        for t in self.transforms:
            img = t(img)
        return img


# =============================================================================
# Transforms for PIL Image
# =============================================================================
class Convert:
    def __init__(self, mode="RGB"):
        self.mode = mode

    def __call__(self, img):
        if self.mode == "BGR":
            img = img.convert("RGB")
            r, g, b = img.split()
            img = Image.merge("RGB", (b, g, r))
            return img
        else:
            return img.convert(self.mode)


class Resize:
    """Resize the input PIL image to the given size.

    Args:
        size (int or (int, int)): Desired output size
        mode (int): Desired interpolation.
    """

    def __init__(self, size, mode=Image.BILINEAR):
        self.size = pair(size)
        self.mode = mode

    def __call__(self, img):
        return img.resize(self.size, self.mode)


class CenterCrop:
    """Resize the input PIL image to the given size.

    Args:
        size (int or (int, int)): Desired output size.
        mode (int): Desired interpolation.
    """

    def __init__(self, size):
        self.size = pair(size)

    def __call__(self, img):
        W, H = img.size
        OW, OH = self.size
        left = (W - OW) // 2
        right = W - ((W - OW) // 2 + (W - OW) % 2)
        up = (H - OH) // 2
        bottom = H - ((H - OH) // 2 + (H - OH) % 2)
        return img.crop((left, up, right, bottom))


class ToArray:
    """Convert PIL Image to NumPy array."""

    def __init__(self, dtype=np.float32):
        self.dtype = dtype

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            return img
        if isinstance(img, Image.Image):
            img = np.asarray(img)
            img = img.transpose(2, 0, 1)
            img = img.astype(self.dtype)
            return img
        else:
            raise TypeError


class ToPIL:
    """Convert NumPy array to PIL Image."""

    def __call__(self, array):
        data = array.transpose(1, 2, 0)
        return Image.fromarray(data)


class RandomHorizontalFlip:
    pass


# =============================================================================
# Transforms for NumPy ndarray
# =============================================================================
class Normalize:
    """Normalize a NumPy array with mean and standard deviation.

    Args:
        mean (float or sequence): mean for all values or sequence of means for
         each channel.
        std (float or sequence):
    """

    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std

    def __call__(self, array):
        mean, std = self.mean, self.std

        if not np.isscalar(mean):
            mshape = [1] * array.ndim
            mshape[0] = len(array) if len(self.mean) == 1 else len(self.mean)
            mean = np.array(self.mean, dtype=array.dtype).reshape(*mshape)
        if not np.isscalar(std):
            rshape = [1] * array.ndim
            rshape[0] = len(array) if len(self.std) == 1 else len(self.std)
            std = np.array(self.std, dtype=array.dtype).reshape(*rshape)
        return (array - mean) / std


class Flatten:
    """Flatten a NumPy array."""

    def __call__(self, array):
        return array.flatten()




class AsType:
    def __init__(self, dtype=np.float32):
        self.dtype = dtype

    def __call__(self, array):
        return array.astype(self.dtype)


ToFloat = AsType


class ToInt(AsType):
    def __init__(self, dtype=int):
        self.dtype = dtype





# === 继续追加到你的 transforms 源文件中（与上面类放一起即可） ===
import numpy as np
from PIL import Image, ImageOps, ImageEnhance

# ---------- 小工具 ----------
def _is_pil(x):
    return isinstance(x, Image.Image)

def _is_ndarray(x):
    return isinstance(x, np.ndarray)

def _to_pil_from_chw(x):
    # x: (C,H,W) -> PIL
    if x.ndim != 3:
        raise TypeError("Expect CHW ndarray to convert PIL")
    c,h,w = x.shape
    if c == 1:
        arr = x[0]
        mode = "L"
    elif c == 3:
        arr = x.transpose(1,2,0)
        mode = "RGB"
    else:
        raise ValueError("Only support 1 or 3 channels")
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode=mode)

def _to_chw_from_pil(img, dtype=np.float32):
    # PIL -> (C,H,W)
    arr = np.asarray(img)
    if arr.ndim == 2:        # gray
        arr = arr[None, ...] # (1,H,W)
    else:
        arr = arr.transpose(2,0,1)  # (H,W,C) -> (C,H,W)
    return arr.astype(dtype)

# ---------- 修复：ToArray 兼容灰度 ----------
class ToArray:
    """Convert PIL Image to NumPy array (CHW)."""
    def __init__(self, dtype=np.float32):
        self.dtype = dtype
    def __call__(self, img):
        if isinstance(img, np.ndarray):
            return img.astype(self.dtype, copy=False)
        if isinstance(img, Image.Image):
            arr = np.asarray(img)
            if arr.ndim == 2:              # L
                arr = arr[None, ...]       # (1,H,W)
            else:
                arr = arr.transpose(2, 0, 1)  # (C,H,W)
            return arr.astype(self.dtype)
        raise TypeError("ToArray expects PIL.Image or np.ndarray")

# ---------- PIL/NDArray 双兼容的增强 ----------
class RandomHorizontalFlip:
    """随机水平翻转（PIL 或 CHW ndarray 均可）"""
    def __init__(self, p=0.5):
        self.p = float(p)
    def __call__(self, img):
        if np.random.rand() >= self.p:
            return img
        if _is_pil(img):
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        if _is_ndarray(img):  # CHW
            return np.flip(img, axis=-1).copy()
        raise TypeError

class RandomVerticalFlip:
    """随机垂直翻转"""
    def __init__(self, p=0.0):
        self.p = float(p)
    def __call__(self, img):
        if np.random.rand() >= self.p:
            return img
        if _is_pil(img):
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        if _is_ndarray(img):  # CHW
            return np.flip(img, axis=-2).copy()
        raise TypeError

class Pad:
    """四周填充（常配合 RandomCrop 使用）
    size: int or (pad_h, pad_w)
    fill: 常数填充值（PIL/ndarray 使用0较多）
    """
    def __init__(self, size, fill=0):
        ph, pw = pair(size)
        self.pad = (pw, ph, pw, ph)  # PIL: left, top, right, bottom
        self.ph, self.pw = ph, pw
        self.fill = fill
    def __call__(self, img):
        if _is_pil(img):
            return ImageOps.expand(img, border=self.pad, fill=self.fill)
        if _is_ndarray(img):  # CHW
            c,h,w = img.shape
            pad = ((0,0), (self.ph, self.ph), (self.pw, self.pw))
            return np.pad(img, pad, mode="constant", constant_values=self.fill)
        raise TypeError

class RandomCrop:
    """随机裁剪到指定大小（支持 PIL/CHW）"""
    def __init__(self, size):
        self.oh, self.ow = pair(size)
    def __call__(self, img):
        if _is_pil(img):
            W,H = img.size
            if H == self.oh and W == self.ow:
                return img
            top  = np.random.randint(0, H - self.oh + 1)
            left = np.random.randint(0, W - self.ow + 1)
            return img.crop((left, top, left+self.ow, top+self.oh))
        if _is_ndarray(img):  # CHW
            c,h,w = img.shape
            if h == self.oh and w == self.ow:
                return img
            top  = np.random.randint(0, h - self.oh + 1)
            left = np.random.randint(0, w - self.ow + 1)
            return img[:, top:top+self.oh, left:left+self.ow]
        raise TypeError

class RandomShift:
    """随机平移（像素单位），越界处填 0（支持 PIL/CHW）"""
    def __init__(self, shift=(2,2)):
        self.max_dx, self.max_dy = int(shift[0]), int(shift[1])
    def __call__(self, img):
        dx = np.random.randint(-self.max_dx, self.max_dx + 1)
        dy = np.random.randint(-self.max_dy, self.max_dy + 1)
        if _is_pil(img):
            # PIL 用仿射 + 常数填充
            return ImageOps.offset(ImageOps.expand(img, border=(self.max_dx, self.max_dy, self.max_dx, self.max_dy), fill=0),
                                   dx, dy).crop((self.max_dx, self.max_dy,
                                                 img.size[0]+self.max_dx, img.size[1]+self.max_dy))
        if _is_ndarray(img):
            C,H,W = img.shape
            out = np.zeros_like(img)
            xs = max(0, dx); xe = min(W, W + dx)
            ys = max(0, dy); ye = min(H, H + dy)
            xs2 = max(0, -dx); xe2 = xs2 + (xe - xs)
            ys2 = max(0, -dy); ye2 = ys2 + (ye - ys)
            if xe > xs and ye > ys:
                out[:, ys:ye, xs:xe] = img[:, ys2:ye2, xs2:xe2]
            return out
        raise TypeError

class RandomRotate90:
    """随机 0/90/180/270° 旋转（不插值，最稳妥），支持 PIL/CHW"""
    def __init__(self, p=0.1):
        self.p = float(p)
    def __call__(self, img):
        if np.random.rand() >= self.p:
            return img
        k = np.random.randint(0, 4)
        if k == 0:
            return img
        if _is_pil(img):
            return img.rotate(90*k, resample=Image.NEAREST)
        if _is_ndarray(img):
            return np.rot90(img, k=k, axes=(-2, -1)).copy()
        raise TypeError

class RandomBrightness:
    """亮度抖动：x <- x * factor（PIL/CHW 均可）"""
    def __init__(self, delta=0.2, clip=True):
        self.delta = float(delta)
        self.clip = bool(clip)
    def __call__(self, img):
        factor = 1.0 + (np.random.rand()*2 - 1) * self.delta
        if _is_pil(img):
            return ImageEnhance.Brightness(img).enhance(factor)
        if _is_ndarray(img):
            y = img * factor
            return np.clip(y, 0.0, 255.0 if img.dtype!=np.float32 else 1.0) if self.clip else y
        raise TypeError

class RandomContrast:
    """对比度抖动：x <- (x-mean)*factor + mean（对 PIL 以 0.5 近似）"""
    def __init__(self, delta=0.2, clip=True):
        self.delta = float(delta)
        self.clip = bool(clip)
    def __call__(self, img):
        factor = 1.0 + (np.random.rand()*2 - 1) * self.delta
        if _is_pil(img):
            return ImageEnhance.Contrast(img).enhance(factor)
        if _is_ndarray(img):
            # 假定已归一化到 [0,1]，以 0.5 为中心
            y = (img - 0.5) * factor + 0.5
            return np.clip(y, 0.0, 1.0) if self.clip else y
        raise TypeError

class ColorJitter:
    """Random brightness/contrast/saturation jitter for PIL or ndarray images.

    Args:
        brightness: 0 disables it; a scalar ``d`` samples from ``[1-d, 1+d]``;
            a tuple samples from ``(min, max)``.
        contrast: same rule as brightness.
        saturation: same rule as brightness; RGB images only.
        rng: optional ``numpy.random.Generator`` for deterministic tests.
    """
    def __init__(self, brightness=0.0, contrast=0.0, saturation=0.0, rng=None, clip=True):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.rng = rng
        self.clip = bool(clip)

    def __call__(self, img):
        factors = [
            ("brightness", self._factor(self.brightness)),
            ("contrast", self._factor(self.contrast)),
            ("saturation", self._factor(self.saturation)),
        ]
        order = self._permutation(len(factors))
        out = img
        for idx in order:
            name, factor = factors[int(idx)]
            if factor is None:
                continue
            if name == "brightness":
                out = self._adjust_brightness(out, factor)
            elif name == "contrast":
                out = self._adjust_contrast(out, factor)
            elif name == "saturation":
                out = self._adjust_saturation(out, factor)
        return out

    def _factor(self, value):
        if value is None:
            return None
        if np.isscalar(value):
            delta = float(value)
            if delta == 0:
                return None
            low, high = max(0.0, 1.0 - delta), 1.0 + delta
        else:
            low, high = float(value[0]), float(value[1])
        return float(self._uniform(low, high))

    def _uniform(self, low, high):
        if self.rng is not None and hasattr(self.rng, "uniform"):
            return self.rng.uniform(low, high)
        return np.random.uniform(low, high)

    def _permutation(self, n):
        if self.rng is not None and hasattr(self.rng, "permutation"):
            return self.rng.permutation(n)
        return np.random.permutation(n)

    def _adjust_brightness(self, img, factor):
        if _is_pil(img):
            return ImageEnhance.Brightness(img).enhance(factor)
        if _is_ndarray(img):
            return self._clip(img * factor, img)
        raise TypeError

    def _adjust_contrast(self, img, factor):
        if _is_pil(img):
            return ImageEnhance.Contrast(img).enhance(factor)
        if _is_ndarray(img):
            axes = tuple(range(1, img.ndim)) if _looks_chw(img) else tuple(range(img.ndim - 1))
            mean = img.mean(axis=axes, keepdims=True)
            return self._clip((img - mean) * factor + mean, img)
        raise TypeError

    def _adjust_saturation(self, img, factor):
        if _is_pil(img):
            return ImageEnhance.Color(img).enhance(factor)
        if _is_ndarray(img):
            if _looks_chw(img):
                if img.shape[0] != 3:
                    return img
                gray = (
                    0.299 * img[0:1] +
                    0.587 * img[1:2] +
                    0.114 * img[2:3]
                )
            else:
                if img.shape[-1] != 3:
                    return img
                gray = (
                    0.299 * img[..., 0:1] +
                    0.587 * img[..., 1:2] +
                    0.114 * img[..., 2:3]
                )
            return self._clip((img - gray) * factor + gray, img)
        raise TypeError

    def _clip(self, out, reference):
        if not self.clip:
            return out.astype(reference.dtype, copy=False)
        hi = 1.0 if np.issubdtype(reference.dtype, np.floating) and reference.max(initial=0) <= 1.0 else 255.0
        return np.clip(out, 0.0, hi).astype(reference.dtype, copy=False)


def _looks_chw(array):
    return array.ndim == 3 and array.shape[0] in (1, 3, 4)

class RandomGaussianNoise:
    """加性高斯噪声（建议在 float 域使用），仅 ndarray（CHW）"""
    def __init__(self, sigma=0.03, clip=True):
        self.sigma = float(sigma)
        self.clip = bool(clip)
    def __call__(self, array):
        if not _is_ndarray(array):
            return array
        y = array + np.random.randn(*array.shape) * self.sigma
        return np.clip(y, 0.0, 1.0) if self.clip else y

class RandomErasing:
    """随机擦除（Cutout 风格），仅 ndarray（CHW）"""
    def __init__(self, erase_prob=0.5, area_ratio=(0.02, 0.15), min_aspect=0.3, value=0.0):
        self.erase_prob = float(erase_prob)
        self.area_ratio = area_ratio
        self.min_aspect = float(min_aspect)
        self.value = value
    def __call__(self, array):
        if not _is_ndarray(array) or np.random.rand() >= self.erase_prob:
            return array
        C,H,W = array.shape
        area = H*W
        for _ in range(10):
            target = np.random.uniform(*self.area_ratio) * area
            aspect = np.random.uniform(self.min_aspect, 1/self.min_aspect)
            h = int(round(np.sqrt(target * aspect)))
            w = int(round(np.sqrt(target / aspect)))
            if h < H and w < W:
                top  = np.random.randint(0, H - h)
                left = np.random.randint(0, W - w)
                array[:, top:top+h, left:left+w] = self.value
                return array
        return array


class Standardize:
    """按通道做标准化：x = (x - mean) / std
    支持 (C,H,W) 或 (N,C,H,W)。mean/std 接受标量或按通道列表/ndarray。
    """
    def __init__(self, mean, std, eps=1e-12):
        self.mean = np.asarray(mean, dtype=np.float32).reshape(-1)
        self.std  = np.asarray(std,  dtype=np.float32).reshape(-1)
        self.eps  = eps

    def __call__(self, x):
        # x: (C,H,W) or (N,C,H,W), float32 in [0,1] （已做 Normalize(0,255)）
        if x.ndim == 3:
            C, H, W = x.shape
            mean = self._broadcast(self.mean, C)[:, None, None]
            std  = self._broadcast(self.std,  C)[:, None, None]
        elif x.ndim == 4:
            N, C, H, W = x.shape
            mean = self._broadcast(self.mean, C)[None, :, None, None]
            std  = self._broadcast(self.std,  C)[None, :, None, None]
        else:
            raise ValueError(f"Unsupported shape for Standardize: {x.shape}")
        return (x - mean) / (std + self.eps)

    @staticmethod
    def _broadcast(arr, C):
        if arr.size == 1:
            return np.full((C,), arr.item(), dtype=np.float32)
        if arr.size != C:
            raise ValueError(f"Channel mismatch: got {arr.size}, expect {C}")
        return arr

#
# # 训练增强
# transform_train = Compose([
#     # 若你的 MNIST 不是 PIL 流程，下面两行可省略
#     # Convert("L"),                # 强制单通道
#     # Resize((28, 28)),           # 保证尺寸一致
#
#     ToArray(np.float32),          # PIL->ndarray (CHW)；或直接你的 ToFloat/Normalize
#     Normalize(0.0, 255.0),        # [0,255] -> [0,1]
#
#     Pad(2),                       # 四周补 2（-> 32x32）
#     RandomCrop((28, 28)),         # 随机裁回 28x28（经典 trick）
#     RandomShift((2, 2)),          # 小幅平移
#     RandomRotate90(p=0.1),        # 小概率整数旋转（不失真）
#     RandomBrightness(0.2),
#     RandomContrast(0.2),
#     RandomGaussianNoise(0.03),
#     RandomErasing(erase_prob=0.5),
#
#     Flatten(),                    # 放最后（仅当 MLP 需要）
# ])
#
# # 测试/验证只做确定性处理
# transform_test = Compose([
#     ToArray(np.float32),
#     Normalize(0.0, 255.0),
#     Flatten(),
# ])
