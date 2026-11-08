"""
DonkeyCar Dataset for HyTorch
==============================
数据格式（文件名即标签）：
    {index}_{angle}.jpg
    e.g.  0_-0.0889.jpg  →  steering angle = -0.0889

图像预处理（NVIDIA PilotNet 输入规格）：
    原始 (120, 160, 3)  RGB
    → 裁去顶部 40 行（天空）→ (80, 160, 3)
    → resize 到 (66, 200, 3)
    → 归一化到 [-1, 1]
    → 转置 CHW → (3, 66, 200)  float32

标签：
    np.float32  steering angle（单值，回归）
"""

import os
import re
import numpy as np

try:
    from PIL import Image
    _USE_PIL = True
except ImportError:
    import cv2
    _USE_PIL = False


# ── 预处理参数 ──────────────────────────────────────────────────
TARGET_H  = 66
TARGET_W  = 200
CUT_TOP   = 40   # 裁去天空行数（原图高 120，裁后剩 80）


def _preprocess(img_path: str) -> np.ndarray:
    """
    读取一张图片，返回 (3, 66, 200) float32，值域 [-1, 1]
    """
    if _USE_PIL:
        img = np.array(Image.open(img_path).convert('RGB'), dtype=np.float32)
    else:
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB).astype(np.float32)
    # img: (H, W, 3)

    # 裁去顶部天空
    img = img[CUT_TOP:, :, :]        # (80, 160, 3)

    # Resize → (66, 200)
    if _USE_PIL:
        pil = Image.fromarray(img.astype(np.uint8))
        pil = pil.resize((TARGET_W, TARGET_H), Image.BILINEAR)
        img = np.array(pil, dtype=np.float32)
    else:
        img = cv2.resize(img, (TARGET_W, TARGET_H)).astype(np.float32)

    # [0,255] → [-1, 1]
    img = img / 127.5 - 1.0

    # HWC → CHW
    return img.transpose(2, 0, 1).astype(np.float32)  # (3, 66, 200)


class DonkeyDataset:
    """
    读取 {index}_{angle}.jpg 格式的 DonkeyCar 数据集。

    示例
    ----
    from tub_dataset import DonkeyDataset
    from hytorch.dataloaders import DataLoader

    ds = DonkeyDataset('/path/to/data')
    print(len(ds))   # 10000

    loader = DataLoader(ds, batch_size=64, num_workers=4)
    for x, t in loader:
        # x: (64, 3, 66, 200)  float32
        # t: (64, 1)           float32  steering angle
        pass
    """

    def __init__(self, data_dir: str, augment: bool = False):
        """
        Parameters
        ----------
        data_dir : str   包含 jpg 文件的目录
        augment  : bool  随机水平翻转（翻转时角度取反）
        """
        self.data_dir = data_dir
        self.augment  = augment

        # 扫描所有匹配文件
        pattern = re.compile(r'^(\d+)_([+-]?\d+\.\d+)\.jpg$')
        records = []
        for fname in os.listdir(data_dir):
            m = pattern.match(fname)
            if m:
                idx   = int(m.group(1))
                angle = float(m.group(2))
                path  = os.path.join(data_dir, fname)
                records.append((idx, path, angle))

        if not records:
            raise FileNotFoundError(
                f"在 {data_dir} 中找不到符合 {{index}}_{{angle}}.jpg 格式的文件。"
            )

        # 按 index 排序，保证顺序可复现
        records.sort(key=lambda r: r[0])
        self._paths  = [r[1] for r in records]
        self._angles = np.array([r[2] for r in records], dtype=np.float32)

        # 供 DatasetAnalyzer 快速读取标签
        self.label = self._angles

        print(f"[DonkeyDataset] {len(self._paths)} 张图  "
              f"角度范围 [{self._angles.min():.4f}, {self._angles.max():.4f}]  "
              f"均值={self._angles.mean():.4f}  std={self._angles.std():.4f}")

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, index: int):
        """
        返回 (x, t)
            x : np.ndarray (3, 66, 200) float32
            t : np.ndarray (1,)         float32  [steering_angle]
        """
        x     = _preprocess(self._paths[index])
        angle = self._angles[index]

        if self.augment and np.random.rand() < 0.5:
            x     = x[:, :, ::-1].copy()   # 水平翻转
            angle = -angle                  # 翻转时转向取反

        t = np.array([angle], dtype=np.float32)  # shape (1,)
        return x, t

    def split(self, val_ratio: float = 0.1, seed: int = 42):
        """
        按比例划分训练 / 验证集，返回 (train_ds, val_ds)。
        """
        rng   = np.random.RandomState(seed)
        idx   = rng.permutation(len(self))
        n_val = max(1, int(len(self) * val_ratio))

        train_ds = self._subset(idx[n_val:],  augment=self.augment)
        val_ds   = self._subset(idx[:n_val],  augment=False)
        return train_ds, val_ds

    def _subset(self, indices, augment: bool):
        ds          = DonkeyDataset.__new__(DonkeyDataset)
        ds.data_dir = self.data_dir
        ds.augment  = augment
        ds._paths   = [self._paths[i]  for i in indices]
        ds._angles  = self._angles[indices]
        ds.label    = ds._angles
        return ds


# 向后兼容别名
TubDataset = DonkeyDataset
