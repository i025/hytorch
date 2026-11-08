"""
DonkeyNet — NVIDIA PilotNet (HyTorch 实现)
==========================================
按论文图 2.39 结构，输出改为单值转向角回归：

输入  (N, 3, 66, 200)   归一化到 [-1, 1]
  Conv2d(24, 5×5, s=2)  → (N, 24, 31, 98)
  Conv2d(36, 5×5, s=2)  → (N, 36, 14, 47)
  Conv2d(48, 5×5, s=2)  → (N, 48,  5, 22)
  Conv2d(64, 3×3, s=1)  → (N, 64,  3, 20)
  Conv2d(64, 3×3, s=1)  → (N, 64,  1, 18)
  Flatten               → (N, 1152)
  Linear(100) ReLU
  Linear(50)  ReLU
  Linear(1)   tanh      → 转向角 ∈ [-1, 1]

Loss : mean_square_error（单值回归）
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import hytorch
import hytorch.functions as F
import hytorch.layers    as L
from hytorch.layers import Layer


class DonkeyNet(Layer):
    """
    NVIDIA PilotNet for DonkeyCar — 单转向角输出版本

    输入  : (N, 3, 66, 200)  float32, 已归一化到 [-1, 1]
    输出  : (N, 1)            float32, 转向角 ∈ [-1, 1]
    """

    def __init__(self, dropout_rate: float = 0.1):
        super().__init__()

        # ── 卷积层（严格按 PilotNet 尺寸）──────────────────────
        self.conv1 = L.Conv2d(out_channels=24, kernel_size=5, stride=2, pad=0)
        self.conv2 = L.Conv2d(out_channels=36, kernel_size=5, stride=2, pad=0)
        self.conv3 = L.Conv2d(out_channels=48, kernel_size=5, stride=2, pad=0)
        self.conv4 = L.Conv2d(out_channels=64, kernel_size=3, stride=1, pad=0)
        self.conv5 = L.Conv2d(out_channels=64, kernel_size=3, stride=1, pad=0)

        # ── 全连接层 ────────────────────────────────────────────
        self.fc1 = L.Linear(out_size=100)
        self.fc2 = L.Linear(out_size=50)
        self.fc3 = L.Linear(out_size=1)   # 单值：转向角

        # Dropout（训练时生效）
        self.drop = L.Dropout(p=dropout_rate)

    def forward(self, x):
        # ── 卷积块 ──
        x = F.relu(self.conv1(x))   # → (N, 24, 31, 98)
        x = F.relu(self.conv2(x))   # → (N, 36, 14, 47)
        x = F.relu(self.conv3(x))   # → (N, 48,  5, 22)
        x = F.relu(self.conv4(x))   # → (N, 64,  3, 20)
        x = F.relu(self.conv5(x))   # → (N, 64,  1, 18)

        # ── Flatten ──
        x = F.reshape(x, (x.shape[0], -1))   # → (N, 1152)

        # ── 全连接块 ──
        x = self.drop(F.relu(self.fc1(x)))    # → (N, 100)
        x = self.drop(F.relu(self.fc2(x)))    # → (N, 50)
        x = F.tanh(self.fc3(x))              # → (N, 1)

        return x

    def predict(self, img_np: np.ndarray) -> float:
        """
        单张图片推理，返回转向角（float）。

        img_np : (3, 66, 200) float32
        """
        if img_np.ndim == 3:
            img_np = img_np[np.newaxis]
        with hytorch.no_grad():
            y = self(img_np)
        return float(y.data.ravel()[0])


def build_model(dropout_rate: float = 0.1) -> DonkeyNet:
    return DonkeyNet(dropout_rate=dropout_rate)


# ── 结构验证 ────────────────────────────────────────────────────
if __name__ == '__main__':
    from hytorch.core import Variable

    model = build_model()
    dummy = Variable(np.zeros((4, 3, 66, 200), dtype=np.float32))
    out   = model(dummy)

    print("=" * 50)
    print("  DonkeyNet 结构验证 (NVIDIA PilotNet)")
    print("=" * 50)

    layer_info = [
        ("Conv1 (24, 5×5, s=2)", "(N, 24, 31, 98)"),
        ("Conv2 (36, 5×5, s=2)", "(N, 36, 14, 47)"),
        ("Conv3 (48, 5×5, s=2)", "(N, 48,  5, 22)"),
        ("Conv4 (64, 3×3, s=1)", "(N, 64,  3, 20)"),
        ("Conv5 (64, 3×3, s=1)", "(N, 64,  1, 18)"),
        ("Flatten",               "(N, 1152)       "),
        ("Linear(100) + ReLU",    "(N, 100)        "),
        ("Linear(50)  + ReLU",    "(N, 50)         "),
        ("Linear(1)   + tanh",    "(N, 1)          "),
    ]
    for name, shape in layer_info:
        print(f"  {name:<25} → {shape}")

    total = sum(p.data.size for p in model.params() if p.data is not None)
    print(f"\n  输入  shape  : (4, 3, 66, 200)")
    print(f"  输出  shape  : {out.shape}")
    print(f"  总参数量     : {total:,}")
    assert out.shape == (4, 1), f"期望 (4,1)，实际 {out.shape}"
    print("  验证通过 ✓")
