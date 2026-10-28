import math
import random

import numpy as np


class DatasetAnalyzer:
    """
    纯文字（无 matplotlib）的数据集统计分析工具。
    支持任何实现了 __len__ 和 __getitem__ 的 Dataset。
    """

    def __init__(self, dataset, label_names: dict = None):
        """
        dataset    : HyTorch Dataset 实例
        label_names: {int: str}，可选，用于报告中显示类别名称
        """
        self.dataset = dataset
        self.label_names = label_names or {}

    # ------------------------------------------------------------------
    # 类别分布
    # ------------------------------------------------------------------
    def class_distribution(self, max_samples: int = None) -> dict:
        """
        返回 {class_id: count}，按 class_id 升序排列。

        快速路径：dataset 有 .label 属性（原始标签数组）→ np.bincount，O(N)。
        回退路径：逐条 __getitem__ 取第二个元素（会触发 transform）。
        """
        ds = self.dataset

        # 快速路径
        if hasattr(ds, 'label') and ds.label is not None:
            labels = np.asarray(ds.label).ravel().astype(int)
            if max_samples is not None:
                labels = labels[:max_samples]
            counts = np.bincount(labels)
            return {i: int(c) for i, c in enumerate(counts) if c > 0}

        # 回退路径
        n = len(ds)
        if max_samples is not None:
            n = min(n, max_samples)
        label_list = []
        for i in range(n):
            _, t = ds[i]
            label_list.append(int(t) if not hasattr(t, '__len__') else int(np.argmax(t)))
        counts = {}
        for lbl in label_list:
            counts[lbl] = counts.get(lbl, 0) + 1
        return dict(sorted(counts.items()))

    # ------------------------------------------------------------------
    # 像素统计
    # ------------------------------------------------------------------
    def pixel_stats(self, sample_n: int = 1000, per_channel: bool = True) -> dict:
        """
        随机抽 sample_n 个样本（调用 dataset[i]，触发 transform），
        统计变换后的像素分布。

        返回：
        {
          'mean'    : float 或 np.ndarray (per_channel),
          'std'     : float 或 np.ndarray (per_channel),
          'min'     : float,
          'max'     : float,
          'shape'   : tuple,
          'n_sampled': int,
        }
        """
        ds = self.dataset
        n = len(ds)
        sample_n = min(sample_n, n)
        indices = random.sample(range(n), sample_n)

        samples = []
        for idx in indices:
            x, _ = ds[idx]
            samples.append(np.asarray(x, dtype=np.float32))

        arr = np.stack(samples, axis=0)  # (N, ...)
        shape = arr.shape[1:]

        g_min = float(arr.min())
        g_max = float(arr.max())

        if per_channel and arr.ndim >= 3:
            # arr shape: (N, C, H, W) or (N, C, L)
            axes = tuple(range(2, arr.ndim)) + (0,)   # collapse H,W,N
            mean = arr.mean(axis=(0,) + tuple(range(2, arr.ndim)))  # (C,)
            std  = arr.std( axis=(0,) + tuple(range(2, arr.ndim)))
        else:
            mean = float(arr.mean())
            std  = float(arr.std())

        return {
            'mean'     : mean,
            'std'      : std,
            'min'      : g_min,
            'max'      : g_max,
            'shape'    : shape,
            'n_sampled': sample_n,
        }

    # ------------------------------------------------------------------
    # 均衡性
    # ------------------------------------------------------------------
    def imbalance_ratio(self) -> float:
        """max_count / min_count。1.0 = 完全均衡。"""
        dist = self.class_distribution()
        counts = list(dist.values())
        if not counts or min(counts) == 0:
            return float('inf')
        return max(counts) / min(counts)

    # ------------------------------------------------------------------
    # 完整报告
    # ------------------------------------------------------------------
    def report(self, sample_n_stats: int = 1000) -> str:
        lines = []

        sep = '═' * 52
        lines.append(sep)
        lines.append('  HyTorch DatasetAnalyzer Report')

        ds_name = type(self.dataset).__name__
        n_total = len(self.dataset)
        lines.append(f'  Dataset : {ds_name}   N = {n_total}')
        lines.append(sep)

        # --- 类别分布 ---
        lines.append('')
        lines.append('[Class Distribution]')
        dist = self.class_distribution()
        max_count = max(dist.values()) if dist else 1
        bar_width = 30
        for cls_id, count in dist.items():
            name = self.label_names.get(cls_id, '')
            label_str = f'{cls_id}' + (f' ({name})' if name else '')
            label_str = label_str.ljust(16)
            filled = math.ceil(bar_width * count / max_count)
            bar = '█' * filled + '░' * (bar_width - filled)
            pct = 100.0 * count / n_total
            lines.append(f'  {label_str}: {bar} {count:6d} ({pct:5.1f}%)')

        ratio = self.imbalance_ratio()
        balance_tag = 'BALANCED' if ratio < 1.5 else ('MILD IMBALANCE' if ratio < 5 else 'SEVERE IMBALANCE')
        lines.append(f'  Imbalance Ratio: {ratio:.2f}  [{balance_tag}]')

        # --- 像素统计 ---
        lines.append('')
        lines.append(f'[Pixel Statistics  (sampled n={sample_n_stats}, post-transform)]')
        try:
            stats = self.pixel_stats(sample_n=sample_n_stats)
            lines.append(f'  Shape : {stats["shape"]}')
            lines.append(f'  Range : [{stats["min"]:.4f}, {stats["max"]:.4f}]')
            mean, std = stats['mean'], stats['std']
            if hasattr(mean, '__len__'):
                for c, (m, s) in enumerate(zip(mean, std)):
                    lines.append(f'  Ch {c}  : mean={m:.4f}  std={s:.4f}')
            else:
                lines.append(f'  Global: mean={mean:.4f}  std={std:.4f}')
        except Exception as e:
            lines.append(f'  [统计失败: {e}]')

        # --- 建议 ---
        lines.append('')
        lines.append('[Recommendation]')
        if ratio < 1.5:
            lines.append('  - 类别分布均衡，无需重采样。')
        elif ratio < 5:
            lines.append('  - 存在轻度类别不均衡，可考虑 WeightedSampler 或数据增强。')
        else:
            lines.append('  - 存在严重类别不均衡，建议过采样少数类或使用 Focal Loss。')

        lines.append(sep)
        report_str = '\n'.join(lines)
        print(report_str)
        return report_str


# ------------------------------------------------------------------
# 便利函数
# ------------------------------------------------------------------
def analyze_dataset(dataset, label_names: dict = None, sample_n: int = 1000) -> str:
    """一行调用：打印并返回完整分析报告。"""
    return DatasetAnalyzer(dataset, label_names).report(sample_n_stats=sample_n)
