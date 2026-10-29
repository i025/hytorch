"""CPU model of the exact online tree topology, separate from the 3A reference."""
import numpy as np


def online_tree(partials):
    levels = [None] * len(partials).bit_length()
    for count, value in enumerate(partials):
        value = np.float32(value)
        level = 0
        while count & (1 << level):
            value = np.add(levels[level], value, dtype=np.float32)
            level += 1
        levels[level] = value
    total = None
    for level, value in enumerate(levels):
        if len(partials) & (1 << level):
            total = value if total is None else np.add(value, total, dtype=np.float32)
    return total
