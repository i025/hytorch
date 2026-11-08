# hytorch/transformer/masks.py
import numpy as np

def causal_mask(T, dtype=np.float32, neg_inf=-1e9):
    """
    返回 shape = (1, 1, T, T) 的 causal mask
    允许关注自己和过去，禁止关注未来（上三角为 -inf）
    """
    m = np.triu(np.ones((T, T), dtype=dtype), k=1) * dtype(neg_inf)
    return m[None, None, :, :]


def padding_mask(token_ids, pad_id=0):
    """
    生成 padding mask，pad位置为 -inf，其它为 0
    token_ids: (N, S) int
    return: additive mask (N, 1, 1, S)
    """
    token_ids = np.asarray(token_ids)
    pad = (token_ids == pad_id)
    mask = np.where(pad, np.array(-1e9, dtype=np.float32), np.array(0.0, dtype=np.float32))
    mask = mask.reshape(mask.shape[0], 1, 1, mask.shape[1])
    return mask
