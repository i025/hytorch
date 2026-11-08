# hytorch/transformer/attention.py
import math
from hytorch import Layer
import hytorch.layers as L
import hytorch.functions as F

class MultiHeadAttention(Layer):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # 用 LinearND（专门支持 (N,T,D)）
        self.W_q = L.LinearND(d_model, d_model)
        self.W_k = L.LinearND(d_model, d_model)
        self.W_v = L.LinearND(d_model, d_model)
        self.W_o = L.LinearND(d_model, d_model)

        self.dropout = L.Dropout(dropout)

    def _split_heads(self, x):
        # (N,T,D) -> (N,H,T,Dh)
        N, T, D = x.shape
        x = F.reshape(x, (N, T, self.num_heads, self.d_head))
        return F.transpose(x, (0, 2, 1, 3))

    def _merge_heads(self, x):
        # x: (N,H,T,Dh) -> (N,T,D)
        N, H, T, Dh = x.shape
        x = F.transpose(x, (0, 2, 1, 3))
        return F.reshape(x, (N, T, H * Dh))

    def forward(self, q, k=None, v=None, mask=None):
        if k is None:
            k = q
        if v is None:
            v = k

        q = self._split_heads(self.W_q(q))  # (N,H,T,Dh)
        k = self._split_heads(self.W_k(k))
        v = self._split_heads(self.W_v(v))

        N, H, Tq, Dh = q.shape
        Tk = k.shape[2]

        # ---- score: (N,H,Tq,Tk) ----
        q2 = F.reshape(q, (N * H, Tq, Dh))
        k2 = F.reshape(k, (N * H, Tk, Dh))
        k2T = F.transpose(k2, (0, 2, 1))             # (N*H, Dh, Tk)

        score = F.bmm(q2, k2T)                        # (N*H, Tq, Tk)
        score = score / math.sqrt(self.d_head)
        score = F.reshape(score, (N, H, Tq, Tk))

        if mask is not None:
            score = score + mask

        attn = F.softmax(score, axis=-1)
        attn = self.dropout(attn)

        # ---- out: (N,H,Tq,Dh) ----
        attn2 = F.reshape(attn, (N * H, Tq, Tk))
        v2    = F.reshape(v,    (N * H, Tk, Dh))
        out2  = F.bmm(attn2, v2)                      # (N*H, Tq, Dh)
        out   = F.reshape(out2, (N, H, Tq, Dh))

        out = self._merge_heads(out)                  # (N,Tq,D)
        return self.W_o(out)
