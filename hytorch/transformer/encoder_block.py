# hytorch/transformer/encoder_block.py
from hytorch import Layer
import hytorch.layers as L
from .attention import MultiHeadAttention
from .feedforward import PositionwiseFeedForward


class TransformerEncoderBlock(Layer):
    """
    Pre-LN Transformer Encoder Block

    x -> LN -> SelfAttn -> Dropout -> +res
      -> LN -> FFN     -> Dropout -> +res
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.0):
        super().__init__()
        self.ln1 = L.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.drop1 = L.Dropout(dropout)

        self.ln2 = L.LayerNorm(d_model)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout=dropout)
        self.drop2 = L.Dropout(dropout)

    def forward(self, x, mask=None):
        # --- Self-Attention sublayer ---
        h = self.ln1(x)
        h = self.attn(h, mask=mask)
        h = self.drop1(h)
        x = x + h

        # --- FFN sublayer ---
        h = self.ln2(x)
        h = self.ffn(h)
        h = self.drop2(h)
        x = x + h

        return x
