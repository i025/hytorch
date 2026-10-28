# hytorch/transformer/decoder_block.py
from hytorch import Layer
import hytorch.layers as L
from .attention import MultiHeadAttention
from .feedforward import PositionwiseFeedForward


class TransformerDecoderBlock(Layer):
    """
    Pre-LN Decoder Block

    x -> LN -> SelfAttn(causal) -> +res
      -> LN -> CrossAttn(enc)   -> +res
      -> LN -> FFN             -> +res
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.0):
        super().__init__()

        self.ln1 = L.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.drop1 = L.Dropout(dropout)

        self.ln2 = L.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.drop2 = L.Dropout(dropout)

        self.ln3 = L.LayerNorm(d_model)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout=dropout)
        self.drop3 = L.Dropout(dropout)

    def forward(self, x, enc_out, self_mask=None, cross_mask=None):
        # 1) masked self-attn
        h = self.ln1(x)
        h = self.self_attn(h, mask=self_mask)
        h = self.drop1(h)
        x = x + h

        # 2) cross-attn (q from decoder, k/v from encoder)
        h = self.ln2(x)
        h = self.cross_attn(h, enc_out, enc_out, mask=cross_mask)
        h = self.drop2(h)
        x = x + h

        # 3) ffn
        h = self.ln3(x)
        h = self.ffn(h)
        h = self.drop3(h)
        x = x + h

        return x
