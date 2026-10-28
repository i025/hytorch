# hytorch/transformer/encoder.py
from hytorch import Layer
import hytorch.layers as L
from .encoder_block import TransformerEncoderBlock


class TransformerEncoder(Layer):
    """
    Stack of EncoderBlocks (Pre-LN inside block)
    可选最后加一层 LayerNorm（很多实现会加，稳定）
    """
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.0, final_norm=True):
        super().__init__()
        self.layers = []
        for i in range(num_layers):
            blk = TransformerEncoderBlock(d_model, num_heads, d_ff, dropout=dropout)
            self.layers.append(blk)
            setattr(self, f"layer{i}", blk)

        self.final_norm = final_norm
        self.ln = L.LayerNorm(d_model) if final_norm else None

    def forward(self, x, mask=None):
        for blk in self.layers:
            x = blk(x, mask=mask)
        if self.final_norm:
            x = self.ln(x)
        return x
