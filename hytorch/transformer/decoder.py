# hytorch/transformer/decoder.py
from hytorch import Layer
import hytorch.layers as L
from .decoder_block import TransformerDecoderBlock


class TransformerDecoder(Layer):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.0, final_norm=True):
        super().__init__()
        self.layers = []
        for i in range(num_layers):
            blk = TransformerDecoderBlock(d_model, num_heads, d_ff, dropout=dropout)
            self.layers.append(blk)
            setattr(self, f"layer{i}", blk)

        self.final_norm = final_norm
        self.ln = L.LayerNorm(d_model) if final_norm else None

    def forward(self, x, enc_out, self_mask=None, cross_mask=None):
        for blk in self.layers:
            x = blk(x, enc_out, self_mask=self_mask, cross_mask=cross_mask)
        if self.final_norm:
            x = self.ln(x)
        return x
