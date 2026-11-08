import hytorch
from hytorch import Layer
from .decoder_block import TransformerDecoderBlock


class TransformerDecoderOnly(Layer):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.0, final_norm=True):
        super().__init__()
        self.layers = []
        for i in range(num_layers):
            blk = TransformerDecoderBlock(d_model, num_heads, d_ff, dropout=dropout)
            self.layers.append(blk)
            setattr(self, f"layer{i}", blk)

        self.final_norm = final_norm
        self.ln = hytorch.layers.LayerNorm(d_model) if final_norm else None

    def forward(self, x, self_mask=None):
        for blk in self.layers:
            x = blk(x, x, self_mask=self_mask)  # 解码器使用自身的 Q、K 和 V
        if self.final_norm:
            x = self.ln(x)
        return x
