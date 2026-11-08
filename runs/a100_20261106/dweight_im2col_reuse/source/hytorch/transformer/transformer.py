# hytorch/transformer/transformer.py
from hytorch import Layer
from .encoder import TransformerEncoder
from .decoder import TransformerDecoder


class Transformer(Layer):
    def __init__(self, num_enc_layers, num_dec_layers, d_model, num_heads, d_ff, dropout=0.0):
        super().__init__()
        self.encoder = TransformerEncoder(num_enc_layers, d_model, num_heads, d_ff, dropout=dropout, final_norm=True)
        self.decoder = TransformerDecoder(num_dec_layers, d_model, num_heads, d_ff, dropout=dropout, final_norm=True)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, cross_mask=None):
        enc_out = self.encoder(src, mask=src_mask)
        dec_out = self.decoder(tgt, enc_out, self_mask=tgt_mask, cross_mask=cross_mask)
        return dec_out
