# seq2seq_wrapper.py
import numpy as np
from hytorch import Layer, cuda, as_variable

# 复用你现有的 TokenEmbedding / SinusoidalPositionalEncoding
from .embedding import TokenEmbedding, SinusoidalPositionalEncoding

# 复用你现有的 Transformer / LinearND
# 下面 import 路径按你项目实际改
from .transformer import Transformer
from hytorch.layers import LinearND  # 你 layers.py 里已经有 LinearND


NEG_INF = -1e9


def make_padding_mask(token_ids, pad_id=0):
    """
    token_ids: (N,S) int
    return: additive mask (N,1,1,S): pad -> -1e9, else 0
    """
    token_ids = as_variable(token_ids)
    xp = cuda.get_array_module(token_ids.data)
    pad = (token_ids.data == pad_id)
    mask = xp.where(
        pad,
        xp.array(NEG_INF, dtype=xp.float32),
        xp.array(0.0, dtype=xp.float32),
    )
    mask = mask.reshape(mask.shape[0], 1, 1, mask.shape[1])
    return as_variable(mask)


def make_causal_mask(T, xp=np):
    """
    (1,1,T,T) 上三角(未来)为 -1e9
    """
    m = xp.triu(xp.ones((T, T), dtype=xp.float32), k=1) * xp.array(NEG_INF, dtype=xp.float32)
    return as_variable(m.reshape(1, 1, T, T))


class TransformerSeq2Seq(Layer):
    def __init__(self, vocab_size, d_model,
                 num_enc_layers, num_dec_layers,
                 num_heads, d_ff,
                 max_len=2048, pad_id=0, dtype=np.float32, dropout=0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_id = pad_id

        self.src_emb = TokenEmbedding(vocab_size, d_model, dtype=dtype)
        self.tgt_emb = TokenEmbedding(vocab_size, d_model, dtype=dtype)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len=max_len, dtype=dtype)

        self.transformer = Transformer(
            num_enc_layers=num_enc_layers,
            num_dec_layers=num_dec_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout
        )

        # 输出头： (N,T,D) -> (N,T,V)
        self.lm_head = LinearND(d_model, vocab_size, dtype=dtype)

    def _emb_plus_pos(self, ids, is_src: bool, start_pos=0):
        ids = as_variable(ids)
        x = self.src_emb(ids) if is_src else self.tgt_emb(ids)   # (N,T,D)
        pe = self.pos(x, start_pos=start_pos)                    # (1,T,D)
        return x + pe                                            # (N,T,D)

    def forward(self, src_ids, tgt_ids):
        """
        src_ids: (N,S)
        tgt_ids: (N,T) 作为 decoder input (通常 BOS + y[:-1])
        return: logits (N,T,V)
        """
        src_ids = as_variable(src_ids)
        tgt_ids = as_variable(tgt_ids)
        xp = cuda.get_array_module(src_ids.data)

        src = self._emb_plus_pos(src_ids, is_src=True)    # (N,S,D)
        tgt = self._emb_plus_pos(tgt_ids, is_src=False)   # (N,T,D)

        src_mask = make_padding_mask(src_ids, pad_id=self.pad_id)    # (N,1,1,S)

        tgt_pad = make_padding_mask(tgt_ids, pad_id=self.pad_id)     # (N,1,1,T)
        causal = make_causal_mask(tgt.shape[1], xp=xp)               # (1,1,T,T)
        tgt_mask = tgt_pad + causal                                  # broadcast -> (N,1,T,T)

        cross_mask = src_mask                                        # (N,1,1,S) broadcast -> (N,1,T,S)

        out = self.transformer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask, cross_mask=cross_mask)  # (N,T,D)
        logits = self.lm_head(out)                                   # (N,T,V)
        return logits

    def greedy_decode(self, src_ids, max_len=256):
        """
        使用贪婪解码生成目标序列
        """
        batch_size = src_ids.shape[0]
        xp = cuda.get_array_module(src_ids)

        # 初始化目标输入为 <BOS> 标记
        tgt = xp.full((batch_size, 1), BOS_ID, dtype=np.int64)  # 使用 <BOS> 标记进行开始

        for t in range(max_len):
            # 获取当前的解码结果
            logits = self(src_ids, tgt)  # 使用现有的 forward 方法

            # 选择每个位置上概率最大的 token
            next_token = logits.data.argmax(axis=-1)[:, -1]  # 选择最后一个 token

            # 将预测的 token 追加到目标序列
            tgt = xp.concatenate((tgt, next_token[:, None]), axis=-1)

            # 如果所有序列都已经生成 <EOS>，则停止
            if (next_token == EOS_ID).all():  # 如果所有序列已经生成结束符
                break

        return tgt

    def encode(self, src_ids):
        """
        src_ids: (N,S) int64 (numpy/cupy)
        return:
          enc_out: (N,S,D)
          src_mask: (N,1,1,S) additive
        """
        src_ids = as_variable(src_ids)
        xp = cuda.get_array_module(src_ids.data)

        src = self._emb_plus_pos(src_ids, is_src=True, start_pos=0)
        src_mask = make_padding_mask(src_ids, pad_id=self.pad_id)

        enc_out = self.transformer.encoder(src, mask=src_mask)
        return enc_out, src_mask

    def decode_step(self, enc_out, src_mask, last_token_ids, step_pos, kv_cache=None):
        """
        单步增量解码（KV cache 版本）
        last_token_ids: (N,) or (N,1)  只喂“当前最后一个 token”
        step_pos: int 这个 token 的位置（BOS=0, 第一位数字=1, ...）
        kv_cache: decoder 的 cache list（长度= num_dec_layers），或 None
        return:
          logits: (N,1,V)
          new_cache: 更新后的 cache list
        """
        last_token_ids = as_variable(last_token_ids)
        if len(last_token_ids.shape) == 1:
            # (N,) -> (N,1)
            last_token_ids = as_variable(last_token_ids.data.reshape(last_token_ids.shape[0], 1))

        tgt = self._emb_plus_pos(last_token_ids, is_src=False, start_pos=step_pos)  # (N,1,D)

        dec_out, new_cache = self.transformer.decoder(
            tgt, enc_out,
            self_mask=None,             # Tq=1 时不需要 causal mask
            cross_mask=src_mask,
            kv_cache=kv_cache,
            use_cache=True
        )
        logits = self.lm_head(dec_out)  # (N,1,V)
        return logits, new_cache