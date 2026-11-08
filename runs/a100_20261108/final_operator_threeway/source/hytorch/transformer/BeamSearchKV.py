# BeamSearchKV.py
import numpy as np
import hytorch.functions as F
from hytorch import as_variable, cuda

NEG_INF = -1e9

class BeamSearchDecoderKV:
    def __init__(self, model, beam_size=5, max_len=32, bos_id=1, eos_id=2, pad_id=0,
                 min_len=1, length_penalty_alpha=0.6, repeat_penalty=0.0):
        self.model = model
        self.beam_size = beam_size
        self.max_len = max_len
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.min_len = min_len
        self.length_penalty_alpha = length_penalty_alpha
        self.repeat_penalty = repeat_penalty

    def _len_penalty(self, L):
        # (5+L)^a / (5+1)^a
        a = self.length_penalty_alpha
        return ((5.0 + L) / 6.0) ** a if a > 0 else 1.0

    def decode(self, src_ids):
        """
        src_ids: (N,S) int64 (numpy/cupy)
        return: results[b] = list of (seq(list[int]), score(float))
        """
        xp = cuda.get_array_module(src_ids)
        N = int(src_ids.shape[0])
        all_results = []

        for b in range(N):
            src_b = src_ids[b:b+1]
            enc_out, src_mask = self.model.encode(src_b)

            # beam element: (seq, logp, kv_cache)
            beams = [([self.bos_id], 0.0, None)]
            finished = []

            for _ in range(self.max_len):
                cand = []

                for seq, logp, cache in beams:
                    if seq[-1] == self.eos_id:
                        finished.append((seq, logp))
                        continue

                    # 当前最后 token
                    last = xp.asarray([seq[-1]], dtype=xp.int64).reshape(1, 1)
                    step_pos = len(seq) - 1  # BOS=0

                    logits, new_cache = self.model.decode_step(
                        enc_out, src_mask,
                        last_token_ids=last,
                        step_pos=step_pos,
                        kv_cache=cache
                    )
                    # logits: (1,1,V)
                    lp = F.log_softmax(as_variable(logits.data[0, 0]), axis=-1).data  # (V,)

                    # 约束：禁 BOS
                    lp[self.bos_id] = NEG_INF

                    # 最小长度：太早不许 EOS
                    cur_len = len(seq) - 1
                    if cur_len < self.min_len:
                        lp[self.eos_id] = NEG_INF

                    # repeat penalty（可选）
                    if self.repeat_penalty > 0:
                        for t in set(seq):
                            lp[t] -= self.repeat_penalty

                    # topk
                    topk = int(self.beam_size)
                    idx = xp.argpartition(-lp, topk)[:topk]
                    idx = idx[xp.argsort(-lp[idx])]

                    for t in idx.tolist():
                        t = int(t)
                        cand.append((seq + [t], logp + float(lp[t]), new_cache))

                if not cand:
                    break

                # 选 beam_size 个（按 length penalty 归一化）
                scored = []
                for seq, logp, cache in cand:
                    L = max(1, len(seq) - 1)  # 不算 BOS
                    score = logp / self._len_penalty(L)
                    scored.append((score, seq, logp, cache))

                scored.sort(key=lambda x: x[0], reverse=True)
                beams = [(s[1], s[2], s[3]) for s in scored[:self.beam_size]]

                # early stop：finished 足够多且当前 beams 都不可能更优（简单版可不写）
                if len(finished) >= self.beam_size:
                    pass

            # 汇总 finished + beams
            merged = finished + [(seq, logp) for (seq, logp, _cache) in beams]
            merged.sort(key=lambda x: x[1], reverse=True)
            merged = merged[:self.beam_size]
            all_results.append(merged)

        return all_results