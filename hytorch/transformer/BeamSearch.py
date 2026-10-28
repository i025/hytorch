# BeamSearch.py
# Beam Search decoder for hytorch TransformerSeq2Seq
# Compatible with: logits = model(src_ids, tgt_ids)

import numpy as np
from hytorch import cuda, as_variable
import hytorch.functions as F


NEG_INF = -1e9


class BeamSearchDecoder:
    def __init__(
        self,
        model,
        beam_size=5,
        max_len=32,
        bos_id=1,
        eos_id=2,
        pad_id=0,
        min_len=1,                 # 至少生成多少个 token（不含 BOS）
        length_penalty_alpha=0.6,  # GNMT length penalty
        repeat_penalty=1.0,        # 重复惩罚强度
    ):
        self.model = model
        self.beam_size = int(beam_size)
        self.max_len = int(max_len)
        self.bos_id = int(bos_id)
        self.eos_id = int(eos_id)
        self.pad_id = int(pad_id)
        self.min_len = int(min_len)
        self.length_penalty_alpha = float(length_penalty_alpha)
        self.repeat_penalty = float(repeat_penalty)

    def _length_penalty(self, length: int) -> float:
        # GNMT: ((5 + len) / 6) ^ alpha
        if self.length_penalty_alpha <= 0:
            return 1.0
        return float(((5.0 + float(length)) / 6.0) ** self.length_penalty_alpha)

    def _score(self, seq, logp: float) -> float:
        # normalize by length penalty
        lp = self._length_penalty(len(seq))
        return float(logp) / lp

    def decode(self, src_ids):
        """
        src_ids: (N, S) int64 (numpy / cupy)

        return:
            results: List[List[(seq, score)]]
                     results[b][k] = (token_id_list, normalized_score)
        """
        xp = cuda.get_array_module(src_ids)
        batch_size = int(src_ids.shape[0])

        all_results = []

        for b in range(batch_size):
            beams = [([self.bos_id], 0.0)]  # (seq, logp)
            finished = []

            for _ in range(self.max_len):
                new_beams = []

                for seq, logp in beams:
                    # ended beam -> keep
                    if seq[-1] == self.eos_id:
                        finished.append((seq, logp))
                        continue

                    # build tgt
                    tgt = xp.asarray(seq, dtype=xp.int64)[None, :]
                    tgt = as_variable(tgt)

                    # ⭐ call your model forward: logits = model(src, tgt)
                    logits = self.model(src_ids[b:b+1], tgt)
                    last_logits = logits.data[0, -1]  # (V,)

                    # log-softmax
                    log_probs = F.log_softmax(as_variable(last_logits), axis=-1).data

                    # make writable copy (safe for numpy/cupy)
                    log_probs = xp.array(log_probs, dtype=xp.float32)

                    # 1) forbid BOS in the middle
                    log_probs[self.bos_id] = xp.float32(NEG_INF)

                    # 2) min length: forbid early EOS
                    cur_len = len(seq) - 1  # excluding BOS
                    if cur_len < self.min_len:
                        log_probs[self.eos_id] = xp.float32(NEG_INF)

                    # 3) repeat penalty
                    if self.repeat_penalty > 0:
                        for t in set(seq):
                            log_probs[int(t)] -= xp.float32(self.repeat_penalty)

                    # top-k
                    topk = xp.argsort(log_probs)[-self.beam_size:][::-1]
                    # 4) vocab constraint: only allow digits(3..12) + EOS
                    #    forbid PAD/BOS/PLUS/EQUAL etc.
                    allowed = set([self.eos_id] + list(range(3, 13)))  # 2 + (3..12)

                    V = int(log_probs.shape[0])
                    for tid in range(V):
                        if tid not in allowed:
                            log_probs[tid] = xp.float32(NEG_INF)
                    # expand
                    for wid in topk.tolist() if hasattr(topk, "tolist") else list(topk):
                        wid = int(wid)
                        new_seq = seq + [wid]
                        new_logp = float(logp) + float(log_probs[wid])
                        new_beams.append((new_seq, new_logp))

                if not new_beams:
                    break

                # keep best beams by normalized score
                new_beams.sort(key=lambda x: self._score(x[0], x[1]), reverse=True)
                beams = new_beams[: self.beam_size]

                # early stop if best ended
                if beams[0][0][-1] == self.eos_id:
                    break

            # finalize: finished if any else beams
            finished.extend(beams)
            finished.sort(key=lambda x: self._score(x[0], x[1]), reverse=True)

            topk_results = []
            for seq, logp in finished[: self.beam_size]:
                topk_results.append((seq, self._score(seq, logp)))

            all_results.append(topk_results)

        # ✅ critical: must return list
        return all_results