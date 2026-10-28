"""Neighbor-only ring collectives over the common gRPC/protobuf transport.

Receive handlers acknowledge after enqueueing, never after waiting for a peer.
This is essential: all ranks send before receiving and must not deadlock.
"""
import hmac
import threading
import time

import numpy as np

from .transport import Client, start_transport


class RingInbox:
    def __init__(self, rank, size, token, timeout, stop_event=None):
        self.rank, self.size, self.token, self.timeout = rank, size, token, timeout
        self.stop_event = stop_event
        self.condition = threading.Condition()
        self.messages, self.seen = {}, set()
        self.sequence, self.failed = 0, None

    def abort(self, message):
        with self.condition:
            self.failed = self.failed or str(message)
            self.condition.notify_all()

    def dispatch(self, metadata, tensors):
        with self.condition:
            supplied = metadata.get('token', '')
            if not isinstance(supplied, str) or not hmac.compare_digest(supplied, self.token):
                raise ValueError('Invalid ring run token')
            if metadata.get('op') != 'ring' or metadata.get('rank') != (self.rank - 1) % self.size:
                raise ValueError('Only the left neighbor may send ring messages')
            sequence, stage, round_id = metadata.get('sequence'), metadata.get('stage'), metadata.get('round')
            if type(sequence) is not int or not self.sequence <= sequence <= self.sequence + 1:
                raise ValueError('Stale or out-of-window ring collective sequence')
            if stage not in ('reduce_scatter', 'allgather', 'metadata') or type(round_id) is not int or not 0 <= round_id < self.size - 1:
                raise ValueError('Invalid ring stage/round')
            expected = (self.rank - round_id - (0 if stage == 'allgather' else 1)) % self.size
            if metadata.get('chunk') != expected:
                raise ValueError('Unexpected chunk/origin for ring round')
            key = (sequence, stage, round_id)
            if key in self.seen:
                raise ValueError('Duplicate ring message')
            if self.failed:
                raise RuntimeError(self.failed)
            self.seen.add(key)
            self.messages[key] = (metadata, tensors)
            self.condition.notify_all()
            return {'ok': True, 'sequence': sequence}, {}

    def receive(self, sequence, stage, round_id, label):
        deadline = time.perf_counter() + self.timeout
        key = (sequence, stage, round_id)
        with self.condition:
            while key not in self.messages:
                if self.failed:
                    raise RuntimeError(self.failed)
                if self.stop_event is not None and self.stop_event.is_set():
                    raise RuntimeError('Ring stopped by Launcher')
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise TimeoutError(f'Ring rank {self.rank} waiting for {key} from left neighbor')
                self.condition.wait(min(remaining, .1))
            metadata, arrays = self.messages.pop(key)
            if metadata.get('label') != label:
                raise ValueError('Ranks entered different ring collectives')
            return metadata, arrays

    def advance(self):
        with self.condition:
            if any(key[0] == self.sequence for key in self.messages):
                raise ValueError('Unconsumed ring messages')
            self.sequence += 1
            self.seen = {key for key in self.seen if key[0] >= self.sequence}


class RingCollective:
    def __init__(self, config, rank, token, stop_event=None):
        self.config, self.rank, self.size = config, rank, config.num_workers
        self.inbox = RingInbox(rank, self.size, token, config.rpc_timeout_s, stop_event)
        self.address, self.stop_server = start_transport(config, self.inbox.dispatch, self.inbox.abort)
        self.token, self.client = token, None

    def connect(self, addresses):
        self.client = Client(self.config, addresses[(self.rank + 1) % self.size], self.token, self.rank)

    def close(self):
        if self.client is not None:
            self.client.close()
        self.stop_server()

    def exchange(self, stage, round_id, chunk, label, arrays, totals, payload=None):
        started = time.perf_counter()
        response, _, timing = self.client.call('ring', tensors=arrays, sequence=self.inbox.sequence,
                                               stage=stage, round=round_id, chunk=chunk,
                                               label=label, payload=payload)
        if response['sequence'] != self.inbox.sequence:
            raise ValueError('Ring acknowledgment sequence mismatch')
        metadata, received = self.inbox.receive(self.inbox.sequence, stage, round_id, label)
        totals['communication_s'] += time.perf_counter() - started
        for key in ('sent_bytes', 'received_bytes', 'encode_s', 'decode_s'):
            totals[key] += timing[key]
        totals['ring_rpc_count'] += 1
        totals['ring_gradient_sent_bytes'] += arrays.get('gradient', np.empty(0)).nbytes
        totals['ring_statistics_sent_bytes'] += arrays.get('statistics', np.empty(0)).nbytes
        return metadata, received

    @staticmethod
    def counters():
        return dict(communication_s=0.0, sent_bytes=0, received_bytes=0, encode_s=0.0,
                    decode_s=0.0, ring_rpc_count=0, ring_gradient_sent_bytes=0,
                    ring_statistics_sent_bytes=0, aggregation_s=0.0)

    def allgather_metadata(self, payload, label):
        records = {self.rank: payload}
        current, origin = payload, self.rank
        totals = self.counters()
        for step in range(self.size - 1):
            metadata, arrays = self.exchange('metadata', step, origin, label, {}, totals, current)
            if arrays:
                raise ValueError('Unexpected tensors in metadata collective')
            origin, current = metadata['chunk'], metadata['payload']
            records[origin] = current
        self.inbox.advance()
        return [records[rank] for rank in range(self.size)], totals

    def allreduce(self, arrays, label):
        """Sum flat numeric tensors; weighting/division belongs to the caller.

        Arrays split independently, preserving float32 gradients/float64 metrics.
        Uneven and empty chunks are valid. No padding or central reduction.
        """
        started = time.perf_counter()
        for value in arrays.values():
            if value.ndim != 1 or value.dtype.kind != 'f' or not np.isfinite(value).all():
                raise ValueError('Ring allreduce expects finite flat float tensors')
        split = {key: [part.copy() for part in np.array_split(value, self.size)] for key, value in arrays.items()}
        chunks = [{key: split[key][i] for key in sorted(split)} for i in range(self.size)]
        totals = self.counters()
        totals['ring_split_s'] = time.perf_counter() - started
        for stage in ('reduce_scatter', 'allgather'):
            phase_started = time.perf_counter()
            for step in range(self.size - 1):
                send = (self.rank - step + (1 if stage == 'allgather' else 0)) % self.size
                receive = (send - 1) % self.size
                _, values = self.exchange(stage, step, send, label, chunks[send], totals)
                if values.keys() != chunks[receive].keys():
                    raise ValueError('Ring tensor names differ across ranks')
                local_started = time.perf_counter()
                for key, value in values.items():
                    target = chunks[receive][key]
                    if value.shape != target.shape or value.dtype != target.dtype:
                        raise ValueError('Ring chunk shape/dtype mismatch')
                    if stage == 'reduce_scatter':
                        target += value
                    else:
                        target[...] = value
                totals['aggregation_s'] += time.perf_counter() - local_started
            totals['ring_' + stage + '_s'] = time.perf_counter() - phase_started
        joined = time.perf_counter()
        result = {key: np.concatenate([chunk[key] for chunk in chunks]) for key in sorted(arrays)}
        if any(not np.isfinite(value).all() for value in result.values()):
            raise FloatingPointError('Non-finite ring reduction result')
        totals['ring_join_s'] = time.perf_counter() - joined
        totals['ring_collective_s'] = time.perf_counter() - started
        self.inbox.advance()
        return result, totals
