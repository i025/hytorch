import math
import queue
import threading
import traceback

import numpy as np

from hytorch import cuda


# =============================================================================
# Sync DataLoader (original implementation, unchanged)
# =============================================================================
class _SyncDataLoader:
    def __init__(self, dataset, batch_size, shuffle=True, gpu=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_size = len(dataset)
        self.max_iter = math.ceil(self.data_size / self.batch_size)
        self.gpu = gpu
        self.reset()

    def reset(self):
        self.iteration = 0
        if self.shuffle:
            self.index = np.random.permutation(len(self.dataset))
        else:
            self.index = np.arange(len(self.dataset))

    def __iter__(self):
        return self

    def __next__(self):
        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration

        i, batch_size = self.iteration, self.batch_size
        batch_index = self.index[i * batch_size:(i + 1) * batch_size]
        batch = [self.dataset[i] for i in batch_index]

        xp = cuda.cupy if self.gpu else np
        x = xp.array([example[0] for example in batch])
        t = xp.array([example[1] for example in batch])

        self.iteration += 1
        return x, t

    def next(self):
        return self.__next__()

    def to_cpu(self):
        self.gpu = False

    def to_gpu(self):
        self.gpu = True


# =============================================================================
# Async DataLoader — producer-consumer pattern
# =============================================================================
_SENTINEL = object()


class _WorkerError:
    def __init__(self, exc: Exception, tb_str: str):
        self.exc = exc
        self.tb_str = tb_str


class _DatasetWorker(threading.Thread):
    def __init__(self, dataset, index_queue: queue.Queue, result_queue: queue.Queue):
        super().__init__(daemon=True)
        self.dataset = dataset
        self.index_queue = index_queue
        self.result_queue = result_queue

    def run(self):
        while True:
            task = self.index_queue.get()
            if task is _SENTINEL:
                self.result_queue.put(_SENTINEL)
                return
            seq, batch_index = task
            try:
                batch = [self.dataset[i] for i in batch_index]
                x = np.array([b[0] for b in batch])
                t = np.array([b[1] for b in batch])
                self.result_queue.put((seq, x, t))
            except Exception as exc:
                self.result_queue.put(_WorkerError(exc, traceback.format_exc()))
                return


class AsyncDataLoader:
    def __init__(self, dataset, batch_size, shuffle=True,
                 num_workers=2, prefetch_factor=2, gpu=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.gpu = gpu
        self.data_size = len(dataset)
        self.max_iter = math.ceil(self.data_size / self.batch_size)

        self._index_queue: queue.Queue = None
        self._result_queue: queue.Queue = None
        self._workers: list = []
        self._active_workers = 0
        self._started = False
        self._next_seq = 0          # next sequence number to emit to caller
        self._pending = {}          # seq -> (x, t) buffer for out-of-order results

    def to_gpu(self):
        self.gpu = True

    def to_cpu(self):
        self.gpu = False

    def __iter__(self):
        self._shutdown_workers()
        self._start_epoch()
        return self

    def __next__(self):
        # Emit batches in original sequence order using a pending buffer.
        while self._next_seq not in self._pending:
            try:
                item = self._result_queue.get(timeout=60)
            except queue.Empty:
                raise RuntimeError(
                    "DataLoader worker timed out after 60s — possible deadlock."
                )

            if item is _SENTINEL:
                self._active_workers -= 1
                if self._active_workers == 0:
                    self._started = False
                    raise StopIteration
                continue

            if isinstance(item, _WorkerError):
                self._shutdown_workers()
                raise RuntimeError(
                    f"DataLoader worker crashed:\n{item.tb_str}"
                ) from item.exc

            seq, x_np, t_np = item
            self._pending[seq] = (x_np, t_np)

        x_np, t_np = self._pending.pop(self._next_seq)
        self._next_seq += 1

        if self.gpu and cuda.gpu_enable:
            x = cuda.cupy.asarray(x_np)
            t = cuda.cupy.asarray(t_np)
        else:
            x, t = x_np, t_np
        return x, t

    def next(self):
        return self.__next__()

    def _start_epoch(self):
        if self.shuffle:
            index = np.random.permutation(self.data_size)
        else:
            index = np.arange(self.data_size)

        self._index_queue = queue.Queue()
        self._result_queue = queue.Queue(maxsize=self.prefetch_factor * self.num_workers)
        self._next_seq = 0
        self._pending = {}

        for i in range(self.max_iter):
            batch_index = index[i * self.batch_size:(i + 1) * self.batch_size]
            self._index_queue.put((i, batch_index))   # tagged with sequence number

        self._workers = []
        for _ in range(self.num_workers):
            self._index_queue.put(_SENTINEL)
            w = _DatasetWorker(self.dataset, self._index_queue, self._result_queue)
            w.start()
            self._workers.append(w)

        self._active_workers = self.num_workers
        self._started = True

    def _shutdown_workers(self):
        if not self._started:
            return
        # drain index_queue so workers don't block on put
        try:
            while True:
                self._index_queue.get_nowait()
        except queue.Empty:
            pass
        for _ in self._workers:
            self._index_queue.put(_SENTINEL)
        for w in self._workers:
            w.join(timeout=5.0)
        self._workers = []
        self._started = False

    def __del__(self):
        self._shutdown_workers()


# =============================================================================
# Public DataLoader — thin wrapper, 100% backward compatible
# =============================================================================
class DataLoader:
    """
    Drop-in replacement for the original DataLoader.
    num_workers=0 (default): synchronous, identical to the original behavior.
    num_workers>0: async producer-consumer with background threads.
    """

    def __init__(self, dataset, batch_size, shuffle=True, gpu=False,
                 num_workers=0, prefetch_factor=2):
        if num_workers == 0:
            self._impl = _SyncDataLoader(dataset, batch_size, shuffle, gpu)
        else:
            self._impl = AsyncDataLoader(
                dataset, batch_size, shuffle, num_workers, prefetch_factor, gpu
            )

    # --- iterator protocol ---
    def __iter__(self):
        return iter(self._impl)

    def __next__(self):
        return next(self._impl)

    def next(self):
        return self._impl.next()

    # --- device control ---
    def to_gpu(self):
        self._impl.to_gpu()

    def to_cpu(self):
        self._impl.to_cpu()

    # --- convenience properties ---
    @property
    def max_iter(self):
        return self._impl.max_iter

    @property
    def dataset(self):
        return self._impl.dataset

    @property
    def gpu(self):
        return self._impl.gpu

    @property
    def batch_size(self):
        return self._impl.batch_size
