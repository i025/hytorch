"""CPU regressions for bounded queues and epoch worker ownership."""
import gc
import time
import weakref

import numpy as np
import pytest

from hytorch.dataloaders import AsyncDataLoader, DataLoader


class Dataset:
    def __len__(self):
        return 100

    def __getitem__(self, index):
        return np.array([index], np.float32), np.array([-index], np.float32)


def fill_queue(loader):
    deadline = time.monotonic() + 2
    while not loader._result_queue.full() and time.monotonic() < deadline:
        time.sleep(.001)
    assert loader._result_queue.full()


@pytest.mark.parametrize('workers', [1, 2, 4])
def test_close_full_queue_joins_all_workers(workers):
    loader = AsyncDataLoader(Dataset(), 1, shuffle=False, num_workers=workers,
                             prefetch_factor=1)
    iter(loader)
    threads = list(loader._workers)
    fill_queue(loader)
    loader.close()
    assert not any(w.is_alive() for w in threads)
    assert not loader._workers and not loader._pending
    assert loader._result_queue is None
    loader.close()
    assert next(loader, None) is None


def test_restart_early_epoch_joins_previous_workers():
    loader = AsyncDataLoader(Dataset(), 3, shuffle=False, num_workers=2,
                             prefetch_factor=1, indices=[9, 2, 9, 1, 0, 12, 4])
    iter(loader)
    old = list(loader._workers)
    fill_queue(loader)
    np.testing.assert_array_equal(next(loader)[0].ravel(), [9, 2, 9])
    values = list(loader)
    assert not any(w.is_alive() for w in old)
    np.testing.assert_array_equal(np.concatenate([x for x, _ in values]).ravel(),
                                  [9, 2, 9, 1, 0, 12, 4])
    assert not loader._workers


@pytest.mark.parametrize('workers', [0, 2, 4])
def test_context_manager_closes_after_break(workers):
    loader = DataLoader(Dataset(), 1, num_workers=workers, prefetch_factor=1)
    with loader:
        for x, t in loader:
            threads = list(loader._impl._workers) if workers else []
            break
    assert not any(w.is_alive() for w in threads)
    loader.close()


def test_worker_exception_closes_full_queue():
    class Broken(Dataset):
        def __getitem__(self, index):
            if index == 0:
                raise ValueError('bad sample 0')
            return super().__getitem__(index)

    loader = AsyncDataLoader(Broken(), 1, shuffle=False, num_workers=2,
                             prefetch_factor=1)
    iter(loader)
    threads = list(loader._workers)
    fill_queue(loader)
    with pytest.raises(RuntimeError, match='bad sample 0') as error:
        next(loader)
    assert isinstance(error.value.__cause__, ValueError)
    assert not any(w.is_alive() for w in threads)
    assert not loader._started


def test_destructor_closes_full_queue():
    loader = AsyncDataLoader(Dataset(), 1, num_workers=2, prefetch_factor=1)
    iter(loader)
    threads = list(loader._workers)
    fill_queue(loader)
    reference = weakref.ref(loader)
    del loader
    gc.collect()
    assert reference() is None
    assert not any(w.is_alive() for w in threads)


@pytest.mark.parametrize('indices', [[], [8], [2, 2, 5, 0, 9]])
def test_exhaustion_is_repeatable_and_stops_threads(indices):
    loader = AsyncDataLoader(Dataset(), 2, shuffle=False, num_workers=4,
                             prefetch_factor=1, indices=indices)
    for _ in range(3):
        values = list(loader)
        actual = np.concatenate([x.ravel() for x, _ in values]) if values else []
        np.testing.assert_array_equal(actual, indices)
        assert next(loader, None) is None
        assert not loader._workers


def test_close_waits_for_inflight_sample():
    import threading
    started, release = threading.Event(), threading.Event()

    class Paused(Dataset):
        def __getitem__(self, index):
            started.set()
            assert release.wait(2)
            return super().__getitem__(index)

    loader = AsyncDataLoader(Paused(), 1, num_workers=1)
    iter(loader)
    assert started.wait(2)
    threads = list(loader._workers)
    closer = threading.Thread(target=loader.close)
    closer.start()
    release.set()
    closer.join(2)
    assert not closer.is_alive()
    assert not any(w.is_alive() for w in threads)
