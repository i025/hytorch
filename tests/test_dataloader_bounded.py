"""A single bounded window covers queued, running and buffered batches."""
import threading
import time

import numpy as np
import pytest

from hytorch.dataloaders import AsyncDataLoader


class Dataset:
    def __len__(self):
        return 100

    def __getitem__(self, index):
        return np.array([index], np.float32), np.array([-index], np.float32)


@pytest.mark.parametrize('workers,prefetch', [(2, 1), (2, 2), (4, 1), (4, 2)])
def test_slow_first_batch_cannot_exhaust_epoch(workers, prefetch):
    entered, release = threading.Event(), threading.Event()
    called = []
    class Paused(Dataset):
        def __getitem__(self, index):
            called.append(index)
            if index == 0:
                entered.set()
                assert release.wait(3)
            return super().__getitem__(index)

    loader = AsyncDataLoader(Paused(), 1, shuffle=False, num_workers=workers,
                             prefetch_factor=prefetch)
    iter(loader)
    result = []
    consumer = threading.Thread(target=lambda: result.append(next(loader)))
    consumer.start()
    try:
        assert entered.wait(2)
        time.sleep(.1)
        assert len(called) <= workers * prefetch
        assert len(loader._pending) <= workers * prefetch - 1
    finally:
        release.set()
        consumer.join(2)
        loader.close()
    assert not consumer.is_alive()
    np.testing.assert_array_equal(result[0][0], [[0]])


@pytest.mark.parametrize('workers,prefetch', [(1, 1), (2, 1), (2, 2), (4, 1), (4, 4)])
def test_window_replenishes_without_changing_order(workers, prefetch):
    indices = [9, 2, 9, 1, 0, 12, 4, 1, 7, 3, 5]
    loader = AsyncDataLoader(Dataset(), 2, shuffle=False, indices=indices,
                             num_workers=workers, prefetch_factor=prefetch)
    for _ in range(3):
        iter(loader)
        values = []
        for step in range(loader.max_iter):
            x, t = next(loader)
            values.extend(x.ravel())
            np.testing.assert_array_equal(t, -x)
            assert 0 <= loader._send_seq - loader._next_seq <= workers * prefetch
            assert len(loader._pending) <= workers * prefetch - 1
        assert next(loader, None) is None
        assert not loader._workers
        np.testing.assert_array_equal(values, indices)


@pytest.mark.parametrize('indices', [[], [5], [5, 2]])
def test_idle_workers_stop_at_short_epoch(indices):
    loader = AsyncDataLoader(Dataset(), 2, shuffle=False, indices=indices,
                             num_workers=4, prefetch_factor=1)
    values = list(loader)
    actual = np.concatenate([x.ravel() for x, _ in values]) if values else []
    np.testing.assert_array_equal(actual, indices)
    assert not loader._workers


def test_early_restart_discards_previous_window():
    loader = AsyncDataLoader(Dataset(), 1, shuffle=False, num_workers=4,
                             prefetch_factor=1)
    iter(loader)
    old = list(loader._workers)
    assert next(loader)[0][0, 0] == 0
    iter(loader)
    assert not any(w.is_alive() for w in old)
    assert loader._send_seq == 4 and loader._next_seq == 0
    assert next(loader)[0][0, 0] == 0
    loader.close()
    assert not loader._workers and not loader._pending


@pytest.mark.parametrize('prefetch', [0, -1])
def test_prefetch_window_must_be_positive(prefetch):
    with pytest.raises(ValueError, match='prefetch_factor must be positive'):
        AsyncDataLoader(Dataset(), 1, prefetch_factor=prefetch)
