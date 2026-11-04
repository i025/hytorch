"""Ordered preprocessing must preserve the exact training protocol, including repeats."""
import time

import numpy as np
import pytest

from hytorch.dataloaders import DataLoader


class Dataset:
    def __len__(self):
        return 13

    def __getitem__(self, index):
        # Force out-of-order worker completion.
        if index == 9:
            time.sleep(.01)
        return np.array([index], np.float32), np.array([-index], np.float32)


@pytest.mark.parametrize('workers', [0, 1, 2, 4])
def test_explicit_order_and_partial_batch(workers):
    indices = np.array([9, 2, 9, 1, 0, 12, 4])
    loader = DataLoader(Dataset(), 3, shuffle=False, num_workers=workers,
                        indices=indices)
    for _ in range(2):
        batches = list(loader)
        assert [len(x) for x, _ in batches] == [3, 3, 1]
        x, t = (np.concatenate([b[i] for b in batches]).ravel() for i in (0, 1))
        np.testing.assert_array_equal(x, indices)
        np.testing.assert_array_equal(t, -indices)


@pytest.mark.parametrize('workers', [0, 2])
def test_existing_full_dataset_default(workers):
    values = list(DataLoader(Dataset(), 4, shuffle=False, num_workers=workers))
    np.testing.assert_array_equal(np.concatenate([x for x, _ in values]).ravel(),
                                  np.arange(13))


@pytest.mark.parametrize('workers', [0, 2])
def test_shuffle_only_selected_indices(workers):
    np.random.seed(42)
    values = list(DataLoader(Dataset(), 2, num_workers=workers, indices=[8, 1, 5, 8]))
    np.testing.assert_array_equal(np.sort(np.concatenate([x for x, _ in values]).ravel()),
                                  [1, 5, 8, 8])
