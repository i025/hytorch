"""CPU-only preprocessing, CSV provenance and temporal split contracts.

Real-data tests use HYTORCH_UDACITY_DATA or ~/datasets/udacity/data. They skip
only when the default dataset is absent; an explicit missing path is an error.
The real dataset is never written to. No model forward/backward is run.
"""

import csv
from datetime import datetime, timedelta
from io import BytesIO
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from donkeycar_hy.preprocess import load_rgb_image, preprocess_rgb
from donkeycar_hy.udacity_dataset import UdacityDataset


def write_csv(root, rows, header=("center", "steering")):
    with (root / "driving_log.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


@pytest.fixture
def small_data(tmp_path):
    images = tmp_path / "IMG"
    images.mkdir()
    rows = []
    for index in range(10):
        stamp = (datetime(2016, 12, 1, 13, 30) + timedelta(seconds=index))
        name = "center_" + stamp.strftime("%Y_%m_%d_%H_%M_%S_%f")[:-3] + ".jpg"
        Image.new("RGB", (320, 160), (index * 20, 10, 30)).save(images / name)
        rows.append([" IMG/" + name + " ", " {} ".format((index - 5) / 10)])
    write_csv(tmp_path, rows, header=(" center ", " steering "))
    return tmp_path, rows


@pytest.fixture(scope="module")
def real_dataset():
    configured = os.environ.get("HYTORCH_UDACITY_DATA")
    root = Path(configured).expanduser() if configured else Path.home() / "datasets/udacity/data"
    if not (root / "driving_log.csv").is_file():
        if configured:
            pytest.fail("HYTORCH_UDACITY_DATA has no driving_log.csv: {}".format(root))
        pytest.skip("Official Udacity data absent; set HYTORCH_UDACITY_DATA to validate it")
    return UdacityDataset(root)


def test_preprocess_crop_channels_range_and_input_unchanged():
    pixels = np.zeros((160, 320, 3), dtype=np.uint8)
    pixels[:40] = [0, 255, 0]  # All of this must disappear.
    pixels[40:] = [255, 0, 128]
    before = pixels.copy()
    result = preprocess_rgb(pixels)
    assert result.shape == (3, 66, 200)
    assert result.dtype == np.float32 and result.flags.c_contiguous
    assert np.isfinite(result).all() and result.min() == -1 and result.max() == 1
    expected = np.array([1, -1, 128 / 127.5 - 1], dtype=np.float32)
    np.testing.assert_allclose(result, np.broadcast_to(expected[:, None, None], result.shape), atol=1e-7)
    np.testing.assert_array_equal(pixels, before)


def test_preprocess_keeps_bottom_and_uses_pillow_bilinear():
    pixels = np.random.default_rng(9).integers(0, 256, (160, 320, 3), dtype=np.uint8)
    image = Image.fromarray(pixels)
    expected = np.asarray(image.crop((0, 40, 320, 160)).resize(
        (200, 66), Image.Resampling.BILINEAR), dtype=np.float32) / np.float32(127.5) - np.float32(1)
    np.testing.assert_array_equal(preprocess_rgb(image), expected.transpose(2, 0, 1))


@pytest.mark.parametrize("image,error", [
    (np.zeros((160, 320, 3), dtype=np.float32), TypeError),
    (np.zeros((160, 320), dtype=np.uint8), ValueError),
    (np.zeros((160, 320, 4), dtype=np.uint8), ValueError),
    (np.zeros((40, 320, 3), dtype=np.uint8), ValueError),
    (Image.new("L", (320, 160)), ValueError),
    ("image.jpg", TypeError),
])
def test_preprocess_rejects_ambiguous_input(image, error):
    with pytest.raises(error):
        preprocess_rgb(image)


def test_csv_whitespace_paths_labels_provenance_and_cwd(small_data, monkeypatch, tmp_path):
    root, rows = small_data
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    dataset = UdacityDataset(root)
    assert len(dataset) == 10 and dataset.indices == tuple(range(10))
    assert dataset.records[0].csv_line == 2
    assert dataset.records[-1].csv_line == 11
    assert dataset.records[0].image_path == (root / rows[0][0].strip()).resolve()
    assert dataset.records[1].timestamp == datetime(2016, 12, 1, 13, 30, 1)
    assert dataset.label.dtype == np.float32
    _, label = dataset[0]
    np.testing.assert_array_equal(label, np.array([-0.5], dtype=np.float32))


def test_absolute_path_and_blank_csv_lines_keep_physical_line(small_data):
    root, rows = small_data
    rows[0][0] = str((root / rows[0][0].strip()).resolve())
    write_csv(root, [rows[0], [], rows[1]])
    dataset = UdacityDataset(root)
    assert dataset.indices == (0, 1)
    assert [record.csv_line for record in dataset.records] == [2, 4]
    assert dataset.records[0].image_path.is_absolute()


@pytest.mark.parametrize("bad", ["nan", "inf", "-inf", "1e100", "not-a-number"])
def test_invalid_steering_rejected(small_data, bad):
    root, rows = small_data
    rows[0][1] = bad
    write_csv(root, rows)
    with pytest.raises(ValueError, match="line 2.*steering"):
        UdacityDataset(root)


def test_missing_image_rejected(small_data):
    root, rows = small_data
    (root / rows[0][0].strip()).unlink()
    with pytest.raises(FileNotFoundError, match="line 2.*missing"):
        UdacityDataset(root)


@pytest.mark.parametrize("header,rows,match", [
    (["center"], [["anything.jpg"]], "column names"),
    (["center", "center", "steering"], [], "column names"),
    (["center", "steering"], [], "no driving records"),
    (["center", "steering"], [["", "0"]], "empty center"),
    (["center", "steering"], [["file.jpg", "0", "extra"]], "field count"),
])
def test_malformed_csv_rejected(tmp_path, header, rows, match):
    write_csv(tmp_path, rows, header)
    with pytest.raises(ValueError, match=match):
        UdacityDataset(tmp_path)


def test_duplicate_image_rejected(small_data):
    root, rows = small_data
    write_csv(root, [rows[0], rows[0]])
    with pytest.raises(ValueError, match="duplicate center"):
        UdacityDataset(root)


def test_unparseable_timestamp_rejected(small_data):
    root, rows = small_data
    image = root / rows[0][0].strip()
    image.rename(root / "IMG/center_unknown.jpg")
    rows[0][0] = "IMG/center_unknown.jpg"
    write_csv(root, rows)
    with pytest.raises(ValueError, match="line 2.*timestamp"):
        UdacityDataset(root)


def test_temporal_split_indices_embargo_and_reproducibility(small_data, tmp_path):
    root, _ = small_data
    dataset = UdacityDataset(root)
    split = dataset.split_temporal(val_ratio=0.2, gap_seconds=2)
    assert split.train.indices == (0, 1, 2, 3, 4, 5)
    assert split.excluded.indices == (6, 7)  # Exactly two seconds before val is excluded too.
    assert split.validation.indices == (8, 9)
    assert split.to_dict() == dataset.split_temporal(0.2, 2).to_dict()
    assert split.train.records[0] is dataset.records[0]
    assert split.validation.records[0].csv_line == 10
    assert dataset.indices == tuple(range(10))
    manifest = tmp_path / "split.json"
    split.save_manifest(manifest)
    assert json.loads(manifest.read_text()) == split.to_dict()
    with pytest.raises(FileExistsError):
        split.save_manifest(manifest)


def test_zero_gap_and_empty_excluded_statistics(small_data):
    dataset = UdacityDataset(small_data[0])
    split = dataset.split_temporal(0.1, 0)
    assert split.train.indices == tuple(range(9))
    assert split.validation.indices == (9,)
    assert not split.excluded.indices
    stats = split.excluded.statistics()
    assert stats["count"] == 0 and stats["steering_min"] is None
    json.dumps(split.to_dict(), allow_nan=False)


@pytest.mark.parametrize("ratio,gap", [
    (0, 0), (1, 0), (-0.1, 0), (float("nan"), 0), (float("inf"), 0),
    (0.2, -1), (0.2, float("nan")), (0.2, float("inf")), (0.2, 8),
    (0.99, 0),
])
def test_invalid_or_empty_splits_rejected(small_data, ratio, gap):
    with pytest.raises(ValueError):
        UdacityDataset(small_data[0]).split_temporal(ratio, gap)


def test_single_record_cannot_split(small_data):
    root, rows = small_data
    write_csv(root, rows[:1])
    with pytest.raises(ValueError, match="training records"):
        UdacityDataset(root).split_temporal()


def test_nonchronological_csv_rejected_at_split(small_data):
    root, rows = small_data
    rows[0], rows[1] = rows[1], rows[0]
    write_csv(root, rows)
    dataset = UdacityDataset(root)
    with pytest.raises(ValueError, match="strictly increasing"):
        dataset.split_temporal()


def test_real_csv_records_paths_and_label_statistics(real_dataset):
    assert len(real_dataset) == 8036
    assert real_dataset.indices == tuple(range(8036))
    assert [record.csv_line for record in real_dataset.records] == list(range(2, 8038))
    assert all(record.image_path.is_file() for record in real_dataset.records)
    assert len({record.image_path for record in real_dataset.records}) == 8036
    labels = real_dataset.label
    assert labels.shape == (8036,) and np.isfinite(labels).all()
    assert labels.min() == pytest.approx(-0.9426954)
    assert labels.max() == 1
    assert np.mean(labels == 0) == pytest.approx(0.5426829268292683)


def test_real_all_images_and_labels(real_dataset):
    # Decode all 8,036 center images on CPU, not just a synthetic fixture.
    for index in range(len(real_dataset)):
        image, label = real_dataset[index]
        assert image.shape == (3, 66, 200), index
        assert image.dtype == np.float32 and image.flags.c_contiguous, index
        assert np.isfinite(image).all() and -1 <= image.min() <= image.max() <= 1, index
        assert label.shape == (1,) and label.dtype == np.float32, index
        assert np.isfinite(label).all(), index
        assert label[0] == np.float32(real_dataset.records[index].steering), index


def test_real_file_and_telemetry_bytes_preprocess_identically(real_dataset):
    record = real_dataset.records[len(real_dataset) // 2]
    from_file = preprocess_rgb(load_rgb_image(record.image_path))
    # Same original JPEG bytes: no lossy re-encoding and no simulator process.
    with Image.open(BytesIO(record.image_path.read_bytes())) as image:
        from_telemetry = preprocess_rgb(image.convert("RGB"))
    np.testing.assert_array_equal(from_file, from_telemetry)


def test_real_legacy_pillow_preprocess_compatible(real_dataset):
    from donkeycar_hy.tub_dataset import _preprocess

    record = real_dataset.records[0]
    np.testing.assert_array_equal(preprocess_rgb(load_rgb_image(record.image_path)),
                                  _preprocess(str(record.image_path)))


def test_real_temporal_split_disjoint_complete_and_reproducible(real_dataset):
    split = real_dataset.split_temporal()
    assert len(split.validation) == math.ceil(8036 * 0.1) == 804
    groups = [set(getattr(split, name).indices) for name in ("train", "validation", "excluded")]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    assert set.union(*groups) == set(real_dataset.indices)
    assert tuple(index for part in (split.train, split.excluded, split.validation)
                 for index in part.indices) == real_dataset.indices
    cutoff = split.validation_start - timedelta(seconds=split.gap_seconds)
    assert all(record.timestamp < cutoff for record in split.train.records)
    assert all(cutoff <= record.timestamp < split.validation_start for record in split.excluded.records)
    assert len(split.excluded) > 0
    assert (split.validation_start - split.train.records[-1].timestamp).total_seconds() > split.gap_seconds
    assert split.to_dict() == real_dataset.split_temporal().to_dict()


@pytest.mark.parametrize("workers", [0, 2])
def test_real_dataloader_batches_order_and_tail(real_dataset, workers):
    from hytorch.dataloaders import DataLoader

    subset = real_dataset.split_temporal(val_ratio=4.5 / len(real_dataset), gap_seconds=0).validation
    assert len(subset) == 5
    loader = DataLoader(subset, batch_size=2, shuffle=False, num_workers=workers)
    for _ in range(2):  # A fresh iterator for each epoch, including async workers.
        batches = list(loader)
        assert [len(image) for image, _ in batches] == [2, 2, 1]
        for images, labels in batches:
            assert images.shape == (len(images), 3, 66, 200) and labels.shape == (len(images), 1)
            assert images.dtype == labels.dtype == np.float32
            assert images.flags.c_contiguous and labels.flags.c_contiguous
        np.testing.assert_array_equal(np.concatenate([labels for _, labels in batches])[:, 0], subset.label)
