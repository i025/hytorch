import csv
from datetime import datetime, timedelta

import numpy as np
import pytest
from PIL import Image

from donkeycar_hy.udacity_dataset import UdacityDataset, preprocess_udacity


def make_data(path, count=30):
    (path/'IMG').mkdir()
    records=[]
    for i in range(count):
        name='center_'+(datetime(2016,12,1)+timedelta(seconds=i)).strftime('%Y_%m_%d_%H_%M_%S_%f')+'.jpg'
        Image.new('RGB',(320,160),(255,0,128)).save(path/'IMG'/name)
        records.append(dict(center=' IMG/'+name, left='unused',right='unused',steering=i/100))
    with (path/'driving_log.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    return UdacityDataset(path/'driving_log.csv')


def test_csv_labels_and_shape(tmp_path):
    ds=make_data(tmp_path)
    x,t=ds[7]
    assert x.shape==(3,66,200) and x.dtype==np.float32 and x.flags.c_contiguous
    assert t.shape==(1,) and t.dtype==np.float32 and t[0]==np.float32(.07)
    assert -1<=x.min()<=x.max()<=1
    assert x[0].mean()>.95 and x[1].mean()<-.95
    np.testing.assert_array_equal(x,preprocess_udacity(ds.paths[7]))


def test_crop_resize_channels(tmp_path):
    pixels=np.zeros((160,320,3),np.uint8)
    pixels[:70]=255;pixels[135:]=255;pixels[70:135,:,2]=255
    path=tmp_path/'frame.png';Image.fromarray(pixels).save(path)
    value=preprocess_udacity(path)
    np.testing.assert_array_equal(value[0],-np.ones((66,200),np.float32))
    np.testing.assert_array_equal(value[2],np.ones((66,200),np.float32))


def test_temporal_split(tmp_path):
    ds=make_data(tmp_path)
    train,val,excluded=ds.split_indices(gap_seconds=5)
    assert len(train)==19 and len(val)==6 and len(excluded)==5
    assert set(train).isdisjoint(val) and set(excluded).isdisjoint(train)
    assert (min(ds.times[i] for i in val)-max(ds.times[i] for i in train)).total_seconds()>5
    for a,b in zip(ds.split_indices(),ds.split_indices()):np.testing.assert_array_equal(a,b)


def test_portable_paths_and_missing_file(tmp_path):
    ds=make_data(tmp_path)
    assert ds.image_path(r'C:\old\IMG\x.jpg')==tmp_path/'IMG/x.jpg'
    assert ds.image_path('/old/IMG/x.jpg')==tmp_path/'IMG/x.jpg'
    ds.paths[0].unlink()
    with pytest.raises(FileNotFoundError):ds[0]


def test_explicit_stream_keeps_pixels_and_closes_file(tmp_path, monkeypatch):
    original = make_data(tmp_path)
    expected = original[7]
    dataset = UdacityDataset(original.csv_path, open_stream=True)
    opened = []
    image_open = Image.open

    def record_open(stream, *args, **kwargs):
        opened.append(stream)
        return image_open(stream, *args, **kwargs)

    monkeypatch.setattr(Image, 'open', record_open)
    for _ in range(3):
        actual = dataset[7]
        for value, reference in zip(actual, expected):
            np.testing.assert_array_equal(value, reference)
    assert all(stream.closed for stream in opened)
    assert original.open_stream is False
