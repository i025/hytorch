"""Inspect every CSV image, fix temporal splits, and cache real throughput inputs."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from donkeycar_hy.udacity_dataset import UdacityDataset,PREPROCESS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    ds=UdacityDataset(a.csv);train,val,purge=ds.split_indices()
    split={int(i):name for name,indices in [('train',train),('validation',val),('purged',purge)] for i in indices}
    missing=[];modes={};sizes={};manifest=[]
    for i,row in enumerate(ds.records):
        for camera in ('center','left','right'):
            path=ds.image_path(row[camera])
            if not path.is_file():missing.append(dict(row=i,camera=camera,path=str(path)));continue
            with Image.open(path) as image:
                key=str(image.size);sizes[key]=sizes.get(key,0)+1;modes[image.mode]=modes.get(image.mode,0)+1
                image.verify()
        manifest.append(dict(row=i,center=ds.paths[i].name,timestamp=ds.times[i].isoformat(),
            steering=float(ds.labels[i]),split=split[i],bytes=ds.paths[i].stat().st_size if ds.paths[i].exists() else None))
    report=dict(status='PASSED' if not missing else 'FAILED',csv=str(a.csv.resolve()),rows=len(ds),
        actual_jpeg_files=len(list((a.csv.parent/'IMG').glob('*.jpg'))),missing=missing,image_modes=modes,image_sizes=sizes,
        unique_center_files=len(set(ds.paths)),steering_min=float(ds.labels.min()),steering_max=float(ds.labels.max()),
        steering_mean=float(ds.labels.mean()),steering_std=float(ds.labels.std()),zero_ratio=float(np.mean(ds.labels==0)),
        train=len(train),validation=len(val),purged=len(purge),gap_seconds=10,
        actual_boundary_gap_seconds=(min(ds.times[i] for i in val)-max(ds.times[i] for i in train)).total_seconds(),
        preprocessing=PREPROCESS,provenance='Existing downloaded Udacity sample; no external provenance or checksum verification performed')
    (a.output/'dataset.json').write_text(json.dumps(report,indent=2)+'\n')
    with (a.output/'manifest.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(manifest[0]));w.writeheader();w.writerows(manifest)
    assert not missing,'Missing files saved in dataset.json'
    selection=np.random.default_rng(42).permutation(train)[:1024]
    xs=[];ts=[];times=[]
    for index in selection:
        start=time.perf_counter();x,t=ds[int(index)];xs.append(x);ts.append(t)
        times.append(dict(row=int(index),read_preprocess_s=time.perf_counter()-start))
    np.savez(a.output/'throughput-inputs.npz',x=np.stack(xs),t=np.stack(ts),indices=selection)
    with (a.output/'preprocess-times.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(times[0]));w.writeheader();w.writerows(times)
    config=dict(csv=str(a.csv.resolve()),train=train.tolist(),validation=val.tolist(),purged=purge.tolist(),
        throughput=selection.tolist(),epoch_order=[np.random.default_rng(42+e).permutation(train).tolist() for e in range(1,4)],
        seed=42,epochs=3,batch=4,lr=.001,dropout=0,tf32=False,no_fingerprints=True,
        data_kind='UDACITY_REAL_CENTER',resource_condition='SHARED_GPU',preprocessing=PREPROCESS)
    (a.output/'protocol.json').write_text(json.dumps(config,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
