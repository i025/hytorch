"""Exact CPU batches, original split and index order, without hash verification."""
import argparse
import json
from pathlib import Path

import numpy as np
from common import inputs, loader_module, write_csv, write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    with np.load(args.prepared/'throughput-inputs.npz') as data:
        rx,rt=data['x'],data['t']
    dataset,indices=inputs(args.prepared,'real-fast',1024)
    protocol=json.loads((args.prepared/'protocol.json').read_text())
    for key,value in zip(('train','validation','purged'),dataset.split_indices()):
        np.testing.assert_array_equal(value,protocol[key])
    rows=[]
    for variant in ('baseline','bounded'):
        module=loader_module(variant,args.baseline)
        for workers,prefetch in [(0,2)]+[(w,p) for w in (1,2,4) for p in (1,2,4)]:
            offset=0
            with module.DataLoader(dataset,4,shuffle=False,indices=indices,num_workers=workers,prefetch_factor=prefetch) as loader:
                for x,t in loader:
                    np.testing.assert_array_equal(x,rx[offset:offset+len(x)])
                    np.testing.assert_array_equal(t,rt[offset:offset+len(t)])
                    offset+=len(x)
            assert offset==1024
            rows.append(dict(variant=variant,workers=workers,prefetch_factor=prefetch,samples=offset,exact_pixels=True,exact_labels=True,unchanged_split=True))
    write_csv(args.output/'batches.csv',rows)
    write_json(args.output/'summary.json',dict(status='PASS',configurations=len(rows),image_presentations=20480,gpu='NOT_RUN'))
    print('PASS: 20 real-data configurations and unchanged split',flush=True)


if __name__=='__main__':
    main()
