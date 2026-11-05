"""Show that bounded results do not bound the existing ordered pending dictionary."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from hytorch.dataloaders import AsyncDataLoader


class Delayed:
    def __len__(self):
        return 100
    def __getitem__(self, index):
        if index==0:
            time.sleep(.2)
        return np.array([index], np.float32), np.array([index], np.float32)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    loader=AsyncDataLoader(Delayed(),1,shuffle=False,num_workers=2,prefetch_factor=1)
    iter(loader)
    x,t=next(loader)
    assert x[0,0]==0
    row=dict(status='OBSERVED',result_queue_capacity=loader._result_queue.maxsize,
             pending_after_first_batch=len(loader._pending),samples=100,
             note='Existing head-of-line buffering is not a strict total memory bound; not changed in this lifecycle fix')
    loader.close()
    args.output.write_text(json.dumps(row,indent=2)+'\n')
    print(json.dumps(row))


if __name__=='__main__':
    main()
