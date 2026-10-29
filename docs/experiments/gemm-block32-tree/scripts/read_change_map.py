"""Read a published lossless per-position improvement map (no GPU required)."""
import argparse
import gzip
from pathlib import Path
import numpy as np


def read_map(path,n):
    with gzip.open(path,'rb') as stream:
        packed=np.frombuffer(stream.read(),dtype=np.uint8)
    bits=np.unpackbits(packed,bitorder='little')[:2*n*n].reshape(-1,2)
    return (bits[:,0]+2*bits[:,1]).reshape(n,n)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('path',type=Path)
    p.add_argument('--n',type=int,required=True)
    p.add_argument('--row',type=int)
    p.add_argument('--col',type=int)
    args=p.parse_args()
    values=read_map(args.path,args.n)
    print({name:int(np.count_nonzero(values==code)) for code,name in enumerate(('worse','equal','better'))})
    if args.row is not None and args.col is not None:
        print(('worse','equal','better')[int(values[args.row,args.col])])
