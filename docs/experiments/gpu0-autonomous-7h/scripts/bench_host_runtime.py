"""CPU-only metadata and host ABI lower bounds, not CUDA submission timings."""
import argparse
import ctypes as ct
from datetime import datetime,timezone
from pathlib import Path
import time

from common import ROOT,write_csv,write_json
from cases import CASES
from hytorch.backends.cuda.reduction import plan_sum
from hytorch.backends.cuda import backend


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    library=ct.CDLL(str(a.library.resolve()))
    abi=library.hy_cuda_abi_version;abi.argtypes=[];abi.restype=ct.c_int
    assert abi()==1  # Pure host function, no CUDA device API.
    rows=[]
    def measure(name,fn,**meta):
        for _ in range(100):fn()
        for repeat in range(5):
            wall=time.perf_counter();cpu=time.thread_time()
            for _ in range(1000):fn()
            rows.append(dict(operation=name,repeat=repeat,calls=1000,
                wall_us_per_call=(time.perf_counter()-wall)*1000,
                thread_cpu_us_per_call=(time.thread_time()-cpu)*1000,**meta))
    for case,(shape,axis) in enumerate(CASES):
        measure('plan_sum',lambda:plan_sum(shape,axis),case=case,shape=str(shape),axis=str(axis))
    measure('host_abi_version',abi,case=-1,shape='',axis='')
    def context():
        with backend(reduction='native'):pass
    measure('backend_context_only',context,case=-1,shape='',axis='')
    write_csv(a.output/'timings.csv',rows)
    value=dict(status='COMPLETED',gpu='NOT_RUN',measurements=len(rows),
        warning='Host ABI version has no arguments or launch; NOT equivalent to hy_reduce_sum CUDA submission',
        finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(a.output/'summary.json',value);print(value,flush=True)


if __name__=='__main__':main()
