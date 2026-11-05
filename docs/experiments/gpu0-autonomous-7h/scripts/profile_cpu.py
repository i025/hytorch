"""CPU evidence before choosing a scheduling prototype; bounded threads, no GPU."""
import argparse
import cProfile
from datetime import datetime,timezone
import pstats
from pathlib import Path
import time

import numpy as np
from common import Engine,write_csv,write_json
import hytorch
import hytorch.functions as F


def branch_loss(width):
    x=hytorch.Variable(np.arange(128,dtype=np.float32).reshape(4,32)/128)
    root=x*x
    outputs=[F.tanh(root+np.float32(i/width)) for i in range(width)]
    y=outputs[0]
    for v in outputs[1:]:y=y+v
    return F.sum(y),x


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    records=[];timings=[]
    for name in ('donkey','spatial','branches32','branches256'):
        engine=Engine('A',name,device='cpu') if name in ('donkey','spatial') else None
        rng=np.random.default_rng(42)
        x=rng.normal(0,.1,(4,3,66,200)).astype(np.float32)
        t=rng.normal(0,.1,(4,1)).astype(np.float32)
        iterations=5 if engine else 50
        def step():
            if engine:
                loss=engine.loss(x,t);engine.backward(loss);engine.update()
            else:
                loss,_=branch_loss(int(name[8:]));loss.backward()
        step()
        prof=cProfile.Profile()
        prof.enable()
        for _ in range(iterations):step()
        prof.disable();prof.dump_stats(str(a.output/(name+'.pstats')))
        stats=pstats.Stats(prof)
        for (file,line,function),(primitive,calls,own,total,callers) in stats.stats.items():
            records.append(dict(workload=name,file=file,line=line,function=function,primitive_calls=primitive,
                calls=calls,self_s=own,cumulative_s=total,iterations=iterations))
        for repeat in range(3):
            begin=time.perf_counter()
            for _ in range(iterations):step()
            wall=time.perf_counter()-begin
            timings.append(dict(workload=name,repeat=repeat,iterations=iterations,wall_s=wall,ms_per_step=wall*1000/iterations))
    write_csv(a.output/'profile-functions.csv',records);write_csv(a.output/'timings.csv',timings)
    summary=dict(status='COMPLETED',scope='CPU model or synthetic branched graph; not GPU training',
        finished_utc=datetime.now(timezone.utc).isoformat(),gpu='NOT_RUN')
    write_json(a.output/'summary.json',summary);print(summary,flush=True)


if __name__=='__main__':main()
