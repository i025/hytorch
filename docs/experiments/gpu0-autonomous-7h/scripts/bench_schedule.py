"""Matched CPU prototype checks and repeated complete-step timing; no CUDA."""
import argparse
import csv
from datetime import datetime,timezone
from pathlib import Path
import time

import numpy as np
from common import Engine,compare,result,write_json,write_csv
from profile_cpu import branch_loss
import hytorch
import hytorch.functions as F
from hytorch.optimizers import SGD
from hytorch.transformer.encoder_block import TransformerEncoderBlock
from benchmarks.bench_cuda_training import parameters,snapshot


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prepared',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    data=np.load(a.prepared/'throughput-inputs.npz');timings=[];checks=[]
    for workload in ('donkey','spatial','transformer-block','branches32','branches256'):
        engine=Engine('A',workload,device='cpu') if workload in ('donkey','spatial') else None
        model=None
        if engine:
            restore=engine.restore
            def step():
                loss=engine.loss(data['x'][:4],data['t'][:4]);engine.backward(loss);engine.update()
        elif workload=='transformer-block':
            np.random.seed(42);model=TransformerEncoderBlock(32,4,64,dropout=0.)
            rng=np.random.default_rng(42)
            x=rng.normal(0,.1,(2,8,32)).astype(np.float32)
            target=rng.normal(0,.1,x.shape).astype(np.float32)
            model(x);initial=snapshot(model);optimizer=SGD(.001).setup(model)
            def restore():
                for k,v in parameters(model).items():v.data=initial[k].copy()
                model.cleargrads()
            def step():
                loss=F.mean_square_error(model(x),target);model.cleargrads();loss.backward();optimizer.update()
        else:
            restore=lambda:None
            def step():
                loss,_=branch_loss(int(workload[8:]));loss.backward()
        # Exact scheduler equivalence is stronger than the inherited numerical tolerance.
        if engine:
            values={}
            for mode in ('list','heap'):
                restore()
                with hytorch.using_config('backward_schedule',mode):
                    values[mode]=[result(engine,data['x'][4*i:4*i+4],data['t'][4*i:4*i+4]) for i in range(3)]
            for i,(actual,old) in enumerate(zip(values['heap'],values['list'])):
                for key in actual:
                    row=compare(actual[key],old[key],key,model=workload,step=i+1)
                    checks.append(row)
            np.savez(a.output/(workload+'-numerical.npz'),**{
                mode+'/'+str(i)+'/'+key:value for mode,steps in values.items()
                for i,step_ in enumerate(steps) for key,value in step_.items()})
        if model:
            values={}
            for mode in ('list','heap'):
                restore()
                with hytorch.using_config('backward_schedule',mode):
                    step()
                    values[mode]=snapshot(model)
                    values[mode].update({'grad/'+k:v.grad.data.copy() for k,v in parameters(model).items()})
            for key in values['heap']:
                row=compare(values['heap'][key],values['list'][key],key,model=workload,step=1)
                checks.append(row)
        iterations=10 if engine else 50
        for repeat in range(5):
            order=('list','heap') if repeat%2==0 else ('heap','list')
            for mode in order:
                restore()
                with hytorch.using_config('backward_schedule',mode):
                    for _ in range(3):step()
                    restore()
                    begin=time.perf_counter()
                    for _ in range(iterations):step()
                    wall=time.perf_counter()-begin
                timings.append(dict(workload=workload,mode=mode,repeat=repeat,iterations=iterations,
                    wall_s=wall,ms_per_step=wall*1000/iterations))
                path=a.output/'timings.csv';exists=path.exists()
                with path.open('a',newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=list(timings[-1]),lineterminator='\n')
                    if not exists:writer.writeheader()
                    writer.writerow(timings[-1])
    write_csv(a.output/'correctness.csv',checks)
    value=dict(status='PASS' if all(r['bitwise_equal'] for r in checks) else 'FAIL',
        comparisons=len(checks),unequal=sum(not r['bitwise_equal'] for r in checks),gpu='NOT_RUN',
        scope='CPU matched initial states; exact equality; five alternating timing repeats',
        finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(a.output/'summary.json',value);print(value,flush=True)
    assert value['status']=='PASS'


if __name__=='__main__':main()
