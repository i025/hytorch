"""One 256-step CPU Sum-model diagnosis; automatically yields to the GPU window."""
import argparse
from contextlib import contextmanager
from datetime import datetime,timezone
import json
from pathlib import Path
import time

import numpy as np
from common import Engine,compare,result,write_json,write_csv
from reference import modeled_sum
from hytorch.backends.cuda import runtime


@contextmanager
def arithmetic(mode):
    original=runtime.sum_array
    def sum_model(x,axis=None,keepdims=False):
        if x.dtype==np.float32 and x.size and x.ndim<=8:return modeled_sum(x,axis,keepdims)
        return original(x,axis,keepdims)
    if mode=='cpu_model':runtime.sum_array=sum_model
    try:yield
    finally:runtime.sum_array=original


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,required=True);p.add_argument('--schedule',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(a.schedule.read_text());raw=Path(cfg['raw']);window=raw/'gpu-window-01/window.json'
    cutoff=datetime.fromisoformat(cfg['gpu_cutoff_utc']).timestamp();paused=0.
    def cpu_slot():
        nonlocal paused
        begin=time.monotonic()
        while window.exists() and json.loads(window.read_text())['status']=='RUNNING':
            if time.time()>cutoff:raise RuntimeError('CPU diagnostic cutoff during GPU priority work')
            time.sleep(1)
        paused+=time.monotonic()-begin
        if time.time()>cutoff:raise RuntimeError('CPU diagnostic budget cutoff')
    data=np.load(a.prepared/'throughput-inputs.npz');checks=[];loss_rows=[];replays=[]
    checkpoints=(1,16,64,128,255,256)
    for model in ('donkey','spatial'):
        losses={};states={}
        for mode in ('original','cpu_model'):
            engine=Engine('A',model,device='cpu');losses[mode]=[];states[mode]={}
            if mode=='original':np.savez(a.output/(model+'-initial.npz'),**engine.initial)
            with arithmetic(mode):
                for step in range(1,257):
                    cpu_slot();x=data['x'][(step-1)*4:step*4];t=data['t'][(step-1)*4:step*4]
                    prediction,loss=engine.forward_loss(x,t);engine.backward(loss)
                    value=engine.value(loss);assert np.isfinite(value),(model,mode,step,value)
                    values=None
                    if step in checkpoints:
                        values=dict(forward=prediction.data.copy(),loss=np.array(value))
                        values.update({'grad/'+k:v for k,v in engine.gradients().items()})
                    engine.update();losses[mode].append(value)
                    row=dict(model=model,mode=mode,step=step,loss=value,utc=datetime.now(timezone.utc).isoformat())
                    loss_rows.append(row)
                    with (a.output/'losses.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                    if values is not None:
                        values.update({'weight/'+k:v for k,v in engine.state().items()})
                        states[mode][step]=values
                        np.savez(a.output/f'{model}-{mode}-step{step}.npz',**values)
                        print(model,mode,step,value,flush=True)
            del engine
        for step,(actual,old) in enumerate(zip(losses['cpu_model'],losses['original'])):
            checks.append(compare(np.array(actual),np.array(old),'loss',model=model,step=step+1,scope='trajectory'))
        for step in checkpoints:
            for key,value in states['cpu_model'][step].items():
                if key!='loss':checks.append(compare(value,states['original'][step][key],key,model=model,step=step,scope='checkpoint'))
        # Identical post-update state + identical next batch isolates per-step math from trajectory drift.
        for checkpoint in (0,1,16,64,128,255):
            initial=np.load(a.output/(model+'-initial.npz')) if checkpoint==0 else None
            state={k:initial[k].copy() for k in initial.files} if checkpoint==0 else {
                k[7:]:v.copy() for k,v in states['original'][checkpoint].items() if k.startswith('weight/')}
            values={};index=checkpoint*4
            for mode in ('original','cpu_model'):
                cpu_slot();engine=Engine('A',model,device='cpu');engine.load_state(state)
                with arithmetic(mode):values[mode]=result(engine,data['x'][index:index+4],data['t'][index:index+4])
            for key,value in values['cpu_model'].items():
                replays.append(compare(value,values['original'][key],key,model=model,checkpoint=checkpoint,scope='same-state'))
            np.savez(a.output/f'{model}-replay{checkpoint}.npz',**{
                mode+'/'+key:v for mode,items in values.items() for key,v in items.items()})
        write_csv(a.output/(model+'-trajectory-errors.csv'),[r for r in checks if r['model']==model])
        write_csv(a.output/(model+'-replay-errors.csv'),[r for r in replays if r['model']==model])
    write_csv(a.output/'losses.csv',loss_rows)
    summary=dict(status='DIAGNOSIS_COMPLETED',gpu='NOT_RUN',steps_per_model_per_mode=256,
        trajectory_failed=sum(bool(r['mismatches']) for r in checks),replay_failed=sum(bool(r['mismatches']) for r in replays),
        trajectory_comparisons=len(checks),replay_comparisons=len(replays),cpu_slot_check_and_wait_s=paused,
        first_loss_failure={m:next((r['step'] for r in checks if r['model']==m and r['scope']=='trajectory' and r['mismatches']),None)
            for m in ('donkey','spatial')},finished_utc=datetime.now(timezone.utc).isoformat(),
        scope='CPU mathematical simulation of existing Sum; not a CUDA execution or performance benchmark')
    write_json(a.output/'summary.json',summary);print(summary,flush=True)


if __name__=='__main__':main()
