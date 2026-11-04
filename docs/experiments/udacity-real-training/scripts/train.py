"""Matched real Udacity training; separate resident, I/O, phases and Nsight scopes."""
import argparse
from contextlib import ExitStack
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from engine import Engine,ROOT
from donkeycar_hy.udacity_dataset import UdacityDataset
from hytorch.backends.cuda import dispatch_stats
from hytorch.profiling import StepProfiler


def json_line(path,row):
    with path.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')


def metrics(engine,ds,indices,out,epoch):
    predictions=[];targets=[]
    for start in range(0,len(indices),4):
        batch=[ds[int(i)] for i in indices[start:start+4]]
        prediction=engine.predict(np.stack([v[0] for v in batch])).reshape(-1)
        target=np.array([v[1][0] for v in batch]);predictions.extend(prediction.tolist());targets.extend(target.tolist())
        for i,y,t in zip(indices[start:start+4],prediction,target):
            json_line(out/'validation-predictions.jsonl',dict(epoch=epoch,row=int(i),prediction=float(y),target=float(t)))
    delta=np.array(predictions)-np.array(targets)
    assert np.isfinite(delta).all()
    return dict(validation_loss=float(np.mean(delta**2)),steering_mae=float(np.mean(abs(delta))),steering_rmse=float(np.sqrt(np.mean(delta**2))))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=list('AFIE'),required=True)
    p.add_argument('--task',choices=['smoke','compute','io','phases','train','trace'],required=True)
    p.add_argument('--prepared',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--no-fingerprints',action='store_true',required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((a.prepared/'protocol.json').read_text());ds=UdacityDataset(protocol['csv'])
    metadata=dict(vars(a),prepared=str(a.prepared.resolve()),output=str(a.output.resolve()),protocol=protocol,
        resource_condition='SHARED_GPU',cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
        model='DonkeyNet',repeat_scope='Independent process',clock_scope='No GPU clock or power modifications',status='RUNNING')
    (a.output/'config.json').write_text(json.dumps(metadata,indent=2)+'\n')
    engine=Engine(a.method);np.savez(a.output/'initial_weights.npz',**engine.initial)
    (a.output/'device.json').write_text(json.dumps(dict(cupy=engine.cp.__version__,
        runtime=engine.cp.cuda.runtime.runtimeGetVersion(),logical_device=0,
        pci_bus_id=engine.cp.cuda.Device(0).pci_bus_id,
        gpu_name=str(engine.cp.cuda.runtime.getDeviceProperties(0)['name'])),indent=2)+'\n')
    data=np.load(a.prepared/'throughput-inputs.npz')
    trace=a.task=='trace';phase_mode='phases' if a.task in ('phases','train') else 'throughput'
    profiler=StepProfiler(a.output/'steps.jsonl',engine.sync,phase_mode,trace)
    with ExitStack() as stack:
        recorder=None
        if trace:
            spec=importlib.util.spec_from_file_location('stage4g_ranges',ROOT/'docs/experiments/conv-dweight-fusion/scripts/profile_training.py')
            mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
            recorder=stack.enter_context(mod.WeightRanges(engine.cp.cuda.nvtx.RangePush,engine.cp.cuda.nvtx.RangePop,engine.cp))
        stack.enter_context(engine.context())
        wx,wt=engine.convert(data['x'][:4]),engine.convert(data['t'][:4])
        for _ in range(10):
            loss=engine.loss(wx,wt);engine.backward(loss);engine.update()
        engine.sync();engine.restore();dispatch_stats(reset=True)
        bank=None
        if a.task=='compute':bank=(engine.convert(data['x']),engine.convert(data['t']))
        if a.task=='train':
            before=metrics(engine,ds,protocol['validation'],a.output,0)
            json_line(a.output/'epochs.jsonl',dict(epoch=0,**before))
        epochs=protocol['epoch_order'] if a.task=='train' else [protocol['throughput'][:(32 if a.task=='smoke' else 64 if trace else 1024)]]
        for epoch,indices in enumerate(epochs,1):
            begin=time.perf_counter();epoch_clock=time.time();weighted=0.;count=0;phase_start=len(profiler.rows)
            if a.method=='E':engine.torch.cuda.reset_peak_memory_stats()
            for start in range(0,len(indices),4):
                profiler.begin()
                with profiler.phase('data'):
                    data_begin=time.perf_counter()
                    if bank is None:
                        batch=[ds[int(i)] for i in indices[start:start+4]]
                        x,t=np.stack([v[0] for v in batch]),np.stack([v[1] for v in batch])
                    else:x,t=bank[0][start:start+4],bank[1][start:start+4]
                    data_s=time.perf_counter()-data_begin
                with profiler.phase('transfer'):
                    if bank is None:x,t=engine.convert(x),engine.convert(t)
                with profiler.phase('forward'):loss=engine.loss(x,t)
                with profiler.phase('backward'):engine.backward(loss)
                with profiler.phase('update'):engine.update()
                record=profiler.end(len(x),epoch,start//4+1);value=engine.value(loss)
                assert np.isfinite(value),'Nonfinite real-data loss'
                json_line(a.output/'losses.jsonl',dict(epoch=epoch,step=start//4+1,loss=value,samples=len(x),
                    read_preprocess_s=data_s if bank is None else 0.,timestamp=record['timestamp']))
                weighted+=value*len(x);count+=len(x)
            engine.sync();wall=time.perf_counter()-begin
            row=dict(epoch=epoch,train_loss=weighted/count,train_wall_s=wall,samples=count,samples_per_sec=count/wall,
                timing_scope='Resident inputs; excludes disk/preprocess/H2D' if bank is not None else 'JPEG read+RGB preprocess+H2D+model+logging',
                phase_mode=phase_mode)
            if phase_mode=='phases':
                rr=profiler.rows[phase_start:]
                for key in ('data','transfer','forward','backward','update'):row[key+'_ms']=np.mean([r[key+'_s'] for r in rr])*1000
            if a.task=='train':
                begin=time.perf_counter();row.update(metrics(engine,ds,protocol['validation'],a.output,epoch));row['validation_wall_s']=time.perf_counter()-begin
            row.update(started_at_unix=epoch_clock,finished_at_unix=time.time(),
                cupy_pool_used_bytes=engine.cp.get_default_memory_pool().used_bytes(),
                cupy_pool_reserved_bytes=engine.cp.get_default_memory_pool().total_bytes(),
                torch_peak_allocated_bytes=engine.torch.cuda.max_memory_allocated() if a.method=='E' else None)
            json_line(a.output/'epochs.jsonl',row);print(json.dumps(row),flush=True)
        final=engine.state();np.savez(a.output/'final_weights.npz',**final)
        difference=np.concatenate([(final[k]-engine.initial[k]).ravel() for k in final])
        assert np.isfinite(difference).all() and np.any(difference!=0)
        summary=dict(status='COMPLETED',resource_condition='SHARED_GPU',dispatch_counts=dispatch_stats(),
            parameter_update_max_abs=float(abs(difference).max()),parameter_update_l2=float(np.linalg.norm(difference)),
            steps=len(profiler.rows),cupy_pool_reserved_bytes=engine.cp.get_default_memory_pool().total_bytes(),
            torch_peak_allocated_bytes=engine.torch.cuda.max_memory_allocated() if a.method=='E' else None,
            torch_peak_reserved_bytes=engine.torch.cuda.max_memory_reserved() if a.method=='E' else None)
        (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        if recorder is not None:(a.output/'instrumentation.json').write_text(json.dumps(recorder.summary(),indent=2)+'\n')


if __name__=='__main__':main()
