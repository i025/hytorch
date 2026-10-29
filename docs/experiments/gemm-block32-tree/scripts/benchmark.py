"""Preallocated FP32 CUDA-event timings, only after the complete accuracy phase."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import numpy as np
import cupy as cp
import torch
from common import setup, metadata, shared_call, inputs, write_csv, write_json

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--library',required=True)
p.add_argument('--accuracy',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
args=p.parse_args()
accuracy=json.loads((args.accuracy/'summary.json').read_text())
assert accuracy['status']=='COMPLETED' and accuracy['cases']==9
assert all(c['bitwise_mismatches']==0 for c in accuracy['large_stage3a_semantic_checks'])
args.output.mkdir(parents=True,exist_ok=False)
kernel=setup(args.library)
raw_rows=[];summaries=[];loads=[];checks=[]
names=['shared32','block32_tree','cupy','torch']


def load():
    fields='index,uuid,clocks.sm,clocks.mem,utilization.gpu,memory.used,power.draw,temperature.gpu'
    value=subprocess.check_output(['nvidia-smi','--id='+metadata()['uuid'],'--query-gpu='+fields,
        '--format=csv,noheader,nounits'],text=True).strip()
    return dict(zip(fields.split(','),[v.strip() for v in value.split(',')]))


for case_index,(n,seed_base) in enumerate((n,s) for n in (1024,2048,4096) for s in (42,1729,20261029)):
    a,b=inputs(n,seed_base)
    ca,cb=cp.asarray(a),cp.asarray(b)
    outputs={name:cp.empty((n,n),cp.float32) for name in names[:-1]}
    ta,tb=torch.from_numpy(a).cuda(0),torch.from_numpy(b).cuda(0)
    ty=torch.empty((n,n),device='cuda',dtype=torch.float32)
    calls={'shared32':shared_call(ca,cb,outputs['shared32']),
           'block32_tree':kernel.prepared(ca,cb,outputs['block32_tree']),
           'cupy':lambda:cp.matmul(ca,cb,out=outputs['cupy']),
           'torch':lambda:torch.matmul(ta,tb,out=ty)}
    # Rotate group order deterministically; use the same input values for all methods.
    order=names[case_index%4:]+names[:case_index%4]
    for order_index,method in enumerate(order):
        fn=calls[method]
        for _ in range(5):fn()
        cp.cuda.get_current_stream().synchronize()
        before=load();values=[]
        start,end=cp.cuda.Event(),cp.cuda.Event()
        for repeat in range(10):
            stamp=datetime.now(timezone.utc).isoformat()
            start.record();fn();end.record();end.synchronize()
            elapsed=float(cp.cuda.get_elapsed_time(start,end))
            values.append(elapsed)
            raw_rows.append(dict(n=n,seed_base=seed_base,seed=n+seed_base,method=method,
                order_index=order_index,repeat=repeat,latency_ms=elapsed,timestamp_utc=stamp))
        after=load()
        loads.append(dict(n=n,seed_base=seed_base,method=method,before=before,after=after))
        mean=float(np.mean(values))
        summaries.append(dict(n=n,seed_base=seed_base,seed=n+seed_base,method=method,warmup=5,repeats=10,
            mean_ms=mean,median_ms=float(np.median(values)),std_ms=float(np.std(values,ddof=1)),
            p95_ms=float(np.percentile(values,95)),gflops=2*n**3/(mean*1e6)))
        actual=ty.cpu().numpy() if method=='torch' else cp.asnumpy(outputs[method])
        reference=np.load(Path(accuracy['raw_directory'])/f'n{n}-seed{seed_base+n}'/f'{method}.npy',mmap_mode='r')
        unequal=int(np.count_nonzero(actual!=reference))
        checks.append(dict(n=n,seed_base=seed_base,method=method,unequal_to_accuracy_run=unequal))
        assert unequal==0,'Timed output differs from the completed accuracy experiment'
        print(json.dumps(summaries[-1]),flush=True)
    del a,b,ca,cb,outputs,ta,tb,ty,calls,fn,actual,reference
    cp.get_default_memory_pool().free_all_blocks()
for row in summaries:
    group={r['method']:r for r in summaries if r['n']==row['n'] and r['seed_base']==row['seed_base']}
    for baseline in ('shared32','cupy','torch'):
        row['speedup_vs_'+baseline]=group[baseline]['mean_ms']/row['mean_ms']
write_csv(args.output/'benchmark-raw.csv',raw_rows)
write_csv(args.output/'benchmark-summary.csv',summaries)
write_csv(args.output/'post-timing-correctness.csv',checks)
write_json(args.output/'group-load.json',loads)
write_json(args.output/'summary.json',dict(**metadata(),status='COMPLETED',groups=len(summaries),measurements=len(raw_rows),
    warmup=5,repeats=10,seed_bases=[42,1729,20261029],timing='CUDA events on shared default stream; preallocated inputs/outputs; no H2D/D2H',
    std='sample ddof=1',p95='NumPy linear percentile',method_order='cyclic rotation per matrix case',
    accuracy_output=str(args.accuracy),new_kernel_version_count=1))
