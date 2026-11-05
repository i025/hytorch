"""Queue and pending diagnostics, kept separate from uninstrumented throughput."""
import argparse
import json
from pathlib import Path
import queue
import resource
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

from common import inputs, loader_module, write_csv, write_json


def measure(args):
    module=loader_module(args.variant,args.baseline)
    dataset,indices=inputs(args.prepared,args.workload,512)
    events,pending_events,batches=[],[],[]
    state=dict(submitted=0,delivered=0,peak_inflight=0,index_peak=0,result_peak=0)
    start=time.perf_counter()
    class TrackedQueue(queue.Queue):
        def __init__(self,maxsize=0):
            super().__init__(maxsize)
            self.kind='result' if maxsize else 'index'
        def put(self,item,*args,**kwargs):
            if self.kind=='index' and isinstance(item,tuple):
                state['submitted']+=1
                state['peak_inflight']=max(state['peak_inflight'],state['submitted']-state['delivered'])
            begin=time.perf_counter();success=False
            try:
                super().put(item,*args,**kwargs)
                success=True
            finally:
                self.record('put',begin,success,kwargs.get('block',args[0] if args else True))
        def get(self,*args,**kwargs):
            begin=time.perf_counter();success=False
            try:
                value=super().get(*args,**kwargs)
                success=True
                return value
            finally:
                self.record('get',begin,success,kwargs.get('block',args[0] if args else True))
        def record(self,operation,begin,success,block):
            depth=self.qsize()
            state[self.kind+'_peak']=max(state[self.kind+'_peak'],depth)
            events.append(dict(queue=self.kind,operation=operation,ms=(time.perf_counter()-begin)*1000,
                success=success,block=block,depth=depth,thread=threading.current_thread().name))
    class Pending(dict):
        def __init__(self):
            super().__init__()
            self.peak=0;self.bytes=0;self.peak_bytes=0
        def __setitem__(self,key,value):
            super().__setitem__(key,value)
            self.bytes+=sum(x.nbytes for x in value)
            self.peak=max(self.peak,len(self));self.peak_bytes=max(self.peak_bytes,self.bytes)
            pending_events.append(dict(elapsed_ms=(time.perf_counter()-start)*1000,sequence=key,
                pending=len(self),pending_bytes=self.bytes,submitted=state['submitted'],delivered=state['delivered']))
        def pop(self,key):
            value=super().pop(key)
            self.bytes-=sum(x.nbytes for x in value)
            state['delivered']+=1
            return value
        def clear(self):
            super().clear()
            self.bytes=0
    original_queue=module.queue
    module.queue=SimpleNamespace(Queue=TrackedQueue,Empty=queue.Empty,Full=queue.Full)
    try:
        loader=module.DataLoader(dataset,4,shuffle=False,indices=indices,
                                 num_workers=args.workers,prefetch_factor=args.prefetch)
        iterator=iter(loader)
        threads=list(loader._impl._workers)
        pending=Pending();loader._impl._pending=pending
        for step in range(loader.max_iter):
            begin=time.perf_counter()
            x,t=next(iterator)
            batches.append(dict(step=step+1,next_ms=(time.perf_counter()-begin)*1000,
                submitted=state['submitted'],delivered=state['delivered'],
                pending=len(pending),index_depth=loader._impl._index_queue.qsize(),
                result_depth=loader._impl._result_queue.qsize()))
            if args.workload=='real-consumer':time.sleep(.004)
        assert next(iterator,None) is None
        row=dict(variant=args.variant,workload=args.workload,workers=args.workers,prefetch_factor=args.prefetch,
            samples=len(indices),**state,pending_peak=pending.peak,pending_peak_mib=pending.peak_bytes/1024**2,
            peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            workers_alive_after=sum(w.is_alive() for w in threads),wall_s=time.perf_counter()-start,
            scope='Instrumented CPU diagnostics, overlapping worker waits; not formal throughput')
        for kind in ('index','result'):
            for op in ('put','get'):
                values=[e for e in events if e['queue']==kind and e['operation']==op]
                row[kind+'_'+op+'_ms']=sum(e['ms'] for e in values)
                row[kind+'_'+op+'_calls']=len(values)
        if args.variant=='bounded':
            assert state['peak_inflight']<=args.workers*args.prefetch
            assert pending.peak<=args.workers*args.prefetch
        assert row['workers_alive_after']==0
        write_csv(args.output/'queue-events.csv',events)
        write_csv(args.output/'pending-events.csv',pending_events)
        write_csv(args.output/'batches.csv',batches)
        write_json(args.output/'summary.json',row)
        print(args.output.name+': peak_inflight='+str(state['peak_inflight'])+', peak_pending='+str(pending.peak),flush=True)
    finally:
        module.queue=original_queue


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--variant',choices=['baseline','bounded'])
    parser.add_argument('--workload',choices=['real-fast','real-consumer','straggler'],default='real-fast')
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--prefetch',type=int,default=2)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    if args.variant:
        measure(args);return
    rows,commands=[],[]
    for workload in ('real-fast','real-consumer','straggler'):
        for workers in (2,4):
            for prefetch in (1,2,4):
                for variant in ('baseline','bounded'):
                    output=args.output/f'{workload}-{variant}-w{workers}-p{prefetch}'
                    command=[sys.executable,__file__,'--prepared',str(args.prepared),'--baseline',str(args.baseline),
                        '--output',str(output),'--workload',workload,'--workers',str(workers),'--prefetch',str(prefetch),'--variant',variant]
                    commands.append(command);write_json(args.output/'commands.json',commands)
                    subprocess.run(command,check=True)
                    rows.append(json.loads((output/'summary.json').read_text()))
                    write_csv(args.output/'summary.csv',rows)
    write_json(args.output/'status.json',dict(status='COMPLETED',configurations=len(rows),gpu='NOT_RUN'))


if __name__=='__main__':
    main()
