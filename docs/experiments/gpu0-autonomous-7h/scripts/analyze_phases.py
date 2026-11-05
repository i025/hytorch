"""Attribute kernels by CUDA API correlation and same-thread NVTX, CPU only."""
import argparse
from bisect import bisect_right
from collections import defaultdict
import csv
import json
from pathlib import Path
import sqlite3


def write(path,rows):
    if rows:
        with path.open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def attribute(ranges,apis,kernels):
    threads=defaultdict(list)
    for r in sorted(ranges,key=lambda v:v['start']):threads[r['globalTid']].append(r)
    starts={t:[r['start'] for r in rr] for t,rr in threads.items()}
    for rr in threads.values():assert all(a['end']<=b['start'] for a,b in zip(rr,rr[1:]))
    by_correlation=defaultdict(list)
    for api in apis:
        rr=threads.get(api['globalTid'],[])
        i=bisect_right(starts.get(api['globalTid'],[]),api['start'])-1
        api['phase']=rr[i]['name'] if i>=0 and api['start']<rr[i]['end'] else 'OUTSIDE_MEASURED_PHASE'
        by_correlation[api['correlationId']].append(api)
    unmatched=0
    for k in kernels:
        candidates=by_correlation.get(k['correlationId'],[])
        if not candidates:
            k['phase']='UNMATCHED_API';unmatched+=1
        else:
            # Runtime and nested driver API may share a correlation. Choose inner submission.
            api=min(candidates,key=lambda v:v['end']-v['start'])
            assert all(a['globalTid']==api['globalTid'] and a['start']<=api['start'] and a['end']>=api['end'] for a in candidates)
            k['phase']=api['phase']
    return unmatched


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    db=sqlite3.connect('file:'+str(a.input.resolve())+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    strings=dict(db.execute('SELECT id,value FROM StringIds'))
    tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    def read(table):return [dict(r) for r in db.execute('SELECT rowid AS id,* FROM '+table+' ORDER BY start')]
    ranges=[]
    for r in read('NVTX_EVENTS'):
        name=r.get('text') or strings.get(r.get('textId'),'')
        if name in ('data','transfer','forward','backward','update'):
            assert r['end'] is not None;r['name']=name;ranges.append(r)
    apis=[]
    for t in ('CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_DRIVER'):
        if t in tables:
            for r in read(t):r['name']=strings[r['nameId']];r['table']=t;apis.append(r)
    kernels=read('CUPTI_ACTIVITY_KIND_KERNEL')
    for k in kernels:k['name']=strings[k['shortName']]
    unmatched=attribute(ranges,apis,kernels)
    copies=read('CUPTI_ACTIVITY_KIND_MEMCPY') if 'CUPTI_ACTIVITY_KIND_MEMCPY' in tables else []
    unmatched_copies=attribute(ranges,apis,copies)
    kk=[dict(name=k['name'],phase=k['phase'],start_ns=k['start'],end_ns=k['end'],
        duration_us=(k['end']-k['start'])/1000,correlation_id=k['correlationId']) for k in kernels]
    aa=[dict(name=k['name'],phase=k['phase'],start_ns=k['start'],end_ns=k['end'],
        duration_us=(k['end']-k['start'])/1000,correlation_id=k['correlationId'],table=k['table']) for k in apis]
    rr=[dict(phase=r['name'],start_ns=r['start'],end_ns=r['end'],host_us=(r['end']-r['start'])/1000) for r in ranges]
    cc=[dict(phase=r['phase'],start_ns=r['start'],end_ns=r['end'],duration_us=(r['end']-r['start'])/1000,
        bytes=r['bytes'],copy_kind=r['copyKind'],correlation_id=r['correlationId']) for r in copies]
    summary=[]
    for phase in dict.fromkeys([k['phase'] for k in kk]+[v['phase'] for v in cc]+[r['phase'] for r in rr]):
        selected=[k for k in kk if k['phase']==phase];api=[v for v in aa if v['phase']==phase]
        copy=[v for v in cc if v['phase']==phase]
        summary.append(dict(phase=phase,kernel_count=len(selected),kernel_sum_us=sum(k['duration_us'] for k in selected),
            host_nvtx_sum_us=sum(r['host_us'] for r in rr if r['phase']==phase),
            launch_api_calls=sum('LaunchKernel' in v['name'] for v in api),
            launch_api_host_us=sum(v['duration_us'] for v in api if 'LaunchKernel' in v['name']),
            sync_api_host_us=sum(v['duration_us'] for v in api if 'Synchronize' in v['name']),
            memcpy_count=len(copy),memcpy_bytes=sum(v['bytes'] for v in copy),memcpy_gpu_us=sum(v['duration_us'] for v in copy)))
    for name,rows in [('kernels',kk),('apis',aa),('ranges',rr),('memcpy',cc),('phase-summary',summary)]:write(a.output/(name+'.csv'),rows)
    (a.output/'summary.json').write_text(json.dumps(dict(status='PASS' if not unmatched and not unmatched_copies else 'UNMATCHED_ACTIVITY',
        kernels=len(kk),unmatched=unmatched,unmatched_copies=unmatched_copies,
        outside_scope='Outside selected same-thread NVTX; not necessarily only initialization/warmup',
        scope='Kernel durations via launch API; NVTX is host range, not synchronized phase wall or GPU utilization'),indent=2)+'\n')
    assert unmatched==0 and unmatched_copies==0,'Unmatched activity retained; per-phase attribution is incomplete'


if __name__=='__main__':main()
