"""Linear/bisect attribution for non-overlapping micro NVTX ranges, CPU only."""
import argparse
from bisect import bisect_right
from collections import defaultdict
import csv
import json
from pathlib import Path
import re
import sqlite3


def write(path,rows):
    if not rows:return
    with path.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader();writer.writerows(rows)


def category(name):
    return ('tiles' if 'conv_dweight_tiles' in name else 'reduce' if 'conv_dweight_reduce' in name
        else 'Copy' if 'cupy_copy' in name else 'Im2col' if name.startswith('im2col')
        else 'GEMM' if 'gemm' in name.lower() else 'other')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--expected-cases',type=int,default=132)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    db=sqlite3.connect('file:'+str(a.input.resolve())+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    strings=dict(db.execute('SELECT id,value FROM StringIds'))
    def read(table):return [dict(r) for r in db.execute('SELECT rowid AS sqlite_rowid,* FROM '+table+' ORDER BY start')]
    ranges=[];by_thread=defaultdict(list)
    for r in read('NVTX_EVENTS'):
        name=r.get('text') or strings.get(r.get('textId'),'')
        match=re.fullmatch(r'OP/micro:(\w+)\|case=(\w+)\|step=(\d+)',name)
        if not match:continue
        assert r['end'] is not None
        r.update(name=name,scope=match[1],case=match[2],step=int(match[3]));ranges.append(r);by_thread[r['globalTid']].append(r)
    starts={tid:[r['start'] for r in rr] for tid,rr in by_thread.items()}
    for rr in by_thread.values():assert all(x['end']<=y['start'] for x,y in zip(rr,rr[1:]))
    apis=[];candidates=defaultdict(list);micro_apis=defaultdict(list)
    for table in ('CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_DRIVER'):
        if table not in tables:continue
        for r in read(table):
            r.update(table=table,name=strings[r['nameId']],micro_id=None)
            rr=by_thread.get(r['globalTid'],[]);index=bisect_right(starts.get(r['globalTid'],[]),r['start'])-1
            if index>=0 and r['start']<rr[index]['end']:r['micro_id']=rr[index]['sqlite_rowid']
            apis.append(r);candidates[r['correlationId']].append(r)
            if r['micro_id'] is not None:micro_apis[r['micro_id']].append(r)
    correlation={}
    for key,rr in candidates.items():
        inner=min(rr,key=lambda r:r['end']-r['start'])
        assert all(r['globalTid']==inner['globalTid'] and r['start']<=inner['start'] and r['end']>=inner['end'] for r in rr)
        correlation[key]=inner
    grouped=defaultdict(list);kernels=read('CUPTI_ACTIVITY_KIND_KERNEL');unmatched=0
    for k in kernels:
        api=correlation.get(k['correlationId']);unmatched+=int(api is None)
        k.update(name=strings[k['shortName']],micro_id=api['micro_id'] if api else None,
            duration_us=(k['end']-k['start'])/1000)
        if k['micro_id'] is not None:grouped[k['micro_id']].append(k)
    assert unmatched==0
    rows=[]
    for r in ranges:
        kk=grouped[r['sqlite_rowid']];aa=micro_apis[r['sqlite_rowid']]
        row=dict(case=r['case'],step=r['step'],scope=r['scope'],nvtx_id=r['sqlite_rowid'],host_nvtx_us=(r['end']-r['start'])/1000,
            kernels=len(kk),kernel_us=sum(k['duration_us'] for k in kk),
            launch_api_count=sum('LaunchKernel' in v['name'] for v in aa),
            launch_api_us=sum((v['end']-v['start'])/1000 for v in aa if 'LaunchKernel' in v['name']))
        for name in ('tiles','reduce','Copy','Im2col','GEMM','other'):
            selected=[k for k in kk if category(k['name'])==name]
            row[name+'_count']=len(selected);row[name+'_us']=sum(k['duration_us'] for k in selected)
        assert row['kernels']>0;rows.append(row)
    counts=defaultdict(int)
    for r in rows:counts[r['case']]+=1
    assert len(counts)==a.expected_cases and set(counts.values())=={30}
    for name,data in [('nvtx-events',ranges),('cuda-api-events',apis),('kernel-events',kernels),('micro-kernels',rows)]:write(a.output/(name+'.csv'),data)
    for table in ('CUPTI_ACTIVITY_KIND_MEMCPY','CUPTI_ACTIVITY_KIND_MEMSET'):
        if table in tables:write(a.output/(table.lower()+'.csv'),read(table))
    summary=dict(status='PASSED',input=str(a.input.resolve()),cases=len(counts),calls=len(rows),
        unmatched_kernel_api_count=unmatched,scope='CUPTI durations via API correlation and same-thread micro NVTX; not Event spans or throughput')
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
