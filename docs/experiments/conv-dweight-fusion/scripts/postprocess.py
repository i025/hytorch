"""Export/analyze completed captures after the controller released CUDA."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'docs/experiments/fused-sgd-cuda/scripts'))
from analyze_fused_results import csv_rows,write_csv,json_file


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,required=True);p.add_argument('--analysis',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reuse-analysis',action='store_true',help='Use the existing exported and parsed captures')
    a=p.parse_args();assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    window=json_file(a.raw/'window.json');assert window['context_released'] and window['status']=='COMPLETED'
    if not a.reuse_analysis:a.analysis.mkdir(parents=True,exist_ok=False)
    a.output.mkdir(parents=True,exist_ok=False)
    logdir=a.output if a.reuse_analysis else a.analysis
    def run(name,command):
        with (logdir/(name+'.log')).open('x') as f:done=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        with (logdir/'commands.jsonl').open('a') as f:f.write(json.dumps(dict(name=name,command=command,returncode=done.returncode))+'\n')
        done.check_returncode()
    databases=[]
    for rep in sorted(a.raw.glob('*.nsys-rep')):
        database=a.analysis/(rep.stem+'.sqlite');databases.append(database)
        if a.reuse_analysis:
            assert database.is_file() and (a.analysis/rep.stem).is_dir()
            continue
        run(rep.stem+'-export',['nsys','export','--type','sqlite','--output',str(database),str(rep)])
        if rep.stem.startswith('trace-micro-'):
            command=[sys.executable,str(HERE/'analyze_micro_trace.py'),'--input',str(database),'--output',str(a.analysis/rep.stem)]
        else:
            command=[sys.executable,str(ROOT/'docs/experiments/donkeynet-backward-profiling/scripts/analyze_trace.py'),
                '--input',str(database),'--output',str(a.analysis/rep.stem),'--label',rep.stem]
        run(rep.stem+'-analyze',command)
    assert len(databases)==17
    run('micro-summary',[sys.executable,str(HERE/'analyze_micro.py'),'--raw',str(a.raw),'--analysis',str(a.analysis),'--output',str(a.output/'micro')])
    run('training-summary',[sys.executable,str(HERE/'analyze_training.py'),'--raw',str(a.raw),'--output',str(a.output/'training')])
    phases,parts,windows,checks=[],[],[],[]
    native=re.compile(r'\b(gemm_basic|gemm_blocked|gemm_shared|im2col_kernel|conv_direct|conv_pack|pool_kernel|multi_tensor_sgd_kernel|conv_dinput_gather|conv_dinput_gather_opt|conv_dweight_tiles|conv_dweight_reduce)\b')
    for model in ('donkey','spatial'):
        for method in 'AFIE':
            folder=a.analysis/('trace-training-'+model+'-'+method);tag=dict(model=model,method=method)
            summary=json_file(folder/'profiling-summary.json');assert summary['unmatched_kernel_api_count']==0
            checks.append(dict(**tag,**summary))
            kk=csv_rows(folder/'kernel-events.csv');rr=csv_rows(folder/'nvtx-events.csv')
            apis={r['api_event_id']:r for r in csv_rows(folder/'cuda-api-events.csv')}
            if method=='A':assert not any(native.search(k['name']) for k in kk)
            for k in kk:k['ancestors']=set(json.loads(k['nvtx_ids']))
            phases.extend(dict(r,**tag) for r in csv_rows(folder/'phase-summary.csv'))
            for r in rr:
                if r['phase']!='backward' or not r['name'].startswith('OP/part:Conv2d.dWeight|layer='):continue
                events=[k for k in kk if int(r['sqlite_rowid']) in k['ancestors']]
                row=dict(**tag,layer=r['name'].split('layer=')[1],nvtx_id=r['sqlite_rowid'],host_us=float(r['duration_ms'])*1000,
                    kernels=len(events),kernel_us=sum(float(k['duration_ms']) for k in events)*1000)
                for label,test in [('tiles',lambda name:'conv_dweight_tiles' in name),('reduce',lambda name:'conv_dweight_reduce' in name),
                    ('Copy',lambda name:'cupy_copy' in name),('Im2col',lambda name:name.startswith('im2col')),('GEMM',lambda name:'gemm' in name.lower())]:
                    selected=[k for k in events if test(k['name'])]
                    row[label+'_count']=len(selected);row[label+'_us']=sum(float(k['duration_ms']) for k in selected)*1000
                if method=='I':assert row['kernels']==2 and row['tiles_count']==row['reduce_count']==1 and row['Copy_count']==row['Im2col_count']==row['GEMM_count']==0
                parts.append(row)
            if method=='I':assert sum(r['model']==model and r['method']==method for r in parts)==16*(5 if model=='donkey' else 3)
            for r in rr:
                if r['name']!='backward':continue
                # Worker APIs inherit a unique main-thread phase without that
                # thread's NVTX ID in their local ancestry. Use API submit time.
                events=[k for k in kk if k['phase']=='backward' and
                    int(r['start'])<=int(apis[k['api_event_id']]['start'])<int(r['end'])]
                intervals=sorted((int(k['start']),int(k['end'])) for k in events)
                union=0;end=0
                for lo,hi in intervals:union+=max(0,hi-max(lo,end));end=max(end,hi)
                span=max(hi for _,hi in intervals)-intervals[0][0]
                windows.append(dict(**tag,nvtx_id=r['sqlite_rowid'],start_ns=r['start'],end_ns=r['end'],host_us=float(r['duration_ms'])*1000,
                    kernel_union_us=union/1000,kernel_span_us=span/1000,between_own_kernels_us=(span-union)/1000))
    write_csv(a.output/'trace-phases.csv',phases);write_csv(a.output/'dweight-components.csv',parts);write_csv(a.output/'backward-windows.csv',windows)
    (a.output/'trace-checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    (a.output/'raw-location.json').write_text(json.dumps(dict(raw=str(a.raw.resolve()),analysis=str(a.analysis.resolve())),indent=2)+'\n')
    result=dict(status='PASSED',trace_count=len(databases),unmatched_kernel_api_count=0,window=window,
        micro=json_file(a.output/'micro/summary.json'),training=json_file(a.output/'training/analysis.json'))
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
