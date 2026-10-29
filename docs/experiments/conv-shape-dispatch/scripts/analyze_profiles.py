"""Correlation-based Nsight attribution, separate from uninstrumented timing."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'docs/experiments/fused-sgd-cuda/scripts'))
from analyze_fused_results import arrays,identical,difference,write_csv,json_file,json_lines,csv_rows
sys.path.insert(0,str(ROOT/'docs/experiments/conv-backward-fusion/scripts'))
from analyze_results import interval_union


def category(name):
    return ('Gather' if 'conv_dinput_gather' in name else 'Copy' if 'cupy_copy' in name
            else 'Col2im' if name.startswith('col2im') else 'GEMM' if 'gemm' in name.lower() else 'other')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--analysis',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);a.analysis.mkdir(parents=True,exist_ok=False)
    assert json_file(a.raw/'completion.json')['status']=='COMPLETED'
    old=ROOT/'docs/experiments/donkeynet-backward-profiling/scripts/analyze_trace.py'
    phases,categories,parts,windows,memory,shapes,choices,errors,checks=[],[],[],[],[],[],[],[],[]
    summaries=[]
    native=re.compile(r'\b(gemm_basic|gemm_blocked|gemm_shared|im2col_kernel|conv_direct|conv_pack|pool_kernel|multi_tensor_sgd_kernel|conv_dinput_gather)\b')
    for database in sorted(a.raw.glob('*.sqlite')):
        name=database.stem;folder=a.analysis/name
        command=[sys.executable,str(old),'--input',str(database),'--output',str(folder),'--label',name]
        with (a.analysis/(name+'.log')).open('x') as f:done=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT)
        with (a.analysis/'commands.jsonl').open('a') as f:f.write(json.dumps(dict(command=command,returncode=done.returncode))+'\n')
        done.check_returncode()
        summary=json_file(folder/'profiling-summary.json');summaries.append(summary)
        assert summary['unmatched_kernel_api_count']==0
        kernels=csv_rows(folder/'kernel-events.csv');ranges=csv_rows(folder/'nvtx-events.csv')
        for k in kernels:k['ancestor_ids']=json.loads(k['nvtx_ids'])
        if name.startswith('trace-shapes-'):
            method=name.split('-')[-1];split=name[len('trace-shapes-'):-(len(method)+1)]
            for r in ranges:
                if not r['name'].startswith('OP/shape:'):continue
                case,variant,step=re.match(r'OP/shape:(\w+)\|method=(\w+)\|step=(\d+)',r['name']).groups()
                events=[k for k in kernels if int(r['sqlite_rowid']) in k['ancestor_ids']]
                row=dict(split=split,case=case,method=variant,step=int(step),kernels=len(events),
                    kernel_us=sum(float(k['duration_ms']) for k in events)*1000,
                    host_nvtx_us=float(r['duration_ms'])*1000)
                for cat in ('Gather','Copy','Col2im','GEMM','other'):
                    selected=[k for k in events if category(k['name'])==cat]
                    row[cat+'_count']=len(selected);row[cat+'_us']=sum(float(k['duration_ms']) for k in selected)*1000
                shapes.append(row)
            continue
        model,method=name[len('trace-training-'):].split('-')
        tag=dict(model=model,method=method)
        phase=csv_rows(folder/'phase-summary.csv');phases.extend(dict(r,**tag) for r in phase)
        if method=='A':assert not any(native.search(k['name']) for k in kernels)
        if method in 'DFG':assert any('gemm_shared' in k['name'] for k in kernels)
        if method!='E':
            instrumentation=json_file(a.raw/name/'instrumentation.json')
            memory.extend(dict(r,**tag) for r in instrumentation['memory'])
            choices.extend(dict(r,**tag) for r in instrumentation['dispatch_decisions'])
        groups=defaultdict(list)
        for k in kernels:
            if k['phase']=='backward':groups[category(k['name'])].append(k)
        for cat,events in groups.items():categories.append(dict(**tag,category=cat,kernels=len(events),kernel_us=sum(float(k['duration_ms']) for k in events)*1000))
        for r in ranges:
            if r['phase']=='backward' and r['name'].startswith('OP/part:Conv2d.'):
                match=re.match(r'OP/part:Conv2d\.(\w+)\|layer=(\w+)',r['name'])
                events=[k for k in kernels if int(r['sqlite_rowid']) in k['ancestor_ids']]
                parts.append(dict(**tag,part=match[1],layer=match[2],nvtx_id=r['sqlite_rowid'],
                    host_us=float(r['duration_ms'])*1000,kernels=len(events),kernel_us=sum(float(k['duration_ms']) for k in events)*1000))
            # Worker-thread PyTorch kernels have a separate, explicit phase rule;
            # do not plot a partial same-thread subset as its entire backward.
            if r['name']=='backward' and method!='E':
                events=[k for k in kernels if int(r['sqlite_rowid']) in k['ancestor_ids']]
                span=max(int(k['end']) for k in events)-min(int(k['start']) for k in events)
                active=interval_union(events)
                windows.append(dict(**tag,nvtx_id=r['sqlite_rowid'],start_ns=int(r['start']),end_ns=int(r['end']),
                    host_us=float(r['duration_ms'])*1000,kernel_span_us=span/1000,kernel_union_us=active/1000,
                    between_own_kernels_us=(span-active)/1000,scope='own submitted work; not SM utilization'))
    for model in ('donkey','spatial'):
        base=a.raw/('trace-training-'+model+'-A')
        initial,inputs,weights=[arrays(base/n) for n in ('initial_weights.npz','comparison_inputs.npz','final_weights.npz')]
        loss=np.array([r['loss'] for r in json_lines(base/'losses.jsonl')])
        for method in 'ADFGE':
            folder=a.raw/('trace-training-'+model+'-'+method)
            check=dict(model=model,method=method,initial_equal=identical(arrays(folder/'initial_weights.npz'),initial),
                inputs_equal=identical(arrays(folder/'comparison_inputs.npz'),inputs))
            checks.append(check);assert check['initial_equal'] and check['inputs_equal']
            actual=arrays(folder/'final_weights.npz')
            for key in list(actual)+['loss']:
                value=np.array([r['loss'] for r in json_lines(folder/'losses.jsonl')]) if key=='loss' else actual[key]
                expected=loss if key=='loss' else weights[key]
                errors.append(dict(model=model,method=method,tensor=key,**difference(value,expected)))
        for method in 'DFG':
            expected=0 if method=='D' or method=='G' and model=='donkey' else 16 if method=='G' else 64 if model=='donkey' else 32
            got=sum(r['kernels'] for r in categories if (r['model'],r['method'],r['category'])==(model,method,'Gather'))
            assert got==expected,(model,method,got,expected)
        cc=[r for r in choices if r['model']==model and r['phase']=='backward']
        assert len(cc)==(64 if model=='donkey' else 32)
        assert sum(r['selected']=='fused' for r in cc)==(0 if model=='donkey' else 16)
    for name,data in [('trace-phases',phases),('kernel-categories',categories),('gradient-components',parts),
                      ('backward-windows',windows),('allocation-requests',memory),('shape-kernels',shapes),
                      ('dispatch-decisions',choices),('trace-errors',errors)]:write_csv(a.output/(name+'.csv'),data)
    (a.output/'attribution-checks.json').write_text(json.dumps(dict(traces=summaries,training_array_checks=checks),indent=2)+'\n')
    (a.output/'raw-location.json').write_text(json.dumps(dict(raw=str(a.raw.resolve()),analysis=str(a.analysis.resolve())),indent=2)+'\n')
    result=dict(status='PASSED' if all(r['passed'] for r in errors) else 'FAILED',traces=len(summaries),
        unmatched_kernel_api_count=sum(r['unmatched_kernel_api_count'] for r in summaries),
        numerical_max_abs=max(r['max_abs'] for r in errors),independent_timing='Nsight only, never mixed with throughput',
        auto_donkey='64 original / 0 gather',auto_spatial='16 original / 16 gather')
    (a.output/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    assert result['status']=='PASSED'


if __name__=='__main__':main()
