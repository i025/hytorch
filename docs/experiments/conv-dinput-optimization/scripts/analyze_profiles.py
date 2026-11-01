"""CUPTI correlation attribution, including three pure-kernel repeats."""
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
from analyze_fused_results import arrays,identical,difference,statistics,write_csv,json_file,json_lines,csv_rows
sys.path.insert(0,str(ROOT/'docs/experiments/conv-backward-fusion/scripts'))
from analyze_results import interval_union
from cases import cases


def category(name):
    return ('gather_opt' if 'conv_dinput_gather_opt' in name else 'gather_old' if 'conv_dinput_gather' in name
            else 'Copy' if 'cupy_copy' in name else 'Col2im' if name.startswith('col2im')
            else 'GEMM' if 'gemm' in name.lower() else 'other')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--micro',type=Path,required=True);p.add_argument('--training',type=Path,required=True)
    p.add_argument('--analysis',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--partial',action='store_true',help='Analyze only completed exports; explicitly report incomplete acceptance')
    p.add_argument('--kernel-repeats',type=int,default=3)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);a.analysis.mkdir(parents=True,exist_ok=False)
    assert json_file(a.training/'completion.json')['status']=='COMPLETED'
    if not a.partial:
        assert a.kernel_repeats==3 and json_file(a.micro/'completion.json')['status']=='COMPLETED'
    old=ROOT/'docs/experiments/donkeynet-backward-profiling/scripts/analyze_trace.py'
    phases,categories,parts,windows,memory,micros,errors,checks=[],[],[],[],[],[],[],[];summaries=[]
    native=re.compile(r'\b(gemm_basic|gemm_blocked|gemm_shared|im2col_kernel|conv_direct|conv_pack|pool_kernel|multi_tensor_sgd_kernel|conv_dinput_gather|conv_dinput_gather_opt)\b')
    databases=list(a.micro.glob('*.sqlite'))+list(a.training.glob('*.sqlite'))
    if a.partial:
        completed={r['name'][:-7] for r in json_file(a.micro/'commands.json') if r['name'].endswith('-export') and r.get('returncode')==0}
        databases=[d for d in databases if d.parent==a.training or d.stem in completed and
            (not d.stem.startswith('kernel-') or int(d.stem.rsplit('-r',1)[1])<=a.kernel_repeats)]
    for database in sorted(databases):
        name=database.stem;folder=a.analysis/name
        command=[sys.executable,str(old),'--input',str(database),'--output',str(folder),'--label',name]
        with (a.analysis/(name+'.log')).open('x') as f:done=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT)
        with (a.analysis/'commands.jsonl').open('a') as f:f.write(json.dumps(dict(command=command,returncode=done.returncode))+'\n')
        done.check_returncode();summary=json_file(folder/'profiling-summary.json');summaries.append(summary)
        assert summary['unmatched_kernel_api_count']==0
        kernels=csv_rows(folder/'kernel-events.csv');ranges=csv_rows(folder/'nvtx-events.csv')
        for k in kernels:k['ancestors']=json.loads(k['nvtx_ids'])
        if not name.startswith('trace-training-'):
            scope,split,rest=name.split('-',2);method,repeat=rest.rsplit('-r',1)
            for r in ranges:
                if not r['name'].startswith('OP/micro:'):continue
                label,case,step=re.match(r'OP/micro:(\w+)\|case=(\w+)\|step=(\d+)',r['name']).groups()
                assert label==scope
                events=[k for k in kernels if int(r['sqlite_rowid']) in k['ancestors']]
                row=dict(scope=scope,split=split,case=case,method=method,repeat=int(repeat),step=int(step),
                    kernels=len(events),kernel_us=sum(float(k['duration_ms']) for k in events)*1000,host_nvtx_us=float(r['duration_ms'])*1000)
                for cat in ('gather_old','gather_opt','Copy','Col2im','GEMM','other'):
                    selected=[k for k in events if category(k['name'])==cat]
                    row[cat+'_count']=len(selected);row[cat+'_us']=sum(float(k['duration_ms']) for k in selected)*1000
                if scope=='kernel':assert len(events)==1 and category(events[0]['name'])==('gather_opt' if method=='fused_opt' else 'gather_old')
                micros.append(row)
            continue
        model,method=name[len('trace-training-'):].split('-');tag=dict(model=model,method=method)
        phases.extend(dict(r,**tag) for r in csv_rows(folder/'phase-summary.csv'))
        if method=='A':assert not any(native.search(k['name']) for k in kernels)
        else:
            assert any('gemm_shared' in k['name'] for k in kernels)
            memory.extend(dict(r,**tag) for r in json_file(a.training/name/'instrumentation.json')['memory'])
        groups=defaultdict(list)
        for k in kernels:
            if k['phase']=='backward':groups[category(k['name'])].append(k)
        for cat,events in groups.items():categories.append(dict(**tag,category=cat,kernels=len(events),kernel_us=sum(float(k['duration_ms']) for k in events)*1000))
        for r in ranges:
            if r['phase']=='backward' and r['name'].startswith('OP/part:Conv2d.'):
                match=re.match(r'OP/part:Conv2d\.(\w+)\|layer=(\w+)',r['name'])
                events=[k for k in kernels if int(r['sqlite_rowid']) in k['ancestors']]
                parts.append(dict(**tag,part=match[1],layer=match[2],nvtx_id=r['sqlite_rowid'],host_us=float(r['duration_ms'])*1000,
                    kernels=len(events),kernel_us=sum(float(k['duration_ms']) for k in events)*1000))
            if r['name']=='backward':
                events=[k for k in kernels if int(r['sqlite_rowid']) in k['ancestors']]
                span=max(int(k['end']) for k in events)-min(int(k['start']) for k in events);active=interval_union(events)
                windows.append(dict(**tag,nvtx_id=r['sqlite_rowid'],start_ns=int(r['start']),end_ns=int(r['end']),host_us=float(r['duration_ms'])*1000,
                    kernel_span_us=span/1000,kernel_union_us=active/1000,between_own_kernels_us=(span-active)/1000))
    for model in ('donkey','spatial'):
        base=a.training/('trace-training-'+model+'-A')
        initial,inputs,weights=[arrays(base/n) for n in ('initial_weights.npz','comparison_inputs.npz','final_weights.npz')]
        loss=np.array([r['loss'] for r in json_lines(base/'losses.jsonl')])
        for method in 'AFH':
            path=a.training/('trace-training-'+model+'-'+method)
            check=dict(model=model,method=method,initial_equal=identical(arrays(path/'initial_weights.npz'),initial),inputs_equal=identical(arrays(path/'comparison_inputs.npz'),inputs))
            checks.append(check);assert check['initial_equal'] and check['inputs_equal']
            actual=arrays(path/'final_weights.npz')
            for key in list(actual)+['loss']:
                value=np.array([r['loss'] for r in json_lines(path/'losses.jsonl')]) if key=='loss' else actual[key]
                errors.append(dict(model=model,method=method,tensor=key,**difference(value,loss if key=='loss' else weights[key])))
            if method!='A':
                cat='gather_opt' if method=='H' else 'gather_old'
                count=sum(r['kernels'] for r in categories if (r['model'],r['method'],r['category'])==(model,method,cat))
                assert count==(64 if model=='donkey' else 32)
                assert not any(r['model']==model and r['method']==method and r['layer']=='conv1' and r['part']=='dInput' for r in parts)
    processes=[];aggregate=[];comparisons=[]
    for case in cases('history')+cases('fresh'):
        for method in ('fused','fused_opt'):
            means=[]
            for repeat in range(1,a.kernel_repeats+1):
                values=[r['kernel_us'] for r in micros if (r['scope'],r['case'],r['method'],r['repeat'])==('kernel',case['case'],method,repeat)]
                assert len(values)==30
                stats=statistics(values);means.append(stats['mean']);processes.append(dict(case=case['case'],method=method,repeat=repeat,**stats))
            aggregate.append(dict(case=case['case'],split='fresh' if case['case'].startswith('o') else 'history',method=method,**statistics(means)))
        old,new=aggregate[-2:]
        comparisons.append(dict(case=case['case'],split=old['split'],old_kernel_us=old['mean'],opt_kernel_us=new['mean'],
            old_std=old['std'],opt_std=new['std'],gpu_old_over_opt=old['mean']/new['mean'],
            separated_5pct=bool(old['mean']/new['mean']>=1.05 and new['maximum']<old['minimum'])))
    for name,data in [('trace-phases',phases),('kernel-categories',categories),('gradient-components',parts),('backward-windows',windows),
        ('allocation-requests',memory),('micro-kernels',micros),('pure-kernel-processes',processes),('pure-kernel-summary',aggregate),
        ('pure-kernel-comparisons',comparisons),('trace-errors',errors)]:write_csv(a.output/(name+'.csv'),data)
    (a.output/'attribution-checks.json').write_text(json.dumps(dict(traces=summaries,training_array_checks=checks),indent=2)+'\n')
    (a.output/'raw-location.json').write_text(json.dumps(dict(micro=str(a.micro.resolve()),training=str(a.training.resolve()),analysis=str(a.analysis.resolve())),indent=2)+'\n')
    numerical=all(r['passed'] for r in errors)
    result=dict(status='PARTIAL' if a.partial else 'PASSED' if numerical else 'FAILED',numerical_passed=numerical,
        kernel_repeats=a.kernel_repeats,required_kernel_repeats=3,traces=len(summaries),unmatched_kernel_api_count=0,
        pure_kernel_faster=sum(r['gpu_old_over_opt']>1 for r in comparisons),pure_kernel_slower=sum(r['gpu_old_over_opt']<1 for r in comparisons),
        pure_kernel_separated_5pct=sum(r['separated_5pct'] for r in comparisons),trace_max_abs=max(r['max_abs'] for r in errors),
        scope='Nsight instrumented CUPTI activity; never substitute for uninstrumented training throughput')
    (a.output/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));assert numerical


if __name__=='__main__':main()
